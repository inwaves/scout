from __future__ import annotations

import json
import logging
import urllib.parse
from dataclasses import asdict
from typing import Any

from .config import AnalysisConfig, ProfileConfig
from .costs import CostTracker
from .knowledge_base import KnowledgeBase
from .models import (
    DeepReadBreakdown,
    DeepReadEntry,
    DeepReadResult,
    Paper,
    ScoredPaper,
)
from .scorer import _extract_json_payload
from .semantic_scholar import SemanticScholarClient

LOGGER = logging.getLogger(__name__)


class DeepReadError(RuntimeError):
    """Raised when deep-read analysis cannot be completed."""


class DeepReadAgent:
    def __init__(
        self,
        api_key: str,
        profile: ProfileConfig,
        analysis: AnalysisConfig,
        knowledge_base: KnowledgeBase,
        s2_client: SemanticScholarClient,
        logger: logging.Logger | None = None,
        cost_tracker: CostTracker | None = None,
    ) -> None:
        if not api_key:
            raise DeepReadError(
                "Anthropic API key is required for deep reads. "
                "Set anthropic_api_key in YAML or ANTHROPIC_API_KEY in environment."
            )

        self.profile = profile
        self.analysis = analysis
        self.knowledge_base = knowledge_base
        self.s2_client = s2_client
        self._logger = logger or LOGGER
        self._client = _build_anthropic_client(api_key)
        self._tools = _build_tools()
        self._cost_tracker = cost_tracker

    def analyze_paper(self, paper: Paper, scored: ScoredPaper) -> DeepReadResult:
        messages: list[dict[str, Any]] = [self._build_initial_user_message(paper, scored)]

        for turn in range(1, self.analysis.max_agent_turns + 1):
            response = self._create_message(messages)
            blocks = _extract_content_blocks(response)

            if turn == 1:
                _strip_document_blocks(messages)

            produce_payload = _extract_produce_payload(blocks)
            if produce_payload is not None:
                return self._build_result_from_payload(paper, scored, produce_payload)

            tool_uses = [block for block in blocks if _block_type(block) == "tool_use"]
            if tool_uses:
                tool_results: list[dict[str, Any]] = []
                for block in tool_uses:
                    tool_name = _as_string(_obj_get(block, "name"))
                    tool_use_id = _as_string(_obj_get(block, "id"))
                    tool_input = _obj_get(block, "input", {})
                    if not isinstance(tool_input, dict):
                        tool_input = {}

                    try:
                        payload = self._execute_tool(tool_name, tool_input, paper)
                        content = json.dumps(payload, ensure_ascii=False)
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": content,
                            }
                        )
                    except Exception as exc:
                        self._logger.exception("Tool execution failed (%s): %s", tool_name, exc)
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": json.dumps({"error": str(exc)}, ensure_ascii=False),
                                "is_error": True,
                            }
                        )

                messages.append({"role": "assistant", "content": _serialize_blocks(blocks)})
                messages.append({"role": "user", "content": tool_results})
                continue

            text_output = _extract_text(blocks)
            if text_output:
                self._logger.warning(
                    "Deep read model returned plain text without produce_deep_read tool on turn %d; using fallback parse.",
                    turn,
                )
                return self._build_result_from_text(paper, scored, text_output)

        raise DeepReadError(
            f"Deep read agent exceeded max_agent_turns={self.analysis.max_agent_turns} for {paper.arxiv_id}."
        )

    def _create_message(self, messages: list[dict[str, Any]]) -> Any:
        payload = {
            "model": self.analysis.model,
            "system": self._build_system_prompt(),
            "temperature": self.analysis.temperature,
            "max_tokens": self.analysis.max_output_tokens,
            "messages": messages,
            "tools": self._tools,
        }

        try:
            response = self._client.messages.create(
                **payload,
                extra_headers={"anthropic-beta": "pdfs-2024-09-25"},
            )
        except TypeError:
            try:
                response = self._client.messages.create(**payload)
            except Exception as exc:
                raise DeepReadError(f"Anthropic deep read request failed: {exc}") from exc
        except Exception as exc:
            raise DeepReadError(f"Anthropic deep read request failed: {exc}") from exc

        self._record_response_usage(response)
        return response

    def _record_response_usage(self, response: Any) -> None:
        if self._cost_tracker is None:
            return

        usage = _obj_get(response, "usage")
        input_tokens = _coerce_int(_obj_get(usage, "input_tokens", 0))
        output_tokens = _coerce_int(_obj_get(usage, "output_tokens", 0))
        self._cost_tracker.record(
            model=self.analysis.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def _build_system_prompt(self) -> str:
        return (
            "You are Scout's deep-read research agent.\n"
            "Read the paper PDF when available, or analyze based on the provided metadata if the item is a web post without a PDF.\n\n"
            "CRITICAL COST CONSTRAINT — BE CONCISE:\n"
            "- Each section of the breakdown must be 2-3 sentences MAX.\n"
            "- Total output should be ~300 words. Do NOT write essay-length sections.\n"
            "- Focus on the main body of the paper or post. If a PDF is available, prioritize the first 15 pages and skip appendices, supplementary material, and bibliography.\n"
            "- Prefer calling produce_deep_read on your FIRST turn if possible.\n"
            "- Only use tools (kb_lookup, s2_paper_info) if you genuinely need external context. Do not call tools speculatively.\n\n"
            "You may call tools when helpful:\n"
            "- kb_lookup: check prior papers in Scout's knowledge base by topic.\n"
            "- s2_paper_info: retrieve Semantic Scholar metadata, references, and affiliations.\n"
            "- s2_reference_details: inspect a referenced paper.\n\n"
            "When you are ready, CALL produce_deep_read exactly once with the complete structured output.\n"
            "Do not return final prose only; finalize via produce_deep_read.\n\n"
            "IMPORTANT for builds_on: include arXiv IDs where you know them, formatted as:\n"
            "  'Author et al. YEAR — Short description (arXiv:XXXX.XXXXX)'\n"
            "If you do not know the arXiv ID, just use the author/year/description format.\n\n"
            f"Research profile: {self.profile.name}\n"
            f"Profile description:\n{self.profile.description}\n\n"
            f"Scoring rubric:\n{self.profile.scoring_rubric}\n"
        )

    def _build_initial_user_message(self, paper: Paper, scored: ScoredPaper) -> dict[str, Any]:
        metadata = {
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "authors": paper.authors,
            "categories": paper.categories,
            "published": paper.published.isoformat(),
            "url": paper.url,
            "pdf_url": paper.pdf_url,
            "abstract": paper.abstract,
            "relevance_score": scored.relevance_score,
            "scoring_rationale": scored.rationale,
            "novelty_signal": scored.novelty_signal,
        }

        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Analyze this paper in depth for the profile above.\n"
                    "Use tools as needed and finalize with produce_deep_read.\n"
                    "Paper metadata:\n"
                    f"{json.dumps(metadata, ensure_ascii=False, indent=2)}"
                ),
            }
        ]

        if _looks_like_pdf_url(paper.pdf_url):
            content.append(
                {
                    "type": "document",
                    "source": {
                        "type": "url",
                        "url": paper.pdf_url,
                    },
                }
            )
        else:
            abstract_text = paper.abstract.strip() or (
                "No abstract or page description was available for this web post."
            )
            content.append(
                {
                    "type": "text",
                    "text": (
                        "This paper is a web post. Full text is not available as a PDF.\n"
                        "Analyze based on the abstract and metadata above.\n\n"
                        "Web post abstract/description:\n"
                        f"{abstract_text}"
                    ),
                }
            )

        content.append(
            {
                "type": "text",
                "text": "Begin your deep-read analysis now.",
            }
        )

        return {
            "role": "user",
            "content": content,
        }

    def _execute_tool(self, tool_name: str, tool_input: dict[str, Any], paper: Paper) -> dict[str, Any]:
        if tool_name == "kb_lookup":
            topics = _coerce_string_list(tool_input.get("topics"))
            records = self.knowledge_base.lookup_topics(topics)
            return {
                "topics": topics,
                "count": len(records),
                "matches": [asdict(record) for record in records[:50]],
            }

        if tool_name == "s2_paper_info":
            arxiv_id = _as_string(tool_input.get("arxiv_id")).strip()
            query = _as_string(tool_input.get("query")).strip()

            if not arxiv_id and query:
                searched = self.s2_client.search_paper(query)
                if searched is not None:
                    return {"paper": asdict(searched), "references": [], "author_affiliations": {}}

            target_arxiv_id = arxiv_id or paper.arxiv_id
            s2_paper = self.s2_client.get_paper(target_arxiv_id)
            references = self.s2_client.get_references(target_arxiv_id)
            author_affiliations = self.s2_client.get_author_affiliations(target_arxiv_id)

            return {
                "paper": asdict(s2_paper) if s2_paper is not None else None,
                "references": [asdict(reference) for reference in references[:30]],
                "author_affiliations": author_affiliations,
            }

        if tool_name == "s2_reference_details":
            paper_id = _as_string(tool_input.get("paper_id")).strip()
            if not paper_id:
                return {"error": "paper_id is required."}

            reference = self.s2_client.get_reference_details(paper_id)
            if reference is None:
                return {"reference": None}
            return {"reference": asdict(reference)}

        if tool_name == "produce_deep_read":
            return {"status": "ok"}

        raise DeepReadError(f"Unknown tool requested by model: {tool_name}")

    def _build_result_from_payload(
        self,
        paper: Paper,
        scored: ScoredPaper,
        payload: dict[str, Any],
    ) -> DeepReadResult:
        if isinstance(payload.get("breakdown"), dict):
            merged = dict(payload.get("breakdown", {}))
            for key in ("topics", "key_findings", "builds_on", "watchlist_match"):
                if key in payload:
                    merged[key] = payload[key]
            payload = merged

        default_tldr = [
            _compact(scored.rationale, 200),
            "See methodology and results for details.",
            "Relevance assessed for the configured profile.",
        ]

        tldr = _coerce_string_list(payload.get("tldr"))
        if not tldr:
            tldr = default_tldr
        while len(tldr) < 3:
            tldr.append(default_tldr[min(len(tldr), len(default_tldr) - 1)])
        tldr = tldr[:3]

        triage = _non_empty(payload.get("triage")) or "TL;DR captures it."
        motivation = _non_empty(payload.get("motivation")) or "Not provided."
        hypothesis = _non_empty(payload.get("hypothesis")) or "Not provided."
        methodology = _non_empty(payload.get("methodology")) or "Not provided."
        results = _non_empty(payload.get("results")) or "Not provided."
        interpretation = _non_empty(payload.get("interpretation")) or "Not provided."
        context = _non_empty(payload.get("context")) or "Not provided."
        limitations = _non_empty(payload.get("limitations")) or "Not provided."
        relevance = _non_empty(payload.get("relevance")) or _compact(scored.rationale, 220)

        breakdown = DeepReadBreakdown(
            triage=triage,
            tldr=tldr,
            motivation=motivation,
            hypothesis=hypothesis,
            methodology=methodology,
            results=results,
            interpretation=interpretation,
            context=context,
            limitations=limitations,
            relevance=relevance,
        )

        entry = DeepReadEntry(
            paper=paper,
            relevance_score=scored.relevance_score,
            rationale=scored.rationale,
            novelty_signal=scored.novelty_signal,
            breakdown=breakdown,
            watchlist_match=_non_empty(payload.get("watchlist_match")),
        )

        topics = _dedupe(_coerce_string_list(payload.get("topics")))
        key_findings = _dedupe(_coerce_string_list(payload.get("key_findings"))) or tldr
        builds_on = _dedupe(_coerce_string_list(payload.get("builds_on")))

        return DeepReadResult(
            entry=entry,
            topics=topics,
            key_findings=key_findings,
            builds_on=builds_on,
        )

    def _build_result_from_text(
        self,
        paper: Paper,
        scored: ScoredPaper,
        text_output: str,
    ) -> DeepReadResult:
        try:
            parsed = _extract_json_payload(text_output)
            if isinstance(parsed, dict):
                return self._build_result_from_payload(paper, scored, parsed)
        except Exception:
            pass

        lines = [
            line.strip().lstrip("-•").strip()
            for line in text_output.splitlines()
            if line.strip()
        ]
        tldr = lines[:3]
        while len(tldr) < 3:
            tldr.append(
                [
                    _compact(scored.rationale, 200),
                    "Model returned plain text fallback output.",
                    "See paper links for full details.",
                ][len(tldr)]
            )

        triage = (
            "Read the full paper — high relevance and incomplete structured output."
            if scored.relevance_score >= 8.0
            else "TL;DR captures it — structured output fallback."
        )

        breakdown = DeepReadBreakdown(
            triage=triage,
            tldr=tldr[:3],
            motivation=_compact(scored.rationale, 260),
            hypothesis="Unable to parse structured hypothesis from fallback output.",
            methodology=_compact(text_output, 900),
            results="Unable to parse structured results from fallback output.",
            interpretation="Unable to parse structured interpretation from fallback output.",
            context="No additional structured context extracted.",
            limitations="Fallback parsing used because produce_deep_read was not called.",
            relevance=_compact(scored.rationale, 220),
        )

        entry = DeepReadEntry(
            paper=paper,
            relevance_score=scored.relevance_score,
            rationale=scored.rationale,
            novelty_signal=scored.novelty_signal,
            breakdown=breakdown,
        )

        return DeepReadResult(
            entry=entry,
            topics=list(paper.categories[:3]),
            key_findings=tldr[:3],
            builds_on=[],
        )


def _build_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "kb_lookup",
            "description": "Look up related papers from Scout's knowledge base by topic keywords.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "topics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Topic keywords to search.",
                    }
                },
                "required": ["topics"],
            },
        },
        {
            "name": "s2_paper_info",
            "description": (
                "Fetch Semantic Scholar metadata for a paper (citation count, references, affiliations)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "arxiv_id": {"type": "string"},
                    "query": {"type": "string"},
                },
            },
        },
        {
            "name": "s2_reference_details",
            "description": "Fetch details for a specific referenced paper by Semantic Scholar paper ID.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string"},
                },
                "required": ["paper_id"],
            },
        },
        {
            "name": "produce_deep_read",
            "description": "Return the final structured deep-read breakdown once analysis is complete.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "triage": {"type": "string"},
                    "tldr": {"type": "array", "items": {"type": "string"}},
                    "motivation": {"type": "string"},
                    "hypothesis": {"type": "string"},
                    "methodology": {"type": "string"},
                    "results": {"type": "string"},
                    "interpretation": {"type": "string"},
                    "context": {"type": "string"},
                    "limitations": {"type": "string"},
                    "relevance": {"type": "string"},
                    "topics": {"type": "array", "items": {"type": "string"}},
                    "key_findings": {"type": "array", "items": {"type": "string"}},
                    "builds_on": {"type": "array", "items": {"type": "string"}, "description": "Prior work this paper builds on. Include arXiv IDs where known, e.g. 'Perez et al. 2022 — Sycophancy (arXiv:2212.09251)'"},
                    "watchlist_match": {"type": "string"},
                },
                "required": [
                    "triage",
                    "tldr",
                    "motivation",
                    "hypothesis",
                    "methodology",
                    "results",
                    "interpretation",
                    "context",
                    "limitations",
                    "relevance",
                ],
            },
        },
    ]


def _build_anthropic_client(api_key: str) -> Any:
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise DeepReadError(
            "anthropic package is not installed. Install dependencies from requirements.txt."
        ) from exc

    return Anthropic(api_key=api_key)


def _extract_content_blocks(response: Any) -> list[Any]:
    content = _obj_get(response, "content")
    if isinstance(content, list):
        return list(content)
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [content]


def _strip_document_blocks(messages: list[dict[str, Any]]) -> None:
    """Remove PDF document blocks from the first user message to avoid re-billing.

    After the model has read the PDF on turn 1, subsequent turns do not need the
    full document. This replaces document blocks with a short text placeholder.
    """
    if not messages:
        return

    first_msg = messages[0]
    content = first_msg.get("content")
    if not isinstance(content, list):
        return

    new_content: list[dict[str, Any]] = []
    stripped = False
    for block in content:
        block_type = ""
        if isinstance(block, dict):
            block_type = block.get("type", "")
        else:
            block_type = str(getattr(block, "type", ""))

        if block_type == "document":
            if not stripped:
                new_content.append({
                    "type": "text",
                    "text": "[PDF was provided on turn 1 and has been read. Refer to your notes above.]",
                })
                stripped = True
        else:
            new_content.append(block)

    if stripped:
        first_msg["content"] = new_content


def _extract_produce_payload(blocks: list[Any]) -> dict[str, Any] | None:
    for block in blocks:
        if _block_type(block) != "tool_use":
            continue
        if _as_string(_obj_get(block, "name")) != "produce_deep_read":
            continue
        payload = _obj_get(block, "input", {})
        if isinstance(payload, dict):
            return payload
        return {}
    return None


def _serialize_blocks(blocks: list[Any]) -> list[dict[str, Any]]:
    return [_serialize_block(block) for block in blocks]


def _serialize_block(block: Any) -> dict[str, Any]:
    if isinstance(block, dict):
        return dict(block)

    block_type = _block_type(block)
    if block_type == "text":
        return {"type": "text", "text": _as_string(_obj_get(block, "text"))}
    if block_type == "tool_use":
        return {
            "type": "tool_use",
            "id": _as_string(_obj_get(block, "id")),
            "name": _as_string(_obj_get(block, "name")),
            "input": _obj_get(block, "input", {}),
        }

    if hasattr(block, "model_dump"):
        dumped = block.model_dump()
        if isinstance(dumped, dict):
            return dumped

    return {"type": block_type or "text", "text": _as_string(block)}


def _extract_text(blocks: list[Any]) -> str:
    parts: list[str] = []
    for block in blocks:
        if _block_type(block) != "text":
            continue
        text = _as_string(_obj_get(block, "text")).strip()
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def _obj_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _block_type(block: Any) -> str:
    return _as_string(_obj_get(block, "type")).strip().lower()


def _as_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _coerce_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        items = [line.strip() for line in value.splitlines() if line.strip()]
    else:
        return []

    cleaned: list[str] = []
    for item in items:
        text = _as_string(item).strip()
        if text:
            cleaned.append(text)
    return cleaned


def _non_empty(value: Any) -> str | None:
    text = _as_string(value).strip()
    return text or None


def _compact(value: str, max_chars: int) -> str:
    collapsed = " ".join(value.split())
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max_chars - 1].rstrip() + "…"


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _coerce_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def _looks_like_pdf_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(_as_string(url))
    return parsed.path.lower().endswith(".pdf")