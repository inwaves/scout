from __future__ import annotations

import json
import logging
import time
from typing import Any, Mapping, Sequence

from .config import ProfileConfig, ScoringConfig
from .models import Paper, ScoredPaper, SummarizedPaper
from .scorer import (
    _build_anthropic_client,
    _chunked,
    _extract_json_payload,
    _extract_message_text,
    _normalize_arxiv_id,
    _obj_get,
)

LOGGER = logging.getLogger(__name__)


class SummarizationError(RuntimeError):
    """Raised when summary generation fails."""


class AnthropicSummarizer:
    """Generate digest-ready paper summaries with Claude."""

    def __init__(
        self,
        profile: ProfileConfig,
        scoring: ScoringConfig,
        api_key: str,
        logger: logging.Logger | None = None,
    ) -> None:
        if not api_key:
            raise SummarizationError(
                "Anthropic API key is required for summarization. "
                "Set anthropic_api_key in YAML or ANTHROPIC_API_KEY in environment."
            )
        self.profile = profile
        self.scoring = scoring
        self._logger = logger or LOGGER
        self._client = _build_anthropic_client(api_key)
        self._system_prompt = self._build_system_prompt()
        self._max_retries = 3

    def summarize_papers(
        self,
        papers: Sequence[Paper],
        scored_lookup: Mapping[str, ScoredPaper] | None = None,
    ) -> list[SummarizedPaper]:
        if not papers:
            return []

        batches = list(_chunked(papers, max(1, self.scoring.batch_size)))
        lookup = scored_lookup or {}

        if self.scoring.use_batch_api:
            try:
                by_id = self._summarize_with_batch_api(batches, lookup)
            except Exception:
                self._logger.exception(
                    "Anthropic batch API summarization failed; falling back to synchronous calls."
                )
                by_id = self._summarize_with_standard_api(batches, lookup)
        else:
            by_id = self._summarize_with_standard_api(batches, lookup)

        ordered: list[SummarizedPaper] = []
        for paper in papers:
            key = _normalize_arxiv_id(paper.arxiv_id)
            item = by_id.get(key)
            if item is not None:
                ordered.append(item)

        self._logger.info("Generated summaries for %d/%d papers.", len(ordered), len(papers))
        return ordered

    def _summarize_with_standard_api(
        self,
        batches: Sequence[list[Paper]],
        scored_lookup: Mapping[str, ScoredPaper],
    ) -> dict[str, SummarizedPaper]:
        results: dict[str, SummarizedPaper] = {}

        for index, batch in enumerate(batches, start=1):
            prompt = self._build_user_prompt(batch, scored_lookup)
            expected_ids = {_normalize_arxiv_id(paper.arxiv_id) for paper in batch}
            self._logger.info(
                "Summarizing batch %d/%d (%d papers) via standard API.",
                index,
                len(batches),
                len(batch),
            )

            try:
                text = self._call_messages_api(prompt)
                parsed = self._parse_summary_response(text, expected_ids)
                for item in parsed:
                    results[_normalize_arxiv_id(item.arxiv_id)] = item
            except Exception:
                self._logger.exception("Summarization failed for batch %d; continuing.", index)

        return results

    def _summarize_with_batch_api(
        self,
        batches: Sequence[list[Paper]],
        scored_lookup: Mapping[str, ScoredPaper],
    ) -> dict[str, SummarizedPaper]:
        requests: list[dict[str, Any]] = []
        expected_by_custom_id: dict[str, set[str]] = {}

        for index, batch in enumerate(batches):
            custom_id = f"summary-{index}"
            expected_by_custom_id[custom_id] = {
                _normalize_arxiv_id(paper.arxiv_id) for paper in batch
            }
            requests.append(
                {
                    "custom_id": custom_id,
                    "params": {
                        "model": self.scoring.model,
                        "system": self._system_prompt,
                        "temperature": self.scoring.temperature,
                        "max_tokens": self.scoring.max_output_tokens,
                        "messages": [
                            {"role": "user", "content": self._build_user_prompt(batch, scored_lookup)}
                        ],
                    },
                }
            )

        if not requests:
            return {}

        self._logger.info("Submitting Anthropic batch with %d summary requests.", len(requests))
        batch_job = self._client.messages.batches.create(requests=requests)
        batch_id = str(_obj_get(batch_job, "id") or "")
        if not batch_id:
            raise SummarizationError("Anthropic summary batch did not return an ID.")

        self._wait_for_batch_completion(batch_id)

        results: dict[str, SummarizedPaper] = {}
        for item in self._iterate_batch_results(batch_id):
            custom_id = str(_obj_get(item, "custom_id") or "")
            expected_ids = expected_by_custom_id.get(custom_id, set())
            if not expected_ids:
                continue

            result_payload = _obj_get(item, "result")
            result_type = str(_obj_get(result_payload, "type") or "").lower()
            if result_type != "succeeded":
                self._logger.warning(
                    "Summary batch item %s did not succeed (status=%s).",
                    custom_id,
                    result_type or "unknown",
                )
                continue

            message = _obj_get(result_payload, "message")
            text = _extract_message_text(message)
            if not text:
                continue

            try:
                parsed = self._parse_summary_response(text, expected_ids)
                for summary in parsed:
                    results[_normalize_arxiv_id(summary.arxiv_id)] = summary
            except Exception:
                self._logger.exception(
                    "Failed to parse summary output for batch request %s.", custom_id
                )

        return results

    def _wait_for_batch_completion(self, batch_id: str) -> None:
        deadline = time.monotonic() + self.scoring.batch_poll_timeout_seconds

        while True:
            status_payload = self._client.messages.batches.retrieve(batch_id)
            status = str(
                _obj_get(status_payload, "processing_status")
                or _obj_get(status_payload, "status")
                or ""
            ).lower()

            if status == "ended":
                return
            if status in {"failed", "cancelled", "canceled", "expired"}:
                raise SummarizationError(
                    f"Anthropic summary batch {batch_id} ended with status '{status}'."
                )
            if time.monotonic() > deadline:
                raise SummarizationError(
                    f"Timed out waiting for Anthropic summary batch {batch_id} to complete."
                )

            self._logger.info("Anthropic summary batch %s status: %s", batch_id, status or "unknown")
            time.sleep(self.scoring.batch_poll_interval_seconds)

    def _iterate_batch_results(self, batch_id: str):
        results_payload = self._client.messages.batches.results(batch_id)

        if hasattr(results_payload, "__iter__"):
            for item in results_payload:
                yield item
            return

        data = _obj_get(results_payload, "data", [])
        if isinstance(data, list):
            for item in data:
                yield item

    def _call_messages_api(self, prompt: str) -> str:
        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._client.messages.create(
                    model=self.scoring.model,
                    system=self._system_prompt,
                    temperature=self.scoring.temperature,
                    max_tokens=self.scoring.max_output_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = _extract_message_text(response)
                if not text:
                    raise SummarizationError("Claude returned empty summary response.")
                return text
            except Exception as exc:
                last_error = exc
                if attempt >= self._max_retries:
                    break
                sleep_seconds = min(30.0, 2.0**attempt)
                self._logger.warning(
                    "Anthropic summary call failed (%d/%d): %s. Retrying in %.1fs.",
                    attempt,
                    self._max_retries,
                    exc,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)

        raise SummarizationError(f"Anthropic summary request failed: {last_error}")

    def _build_system_prompt(self) -> str:
        return (
            "You write concise, technically accurate summaries of research papers.\n"
            "Summaries are for a researcher with this profile:\n"
            f"{self.profile.name}\n\n"
            f"{self.profile.description}\n\n"
            f"Each summary should be around {self.scoring.summary_words} words and include:\n"
            "1) what the paper does,\n"
            "2) why it matters to this profile,\n"
            "3) key technical contribution.\n\n"
            "Output requirements:\n"
            '- Return ONLY valid JSON.\n'
            '- Use schema: {"papers": [{"arxiv_id": "...", "summary": "..."}]}.\n'
            "- No markdown, no commentary, no extra keys."
        )

    def _build_user_prompt(
        self,
        batch: Sequence[Paper],
        scored_lookup: Mapping[str, ScoredPaper],
    ) -> str:
        payload_items: list[dict[str, Any]] = []

        for paper in batch:
            key = _normalize_arxiv_id(paper.arxiv_id)
            scored = scored_lookup.get(key) or scored_lookup.get(paper.arxiv_id)

            item: dict[str, Any] = {
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "abstract": paper.abstract,
                "authors": paper.authors,
                "categories": paper.categories,
            }

            if scored is not None:
                item["relevance_score"] = scored.relevance_score
                item["scoring_rationale"] = scored.rationale
                item["novelty_signal"] = scored.novelty_signal

            payload_items.append(item)

        return (
            "Generate one summary per paper.\n"
            "Return JSON only in the required schema.\n\n"
            f"{json.dumps({'papers': payload_items}, ensure_ascii=False, indent=2)}"
        )

    def _parse_summary_response(
        self,
        response_text: str,
        expected_ids: set[str],
    ) -> list[SummarizedPaper]:
        payload = _extract_json_payload(response_text)

        if isinstance(payload, dict):
            raw_items = payload.get("papers", [])
        elif isinstance(payload, list):
            raw_items = payload
        else:
            raise SummarizationError("Summary response JSON must be an object or array.")

        if not isinstance(raw_items, list):
            raise SummarizationError("Summary response field 'papers' must be a list.")

        parsed: list[SummarizedPaper] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue

            arxiv_id = _normalize_arxiv_id(str(item.get("arxiv_id", "")).strip())
            if not arxiv_id or arxiv_id not in expected_ids:
                continue

            summary = str(item.get("summary", "")).strip()
            if not summary:
                continue

            parsed.append(SummarizedPaper(arxiv_id=arxiv_id, summary=summary))

        return parsed