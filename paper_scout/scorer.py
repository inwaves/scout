from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import replace
from typing import Any, Iterator, Sequence, TypeVar, cast

from .config import ProfileConfig, ScoringConfig
from .costs import CostTracker
from .models import NoveltySignal, Paper, ScoredPaper, VALID_NOVELTY_SIGNALS

LOGGER = logging.getLogger(__name__)
_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_ARXIV_VERSION_RE = re.compile(r"v\d+$")
_SURVEY_PAPER_RE = re.compile(r"\b(survey|review|taxonomy|overview|meta-analysis)\b", re.I)
T = TypeVar("T")


class ScoringError(RuntimeError):
    """Raised when paper scoring cannot be completed."""


class AnthropicScorer:
    """Scores papers against a user profile using Anthropic Claude."""

    def __init__(
        self,
        profile: ProfileConfig,
        scoring: ScoringConfig,
        api_key: str,
        logger: logging.Logger | None = None,
        cost_tracker: CostTracker | None = None,
    ) -> None:
        if not api_key:
            raise ScoringError(
                "Anthropic API key is required for scoring. "
                "Set anthropic_api_key in YAML or ANTHROPIC_API_KEY in environment."
            )
        self.profile = profile
        self.scoring = scoring
        self._logger = logger or LOGGER
        self._client = _build_anthropic_client(api_key)
        self._system_prompt = self._build_system_prompt()
        self._max_retries = 3
        self._cost_tracker = cost_tracker

    def score_papers(self, papers: Sequence[Paper]) -> list[ScoredPaper]:
        if not papers:
            return []

        batches = list(_chunked(papers, max(1, self.scoring.batch_size)))
        if self.scoring.use_batch_api:
            try:
                by_id = self._score_with_batch_api(batches)
            except Exception:
                self._logger.exception(
                    "Anthropic batch API scoring failed; falling back to synchronous calls."
                )
                by_id = self._score_with_standard_api(batches)
        else:
            by_id = self._score_with_standard_api(batches)

        ordered: list[ScoredPaper] = []
        for paper in papers:
            key = _normalize_arxiv_id(paper.arxiv_id)
            scored = by_id.get(key)
            if scored is not None:
                ordered.append(scored)

        self._logger.info("Scored %d/%d papers.", len(ordered), len(papers))
        return ordered

    def _score_with_standard_api(self, batches: Sequence[list[Paper]]) -> dict[str, ScoredPaper]:
        results: dict[str, ScoredPaper] = {}

        for index, batch in enumerate(batches, start=1):
            prompt = self._build_user_prompt(batch)
            expected_ids = {_normalize_arxiv_id(paper.arxiv_id) for paper in batch}
            papers_by_id = {_normalize_arxiv_id(paper.arxiv_id): paper for paper in batch}
            self._logger.info(
                "Scoring batch %d/%d (%d papers) via standard API.",
                index,
                len(batches),
                len(batch),
            )

            try:
                text = self._call_messages_api(prompt)
                parsed = self._parse_scoring_response(text, expected_ids)
                for scored in parsed:
                    normalized_id = _normalize_arxiv_id(scored.arxiv_id)
                    results[normalized_id] = _calibrate_scored_paper(
                        scored,
                        papers_by_id.get(normalized_id),
                    )
            except Exception:
                self._logger.exception("Scoring failed for batch %d; continuing.", index)

        return results

    def _score_with_batch_api(self, batches: Sequence[list[Paper]]) -> dict[str, ScoredPaper]:
        requests: list[dict[str, Any]] = []
        expected_by_custom_id: dict[str, set[str]] = {}
        papers_by_custom_id: dict[str, dict[str, Paper]] = {}

        for index, batch in enumerate(batches):
            custom_id = f"score-{index}"
            expected_by_custom_id[custom_id] = {
                _normalize_arxiv_id(paper.arxiv_id) for paper in batch
            }
            papers_by_custom_id[custom_id] = {
                _normalize_arxiv_id(paper.arxiv_id): paper for paper in batch
            }
            requests.append(
                {
                    "custom_id": custom_id,
                    "params": {
                        "model": self.scoring.model,
                        "system": self._system_prompt,
                        "temperature": self.scoring.temperature,
                        "max_tokens": self.scoring.max_output_tokens,
                        "messages": [{"role": "user", "content": self._build_user_prompt(batch)}],
                    },
                }
            )

        if not requests:
            return {}

        self._logger.info("Submitting Anthropic batch with %d scoring requests.", len(requests))
        batch_job = self._client.messages.batches.create(requests=requests)
        batch_id = _as_string(_obj_get(batch_job, "id"))
        if not batch_id:
            raise ScoringError("Anthropic batch create response did not include a batch ID.")

        self._wait_for_batch_completion(batch_id)

        results: dict[str, ScoredPaper] = {}
        for item in self._iterate_batch_results(batch_id):
            custom_id = _as_string(_obj_get(item, "custom_id"))
            expected_ids = expected_by_custom_id.get(custom_id, set())
            if not expected_ids:
                continue

            result_payload = _obj_get(item, "result")
            result_type = _as_string(_obj_get(result_payload, "type")).lower()
            if result_type != "succeeded":
                self._logger.warning(
                    "Batch request %s did not succeed (status=%s).",
                    custom_id,
                    result_type or "unknown",
                )
                continue

            message = _obj_get(result_payload, "message")
            self._record_response_usage(message)
            text = _extract_message_text(message)
            if not text:
                self._logger.warning("Batch request %s returned empty content.", custom_id)
                continue

            try:
                parsed = self._parse_scoring_response(text, expected_ids)
                for scored in parsed:
                    normalized_id = _normalize_arxiv_id(scored.arxiv_id)
                    results[normalized_id] = _calibrate_scored_paper(
                        scored,
                        papers_by_custom_id.get(custom_id, {}).get(normalized_id),
                    )
            except Exception:
                self._logger.exception(
                    "Failed to parse scoring output for batch request %s.", custom_id
                )

        return results

    def _wait_for_batch_completion(self, batch_id: str) -> None:
        deadline = time.monotonic() + self.scoring.batch_poll_timeout_seconds

        while True:
            status_payload = self._client.messages.batches.retrieve(batch_id)
            status = (
                _as_string(_obj_get(status_payload, "processing_status"))
                or _as_string(_obj_get(status_payload, "status"))
            ).lower()

            if status == "ended":
                return
            if status in {"failed", "cancelled", "canceled", "expired"}:
                raise ScoringError(f"Anthropic batch {batch_id} ended with status '{status}'.")
            if time.monotonic() > deadline:
                raise ScoringError(
                    f"Timed out waiting for Anthropic batch {batch_id} to complete."
                )

            self._logger.info("Anthropic batch %s status: %s", batch_id, status or "unknown")
            time.sleep(self.scoring.batch_poll_interval_seconds)

    def _iterate_batch_results(self, batch_id: str) -> Iterator[Any]:
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
                self._record_response_usage(response)
                text = _extract_message_text(response)
                if not text:
                    raise ScoringError("Claude response did not contain any text output.")
                return text
            except Exception as exc:
                last_error = exc
                if attempt >= self._max_retries:
                    break
                sleep_seconds = min(30.0, 2.0**attempt)
                self._logger.warning(
                    "Anthropic scoring call failed (%d/%d): %s. Retrying in %.1fs.",
                    attempt,
                    self._max_retries,
                    exc,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)

        raise ScoringError(f"Anthropic scoring request failed: {last_error}")

    def _record_response_usage(self, response: Any) -> None:
        if self._cost_tracker is None:
            return

        usage = _obj_get(response, "usage")
        input_tokens = _coerce_int(_obj_get(usage, "input_tokens", 0))
        output_tokens = _coerce_int(_obj_get(usage, "output_tokens", 0))
        self._cost_tracker.record(
            model=self.scoring.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def _build_system_prompt(self) -> str:
        return (
            "You are a precise research relevance scorer for arXiv papers.\n"
            "Given a research profile and a list of papers, score each paper from 1 to 10.\n"
            "Use the rubric exactly. Be strict, skeptical, and calibrated.\n\n"
            f"Research profile name:\n{self.profile.name}\n\n"
            f"Research profile description:\n{self.profile.description}\n\n"
            f"Scoring rubric:\n{self.profile.scoring_rubric}\n\n"
            "Calibration guidance:\n"
            "- Score relevance to this researcher, not generic paper quality.\n"
            "- Separate topic relevance from paper strength. A paper can be in-scope but still only deserve a 6-8.\n"
            "- 9/10 must be rare. Reserve it for papers that are both highly relevant and unusually strong, novel, or urgent to read.\n"
            "- 10/10 should be almost never used.\n"
            "- Surveys, taxonomies, benchmarks, and framework papers should usually score below 9 unless they seem clearly exceptional.\n"
            "- Do not give a high score just because a paper comes from a famous lab or matches a watchlist topic.\n"
            "- Unknown groups should not be penalized, but absent strong evidence of exceptional quality you should stay conservative.\n"
            "- If you are unsure whether something merits a 9, give it a 7 or 8 instead.\n\n"
            "Output format requirements:\n"
            '- Return ONLY valid JSON.\n'
            '- Top-level object must be: {"papers": [...]}.\n'
            '- Each paper object must include keys:\n'
            '  "arxiv_id" (string), "relevance_score" (number), '
            '"rationale" (one concise sentence), '
            '"novelty_signal" ("incremental"|"notable"|"breakthrough").\n'
            "- Do not add extra keys, comments, or markdown."
        )

    def _build_user_prompt(self, batch: Sequence[Paper]) -> str:
        payload = {
            "papers": [
                {
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "authors": paper.authors,
                    "categories": paper.categories,
                }
                for paper in batch
            ]
        }

        return (
            "Score each paper for relevance to the profile.\n"
            "Return JSON only in the required schema.\n\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )

    def _parse_scoring_response(
        self,
        response_text: str,
        expected_ids: set[str],
    ) -> list[ScoredPaper]:
        payload = _extract_json_payload(response_text)

        if isinstance(payload, dict):
            raw_items = payload.get("papers", [])
        elif isinstance(payload, list):
            raw_items = payload
        else:
            raise ScoringError("Scoring response JSON must be an object or array.")

        if not isinstance(raw_items, list):
            raise ScoringError("Scoring response field 'papers' must be a list.")

        parsed: list[ScoredPaper] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue

            arxiv_id = _normalize_arxiv_id(_as_string(item.get("arxiv_id")))
            if not arxiv_id or arxiv_id not in expected_ids:
                continue

            try:
                score = float(item.get("relevance_score"))
            except (TypeError, ValueError):
                self._logger.warning("Skipping paper %s due to invalid relevance_score.", arxiv_id)
                continue

            score = max(1.0, min(10.0, score))
            rationale = _as_string(item.get("rationale")).strip()
            novelty = _as_string(item.get("novelty_signal")).strip().lower()

            if novelty not in VALID_NOVELTY_SIGNALS:
                novelty = "incremental"

            parsed.append(
                ScoredPaper(
                    arxiv_id=arxiv_id,
                    relevance_score=score,
                    rationale=rationale or "No rationale provided.",
                    novelty_signal=cast(NoveltySignal, novelty),
                )
            )

        return parsed


def _build_anthropic_client(api_key: str) -> Any:
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise ScoringError(
            "anthropic package is not installed. Install dependencies from requirements.txt."
        ) from exc

    return Anthropic(api_key=api_key)


def _chunked(items: Sequence[T], size: int) -> Iterator[list[T]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def _obj_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _normalize_arxiv_id(arxiv_id: str) -> str:
    value = arxiv_id.strip()
    if not value:
        return value
    for marker in ("/abs/", "/pdf/"):
        if marker in value:
            value = value.split(marker, 1)[1]
    value = value.removesuffix(".pdf").strip().strip("/")
    value = _ARXIV_VERSION_RE.sub("", value)
    return value


def _calibrate_scored_paper(scored: ScoredPaper, paper: Paper | None) -> ScoredPaper:
    adjusted_score = scored.relevance_score

    if paper is not None:
        title_and_abstract = f"{paper.title}\n{paper.abstract}"
        if _SURVEY_PAPER_RE.search(title_and_abstract) and adjusted_score > 8.0:
            adjusted_score = 8.0

    if adjusted_score == scored.relevance_score:
        return scored

    return replace(scored, relevance_score=adjusted_score)


def _extract_message_text(message: Any) -> str:
    content = _obj_get(message, "content")
    if isinstance(content, str):
        return content.strip()

    if content is None and isinstance(message, list):
        content = message

    if not isinstance(content, list):
        text = _obj_get(message, "text")
        return _as_string(text).strip()

    parts: list[str] = []
    for block in content:
        block_type = _as_string(_obj_get(block, "type")).lower()
        if block_type == "text":
            text = _as_string(_obj_get(block, "text")).strip()
            if text:
                parts.append(text)

    return "\n".join(parts).strip()


def _extract_json_payload(text: str) -> Any:
    candidate = text.strip()
    if not candidate:
        raise ScoringError("Model returned an empty response.")

    fenced = _CODE_BLOCK_RE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()

    attempts: list[str] = [candidate]

    first_obj = candidate.find("{")
    last_obj = candidate.rfind("}")
    if first_obj != -1 and last_obj != -1 and last_obj > first_obj:
        attempts.append(candidate[first_obj : last_obj + 1])

    first_arr = candidate.find("[")
    last_arr = candidate.rfind("]")
    if first_arr != -1 and last_arr != -1 and last_arr > first_arr:
        attempts.append(candidate[first_arr : last_arr + 1])

    last_error: Exception | None = None
    for attempt in attempts:
        try:
            return json.loads(attempt)
        except json.JSONDecodeError as exc:
            last_error = exc

    raise ScoringError(f"Failed to parse JSON response from model: {last_error}")


def _coerce_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)
