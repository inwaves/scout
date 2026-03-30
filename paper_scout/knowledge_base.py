from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import KBPaperRecord

LOGGER = logging.getLogger(__name__)


class KnowledgeBaseError(RuntimeError):
    """Raised when knowledge base storage cannot be read or written."""


class KnowledgeBase:
    def __init__(self, path: str | Path, max_topic_references: int = 50) -> None:
        self.path = Path(path).expanduser()
        self.max_topic_references = max(1, int(max_topic_references))
        self._papers_path = self.path / "papers.json"
        self._topics_path = self.path / "topics.json"
        self._papers: dict[str, KBPaperRecord] = {}
        self._topics: dict[str, list[str]] = {}
        self._lock = threading.RLock()

    def load(self) -> None:
        """Load knowledge base contents from disk."""
        with self._lock:
            self.path.mkdir(parents=True, exist_ok=True)

            papers_payload = self._read_json(self._papers_path, default={})
            topics_payload = self._read_json(self._topics_path, default={})

            self._papers = self._parse_papers_payload(papers_payload)
            parsed_topics = self._parse_topics_payload(
                topics_payload,
                known_ids=set(self._papers),
            )

            if parsed_topics:
                self._topics = parsed_topics
            else:
                self._topics = {}
                self._rebuild_topics_index()

    def save(self) -> None:
        """Persist knowledge base to disk using atomic file replacement."""
        with self._lock:
            self.path.mkdir(parents=True, exist_ok=True)

            papers_payload = {
                arxiv_id: asdict(record) for arxiv_id, record in sorted(self._papers.items())
            }

            topics_payload: dict[str, list[str]] = {}
            for topic_key, arxiv_ids in sorted(self._topics.items()):
                filtered = [arxiv_id for arxiv_id in arxiv_ids if arxiv_id in self._papers]
                if filtered:
                    topics_payload[topic_key] = filtered[-self.max_topic_references :]

            self._atomic_write_json(self._papers_path, papers_payload)
            self._atomic_write_json(self._topics_path, topics_payload)

    def add_paper(self, record: KBPaperRecord) -> None:
        """Add or update a paper record and refresh topic index entries."""
        with self._lock:
            sanitized = KBPaperRecord(
                arxiv_id=record.arxiv_id.strip(),
                title=record.title.strip(),
                authors=[author.strip() for author in record.authors if author and author.strip()],
                date_read=record.date_read.strip(),
                score=float(record.score),
                topics=_dedupe_preserve_order(_sanitize_string_list(record.topics)),
                key_findings=_dedupe_preserve_order(_sanitize_string_list(record.key_findings)),
                builds_on=_dedupe_preserve_order(_sanitize_string_list(record.builds_on)),
                tldr=record.tldr.strip(),
            )
            if not sanitized.arxiv_id:
                return

            self._papers[sanitized.arxiv_id] = sanitized
            self.update_topics(sanitized.arxiv_id, sanitized.topics)

    def lookup_topics(self, topics: list[str]) -> list[KBPaperRecord]:
        """Return paper records related to any of the provided topics."""
        with self._lock:
            topic_keys = [_topic_key(topic) for topic in topics if topic and topic.strip()]
            if not topic_keys:
                return []

            ordered_ids: list[str] = []
            seen_ids: set[str] = set()
            for topic_key in topic_keys:
                for arxiv_id in self._topics.get(topic_key, []):
                    if arxiv_id in seen_ids:
                        continue
                    seen_ids.add(arxiv_id)
                    ordered_ids.append(arxiv_id)

            records = [self._papers[arxiv_id] for arxiv_id in ordered_ids if arxiv_id in self._papers]
            records.sort(
                key=lambda record: (_safe_iso_to_datetime(record.date_read), record.score),
                reverse=True,
            )
            return records

    def get_paper(self, arxiv_id: str) -> KBPaperRecord | None:
        with self._lock:
            return self._papers.get(arxiv_id.strip())

    def has_paper(self, arxiv_id: str) -> bool:
        with self._lock:
            return arxiv_id.strip() in self._papers

    def get_topic_papers(self, topic: str) -> list[str]:
        with self._lock:
            return list(self._topics.get(_topic_key(topic), []))

    def update_topics(self, arxiv_id: str, topics: list[str]) -> None:
        with self._lock:
            normalized_id = arxiv_id.strip()
            if not normalized_id:
                return

            clean_topics = _dedupe_preserve_order(_sanitize_string_list(topics))

            if normalized_id in self._papers:
                self._papers[normalized_id].topics = clean_topics

            self._remove_paper_from_topics(normalized_id)
            self._add_paper_to_topics(normalized_id, clean_topics)

    def _read_json(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        try:
            text = path.read_text(encoding="utf-8")
            payload = json.loads(text)
        except (OSError, json.JSONDecodeError) as exc:
            raise KnowledgeBaseError(f"Failed to read knowledge base file {path}: {exc}") from exc
        return payload

    def _atomic_write_json(self, path: Path, payload: Any) -> None:
        temporary_path = path.with_name(f"{path.name}.tmp")
        try:
            serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            temporary_path.write_text(serialized + "\n", encoding="utf-8")
            temporary_path.replace(path)
        except OSError as exc:
            raise KnowledgeBaseError(f"Failed to write knowledge base file {path}: {exc}") from exc

    def _parse_papers_payload(self, payload: Any) -> dict[str, KBPaperRecord]:
        records: dict[str, KBPaperRecord] = {}
        if isinstance(payload, dict):
            iterable = payload.values()
        elif isinstance(payload, list):
            iterable = payload
        else:
            LOGGER.warning("Ignoring unexpected papers.json shape (%s).", type(payload).__name__)
            return records

        for raw in iterable:
            parsed = _parse_paper_record(raw)
            if parsed is None:
                continue
            records[parsed.arxiv_id] = parsed
        return records

    def _parse_topics_payload(self, payload: Any, known_ids: set[str]) -> dict[str, list[str]]:
        parsed: dict[str, list[str]] = {}
        if not isinstance(payload, dict):
            return parsed

        for raw_topic, raw_ids in payload.items():
            topic_key = _topic_key(str(raw_topic))
            if not topic_key:
                continue
            if not isinstance(raw_ids, list):
                continue

            clean_ids: list[str] = []
            seen_ids: set[str] = set()
            for raw_id in raw_ids:
                arxiv_id = str(raw_id).strip()
                if not arxiv_id or arxiv_id in seen_ids:
                    continue
                if known_ids and arxiv_id not in known_ids:
                    continue
                seen_ids.add(arxiv_id)
                clean_ids.append(arxiv_id)

            if clean_ids:
                parsed[topic_key] = clean_ids[-self.max_topic_references :]
        return parsed

    def _remove_paper_from_topics(self, arxiv_id: str) -> None:
        empty_topics: list[str] = []
        for topic_key, paper_ids in self._topics.items():
            if arxiv_id in paper_ids:
                paper_ids[:] = [item for item in paper_ids if item != arxiv_id]
            if not paper_ids:
                empty_topics.append(topic_key)

        for topic_key in empty_topics:
            self._topics.pop(topic_key, None)

    def _add_paper_to_topics(self, arxiv_id: str, topics: list[str]) -> None:
        for topic in topics:
            topic_key = _topic_key(topic)
            if not topic_key:
                continue
            bucket = self._topics.setdefault(topic_key, [])
            if arxiv_id in bucket:
                bucket.remove(arxiv_id)
            bucket.append(arxiv_id)
            if len(bucket) > self.max_topic_references:
                del bucket[: len(bucket) - self.max_topic_references]

    def _rebuild_topics_index(self) -> None:
        self._topics = {}
        for arxiv_id, record in self._papers.items():
            self._add_paper_to_topics(arxiv_id, record.topics)


def _parse_paper_record(raw: Any) -> KBPaperRecord | None:
    if not isinstance(raw, dict):
        return None

    arxiv_id = str(raw.get("arxiv_id", "")).strip()
    title = str(raw.get("title", "")).strip()
    date_read = str(raw.get("date_read", "")).strip()
    if not arxiv_id or not title or not date_read:
        return None

    try:
        score = float(raw.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0

    return KBPaperRecord(
        arxiv_id=arxiv_id,
        title=title,
        authors=_sanitize_string_list(raw.get("authors")),
        date_read=date_read,
        score=score,
        topics=_sanitize_string_list(raw.get("topics")),
        key_findings=_sanitize_string_list(raw.get("key_findings")),
        builds_on=_sanitize_string_list(raw.get("builds_on")),
        tldr=str(raw.get("tldr", "")).strip(),
    )


def _sanitize_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        source = value
    elif isinstance(value, str):
        source = [value]
    else:
        return []

    cleaned: list[str] = []
    for item in source:
        text = str(item).strip()
        if text:
            cleaned.append(text)
    return cleaned


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _topic_key(topic: str) -> str:
    return " ".join(topic.strip().lower().split())


def _safe_iso_to_datetime(value: str) -> datetime:
    text = value.strip()
    if not text:
        return datetime.min

    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return datetime.min

    if parsed.tzinfo is not None:
        return parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed