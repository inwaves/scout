from __future__ import annotations

import logging
import re
import threading
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .models import KBPaperRecord

LOGGER = logging.getLogger(__name__)

_H1_RE = re.compile(r"^#\s+(.+?)\s*$")
_H2_RE = re.compile(r"^##\s+(.+?)\s*$")
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_BULLET_RE = re.compile(r"^\s*[-*+]\s+")
_NUMBERED_BULLET_RE = re.compile(r"^\s*\d+[.)]\s+")
_ARXIV_ID_RE = re.compile(r"\d{4}\.\d{4,5}")


class KnowledgeBaseError(RuntimeError):
    """Raised when knowledge base storage cannot be read or written."""


class KnowledgeBase:
    def __init__(
        self,
        papers_path: str | Path | None = None,
        max_topic_references: int = 50,
    ) -> None:
        self.papers_path = Path(papers_path).expanduser() if papers_path else None
        self.max_topic_references = max(1, int(max_topic_references))
        self._papers: dict[str, KBPaperRecord] = {}
        self._topics: dict[str, list[str]] = {}
        self._lock = threading.RLock()

    def load(self) -> None:
        """Load all *.md files from papers_path and build an in-memory index."""
        with self._lock:
            self._papers = {}
            self._topics = {}

            if self.papers_path is None:
                return
            if not self.papers_path.exists():
                return
            if not self.papers_path.is_dir():
                LOGGER.warning("KB papers path is not a directory: %s", self.papers_path)
                return

            loaded_count = 0
            used_ids: set[str] = set()

            for note_path in sorted(self.papers_path.glob("*.md")):
                record_id = _build_record_id(note_path, used_ids)
                record = _parse_note_record(note_path, record_id)
                if record is None:
                    continue

                used_ids.add(record.arxiv_id)
                self._papers[record.arxiv_id] = record
                self._add_paper_to_topics(record.arxiv_id, record.topics)
                loaded_count += 1

            LOGGER.info("Loaded %d KB paper note(s) from %s.", loaded_count, self.papers_path)

    def add_paper(self, record: KBPaperRecord) -> None:
        """Add or update a paper record and refresh topic index entries."""
        with self._lock:
            arxiv_id = record.arxiv_id.strip()
            if not arxiv_id:
                return

            try:
                score = float(record.score)
            except (TypeError, ValueError):
                score = 0.0

            sanitized = KBPaperRecord(
                arxiv_id=arxiv_id,
                title=record.title.strip(),
                authors=[author.strip() for author in record.authors if author and author.strip()],
                date_read=record.date_read.strip(),
                score=score,
                topics=_dedupe_preserve_order(_sanitize_string_list(record.topics)),
                key_findings=_dedupe_preserve_order(_sanitize_string_list(record.key_findings)),
                builds_on=_dedupe_preserve_order(_sanitize_string_list(record.builds_on)),
                tldr=record.tldr.strip(),
            )
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


def _parse_note_record(path: Path, record_id: str) -> KBPaperRecord | None:
    try:
        raw_markdown = path.read_text(encoding="utf-8")
    except OSError as exc:
        LOGGER.warning("Failed to read KB note %s: %s", path, exc)
        return None

    frontmatter, body = _parse_markdown_frontmatter(raw_markdown, path)
    sections = _extract_markdown_sections(body)

    title = _coerce_text(frontmatter.get("title")) or _extract_first_h1(body) or path.stem
    if not title:
        return None

    tags = _dedupe_preserve_order(_normalize_frontmatter_list(frontmatter.get("tags")))
    authors = _dedupe_preserve_order(_normalize_frontmatter_list(frontmatter.get("authors")))
    created = _coerce_iso_date(frontmatter.get("created")) or datetime.now(timezone.utc).date().isoformat()

    summary_text = _markdown_to_plain_text(
        sections.get("summary") or _extract_first_paragraph(body)
    )
    refs = _extract_reference_lines(sections.get("refs", ""))

    engaged = _coerce_bool(frontmatter.get("engaged"))
    insightful = _coerce_bool(frontmatter.get("insightful"))
    score = 9.0 if insightful else (8.0 if engaged else 7.0)

    return KBPaperRecord(
        arxiv_id=record_id,
        title=title,
        authors=authors,
        date_read=created,
        score=score,
        topics=tags,
        key_findings=[summary_text] if summary_text else [],
        builds_on=refs,
        tldr=summary_text or title,
    )


def _build_record_id(path: Path, used_ids: set[str]) -> str:
    """Build a stable record ID for a KB note, preferring arXiv IDs when available."""
    try:
        raw = path.read_text(encoding="utf-8")
        frontmatter, _ = _parse_markdown_frontmatter(raw, path)

        explicit_id = _extract_arxiv_id_from_text(_coerce_text(frontmatter.get("arxiv_id")))
        if explicit_id and explicit_id not in used_ids:
            return explicit_id

        url = _coerce_text(frontmatter.get("url"))
        arxiv_id = _extract_arxiv_id_from_url(url)
        if arxiv_id and arxiv_id not in used_ids:
            return arxiv_id
    except Exception:
        pass

    stem_arxiv = _extract_arxiv_id_from_text(path.stem)
    if stem_arxiv and stem_arxiv not in used_ids:
        return stem_arxiv

    slug = _slugify(path.stem) or "paper-note"
    candidate = f"note:{slug}"
    suffix = 2
    while candidate in used_ids:
        candidate = f"note:{slug}-{suffix}"
        suffix += 1
    return candidate


def _extract_arxiv_id_from_url(url: str) -> str:
    """Extract an arXiv ID from a URL like https://arxiv.org/abs/2603.26410."""
    if not url:
        return ""

    cleaned = url.strip()

    if cleaned.lower().startswith("arxiv:"):
        return _extract_arxiv_id_from_text(cleaned.split(":", 1)[1])

    for marker in ("/abs/", "/pdf/"):
        if marker in cleaned:
            candidate = cleaned.split(marker, 1)[1]
            return _extract_arxiv_id_from_text(candidate)

    return _extract_arxiv_id_from_text(cleaned)


def _extract_arxiv_id_from_text(text: str) -> str:
    if not text:
        return ""

    cleaned = str(text).strip()
    cleaned = cleaned.split("?", 1)[0].split("#", 1)[0]
    cleaned = cleaned.rstrip("/").removesuffix(".pdf")
    cleaned = re.sub(r"v\d+$", "", cleaned)

    match = _ARXIV_ID_RE.search(cleaned)
    if not match:
        return ""
    return match.group(0)


def _slugify(text: str) -> str:
    lowered = str(text).strip().lower()
    return re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")


def _parse_markdown_frontmatter(markdown: str, source_path: Path) -> tuple[dict[str, Any], str]:
    lines = markdown.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, markdown

    closing_index: int | None = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            closing_index = idx
            break

    if closing_index is None:
        LOGGER.warning("Missing closing YAML frontmatter delimiter in %s.", source_path)
        return {}, markdown

    frontmatter_text = "\n".join(lines[1:closing_index])
    body = "\n".join(lines[closing_index + 1 :]).lstrip("\n")

    try:
        loaded = yaml.safe_load(frontmatter_text) or {}
    except yaml.YAMLError as exc:
        LOGGER.warning("YAML parse error in KB note %s: %s", source_path, exc)
        return {}, body

    if not isinstance(loaded, dict):
        LOGGER.warning("Frontmatter in KB note %s is not a mapping.", source_path)
        return {}, body

    return loaded, body


def _extract_markdown_sections(body: str) -> dict[str, str]:
    target_sections = {"summary", "questions", "refs", "notes"}
    bucket: dict[str, list[str]] = {name: [] for name in target_sections}
    current: str | None = None

    for raw_line in body.splitlines():
        heading_match = _H2_RE.match(raw_line.strip())
        if heading_match:
            heading = heading_match.group(1).strip().lower().rstrip(":")
            current = heading if heading in target_sections else None
            continue
        if current:
            bucket[current].append(raw_line)

    return {name: "\n".join(lines).strip() for name, lines in bucket.items()}


def _extract_first_h1(body: str) -> str:
    for raw_line in body.splitlines():
        match = _H1_RE.match(raw_line.strip())
        if match:
            return _coerce_text(match.group(1))
    return ""


def _extract_first_paragraph(body: str) -> str:
    for block in re.split(r"\n\s*\n", body):
        paragraph = block.strip()
        if not paragraph or paragraph.startswith("#"):
            continue
        return paragraph
    return ""


def _markdown_to_plain_text(markdown: str) -> str:
    text = markdown or ""
    text = _IMAGE_RE.sub(" ", text)
    text = _LINK_RE.sub(r"\1", text)
    text = re.sub(r"`{1,3}", "", text)
    text = re.sub(r"[*_~]", "", text)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_reference_lines(markdown: str) -> list[str]:
    refs: list[str] = []
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        line = _BULLET_RE.sub("", line)
        line = _NUMBERED_BULLET_RE.sub("", line)
        cleaned = _markdown_to_plain_text(line)
        if cleaned:
            refs.append(cleaned)

    return _dedupe_preserve_order(refs)


def _normalize_frontmatter_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        items = [part.strip() for part in value.split(",")] if "," in value else [value]
    else:
        items = [value]

    cleaned: list[str] = []
    for item in items:
        text = str(item).strip()
        if text:
            cleaned.append(text)
    return cleaned


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return False


def _coerce_iso_date(value: Any) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()

    text = str(value).strip() if value is not None else ""
    if not text:
        return ""

    prefix = text[:10]
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", prefix):
        return prefix

    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return ""

    return parsed.date().isoformat()


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()