from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from .models import DeepReadResult, Paper, ScoredPaper

_MAX_FILENAME_CHARS = 60
_SIMPLE_YAML_SCALAR_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._:/+\-()]*$")


class KBNoteWriter:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_stub_note(
        self,
        paper: Paper,
        scored: ScoredPaper,
        tags: list[str] | None = None,
    ) -> Path:
        """Generate a minimal KB note for papers scoring >= 7."""
        title = _single_line(paper.title) or paper.arxiv_id
        authors = _dedupe(_sanitize_list(paper.authors)) or ["Unknown"]
        provided_tags = _sanitize_list(tags or [])
        note_tags = _dedupe(provided_tags or _sanitize_list(paper.categories)) or ["uncategorized"]
        summary = _clean_block(scored.rationale) or "No scoring rationale provided."

        note_text = self._build_document(
            title=title,
            authors=authors,
            tags=note_tags,
            url=_single_line(paper.url),
            engaged=False,
            insightful=False,
            summary=summary,
            questions="",
            refs="",
            notes="",
        )
        destination = self.output_dir / f"{_slugify(title)}.md"
        destination.write_text(note_text, encoding="utf-8")
        return destination

    def write_deep_read_note(
        self,
        paper: Paper,
        result: DeepReadResult,
        watchlist_match: str | None = None,
    ) -> Path:
        """Generate a full KB note for papers scoring >= 9 or triggering hot alerts."""
        title = _single_line(paper.title) or paper.arxiv_id
        authors = _dedupe(_sanitize_list(paper.authors)) or ["Unknown"]
        note_tags = _dedupe(_sanitize_list(result.topics)) or _dedupe(
            _sanitize_list(paper.categories)
        ) or ["uncategorized"]
        if "scout" not in note_tags:
            note_tags.append("scout")

        summary = _render_bullets(result.entry.breakdown.tldr) or "- No TL;DR generated."
        refs = _render_bullets(result.builds_on)
        notes = _build_deep_notes(result, watchlist_match=watchlist_match)

        note_text = self._build_document(
            title=title,
            authors=authors,
            tags=note_tags,
            url=_single_line(paper.url),
            engaged=False,
            insightful=False,
            summary=summary,
            questions="",
            refs=refs,
            notes=notes,
        )
        destination = self.output_dir / f"{_slugify(title)}.md"
        destination.write_text(note_text, encoding="utf-8")
        return destination

    def _build_document(
        self,
        *,
        title: str,
        authors: list[str],
        tags: list[str],
        url: str,
        engaged: bool,
        insightful: bool,
        summary: str,
        questions: str,
        refs: str,
        notes: str,
    ) -> str:
        created = datetime.now(timezone.utc).date().isoformat()
        frontmatter = _build_frontmatter(
            title=title,
            authors=authors,
            tags=tags,
            url=url,
            created=created,
            engaged=engaged,
            insightful=insightful,
        )
        body = _build_body(
            title=title,
            summary=summary,
            questions=questions,
            refs=refs,
            notes=notes,
        )
        return f"{frontmatter}\n{body}"


def _build_frontmatter(
    *,
    title: str,
    authors: list[str],
    tags: list[str],
    url: str,
    created: str,
    engaged: bool,
    insightful: bool,
) -> str:
    lines = ["---", f'title: "{_escape_double_quotes(title)}"', "authors:"]
    for author in authors:
        lines.append(f"  - {_yaml_scalar(author)}")

    lines.append("tags:")
    for tag in tags:
        lines.append(f"  - {_yaml_scalar(tag)}")

    if url:
        lines.append(f"url: {url}")
    else:
        lines.append('url: ""')

    lines.extend(
        [
            f"created: {created}",
            f"engaged: {'true' if engaged else 'false'}",
            f"insightful: {'true' if insightful else 'false'}",
            "---",
        ]
    )
    return "\n".join(lines)


def _build_body(
    *,
    title: str,
    summary: str,
    questions: str,
    refs: str,
    notes: str,
) -> str:
    sections = (
        ("Summary", summary),
        ("Questions", questions),
        ("Refs", refs),
        ("Notes", notes),
    )

    lines = [f"# {title}", ""]
    for index, (heading, content) in enumerate(sections):
        lines.append(f"## {heading}")
        cleaned = _clean_block(content)
        if cleaned:
            lines.extend(cleaned.splitlines())
        if index < len(sections) - 1:
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _build_deep_notes(result: DeepReadResult, watchlist_match: str | None) -> str:
    breakdown = result.entry.breakdown
    sections = (
        ("Triage", breakdown.triage),
        ("Motivation", breakdown.motivation),
        ("Hypothesis", breakdown.hypothesis),
        ("Methodology", breakdown.methodology),
        ("Results", breakdown.results),
        ("Interpretation", breakdown.interpretation),
        ("Context", breakdown.context),
        ("Limitations", breakdown.limitations),
        ("Why it matters", breakdown.relevance),
    )

    lines: list[str] = []
    if watchlist_match:
        lines.append(f"- Watchlist match: {_single_line(watchlist_match)}")
        lines.append("")

    for index, (heading, text) in enumerate(sections):
        lines.append(f"### {heading}")
        content = _clean_block(text) or "Not provided."
        lines.extend(content.splitlines())
        if index < len(sections) - 1:
            lines.append("")

    return "\n".join(lines).strip()


def _render_bullets(items: list[str]) -> str:
    cleaned: list[str] = []
    for item in items:
        text = _single_line(item)
        if text:
            cleaned.append(text)
    return "\n".join(f"- {item}" for item in cleaned)


def _yaml_scalar(value: str) -> str:
    scalar = _single_line(value)
    if not scalar:
        return '""'

    lowered = scalar.lower()
    if lowered in {"true", "false", "null", "~"}:
        return f'"{_escape_double_quotes(scalar)}"'

    if _SIMPLE_YAML_SCALAR_RE.fullmatch(scalar) and ": " not in scalar:
        return scalar

    return f'"{_escape_double_quotes(scalar)}"'


def _escape_double_quotes(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _slugify(text: str) -> str:
    lowered = _single_line(text).lower()
    slug = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    if len(slug) > _MAX_FILENAME_CHARS:
        truncated = slug[:_MAX_FILENAME_CHARS].rstrip("-")
        if "-" in truncated:
            candidate = truncated.rsplit("-", 1)[0]
            if candidate:
                truncated = candidate
        slug = truncated
    return slug or "paper-note"


def _sanitize_list(values: list[str]) -> list[str]:
    cleaned: list[str] = []
    for value in values:
        text = _single_line(value)
        if text:
            cleaned.append(text)
    return cleaned


def _single_line(value: str) -> str:
    return " ".join(str(value).split()).strip()


def _clean_block(value: str) -> str:
    if not value:
        return ""

    normalized = str(value).replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.split("\n")]
    return "\n".join(lines).strip()


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped
