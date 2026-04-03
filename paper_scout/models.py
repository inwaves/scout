from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

NoveltySignal = Literal["incremental", "notable", "breakthrough"]
VALID_NOVELTY_SIGNALS: tuple[NoveltySignal, ...] = ("incremental", "notable", "breakthrough")


@dataclass(slots=True)
class Paper:
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    published: datetime
    url: str
    pdf_url: str


@dataclass(slots=True)
class ScoredPaper:
    arxiv_id: str
    relevance_score: float
    rationale: str
    novelty_signal: NoveltySignal = "incremental"


@dataclass(slots=True)
class SummarizedPaper:
    arxiv_id: str
    summary: str


@dataclass(slots=True)
class WatchlistMatch:
    match_type: str  # "author" | "organization"
    matched_name: str


@dataclass(slots=True)
class DigestEntry:
    paper: Paper
    relevance_score: float
    rationale: str
    novelty_signal: NoveltySignal
    summary: str
    watchlist_match: str | None = None


@dataclass(slots=True)
class KBPaperRecord:
    arxiv_id: str
    title: str
    authors: list[str]
    date_read: str  # ISO date
    score: float
    topics: list[str]
    key_findings: list[str]
    builds_on: list[str]
    tldr: str


@dataclass(slots=True)
class DeepReadBreakdown:
    triage: str  # "Read the full paper" / "TL;DR captures it"
    tldr: list[str]  # 3 bullets
    motivation: str
    hypothesis: str
    methodology: str
    results: str
    interpretation: str
    context: str
    limitations: str
    relevance: str  # Why it matters to you


@dataclass(slots=True)
class DeepReadEntry:
    paper: Paper
    relevance_score: float
    rationale: str
    novelty_signal: NoveltySignal
    breakdown: DeepReadBreakdown
    watchlist_match: str | None = None  # "Anthropic", "Ethan Perez", etc.


@dataclass(slots=True)
class DeepReadResult:
    entry: DeepReadEntry
    topics: list[str] = field(default_factory=list)
    key_findings: list[str] = field(default_factory=list)
    builds_on: list[str] = field(default_factory=list)


@dataclass(slots=True)
class S2Paper:
    paper_id: str
    title: str
    abstract: str | None
    citation_count: int
    reference_count: int
    authors: list[dict]  # [{"name": "...", "affiliations": ["..."]}]


@dataclass(slots=True)
class S2Reference:
    paper_id: str
    title: str
    abstract: str | None
    authors: list[str]
    year: int | None


@dataclass(slots=True)
class DigestContext:
    generated_at: datetime
    total_reviewed: int
    threshold: float
    run_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    entries: list[DigestEntry] = field(default_factory=list)  # Backward-compatible fallback
    deep_reads: list[DeepReadEntry] = field(default_factory=list)
    noteworthy_entries: list[DigestEntry] = field(default_factory=list)


@dataclass(slots=True)
class RenderedDigest:
    subject: str
    markdown: str
    html: str