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
class DigestEntry:
    paper: Paper
    relevance_score: float
    rationale: str
    novelty_signal: NoveltySignal
    summary: str


@dataclass(slots=True)
class DigestContext:
    generated_at: datetime
    total_reviewed: int
    threshold: float
    entries: list[DigestEntry] = field(default_factory=list)


@dataclass(slots=True)
class RenderedDigest:
    subject: str
    markdown: str
    html: str