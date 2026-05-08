from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

import pytest

from paper_scout.config import FeedbackConfig
from paper_scout.feedback import (
    FeedbackError,
    FeedbackStore,
    FeedbackTokenSigner,
    build_feedback_links,
    load_preference_model,
    vote_from_token_payload,
)
from paper_scout.models import Paper, ScoredPaper


def _make_paper(title: str = "HINTBench: A Benchmark for Safe Agents") -> Paper:
    return Paper(
        arxiv_id="2604.13954",
        title=title,
        abstract="A benchmark paper for agent safety evaluation.",
        authors=["Alice Example"],
        categories=["cs.AI", "cs.LG"],
        published=datetime(2026, 4, 16, 10, 0, 0, tzinfo=timezone.utc),
        url="https://arxiv.org/abs/2604.13954",
        pdf_url="https://arxiv.org/pdf/2604.13954.pdf",
        source_label="arXiv",
    )


def _make_scored(score: float = 8.8) -> ScoredPaper:
    return ScoredPaper(
        arxiv_id="2604.13954",
        relevance_score=score,
        rationale="Relevant benchmark for agent safety.",
        novelty_signal="notable",
    )


def test_feedback_token_round_trip() -> None:
    signer = FeedbackTokenSigner("test-secret")
    links = build_feedback_links(
        base_url="https://scout.tail.example",
        signer=signer,
        paper=_make_paper(),
        scored=_make_scored(),
        digest_date="2026-04-16",
    )

    token = unquote(links["upvote_url"].split("token=", 1)[1])
    payload = signer.loads(token)

    assert payload["pid"] == "2604.13954"
    assert payload["d"] == "2026-04-16"


def test_feedback_token_rejects_tampering() -> None:
    signer = FeedbackTokenSigner("test-secret")
    links = build_feedback_links(
        base_url="https://scout.tail.example",
        signer=signer,
        paper=_make_paper(),
        scored=_make_scored(),
        digest_date="2026-04-16",
    )
    token = unquote(links["upvote_url"].split("token=", 1)[1])
    tampered = token[:-1] + ("A" if token[-1] != "A" else "B")

    with pytest.raises(FeedbackError):
        signer.loads(tampered)


def test_feedback_store_upserts_votes(tmp_path: Path) -> None:
    signer = FeedbackTokenSigner("test-secret")
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    store.initialize()

    payload = signer.loads(
        unquote(
            build_feedback_links(
            base_url="https://scout.tail.example",
            signer=signer,
            paper=_make_paper(),
            scored=_make_scored(),
            digest_date="2026-04-16",
        )["upvote_url"].split("token=", 1)[1]
        )
    )

    store.record_vote(vote_from_token_payload(payload, 1))
    store.record_vote(vote_from_token_payload(payload, -1))

    votes = store.list_votes()
    assert len(votes) == 1
    assert votes[0].vote == -1


def test_feedback_preference_model_downranks_disliked_benchmarks(tmp_path: Path) -> None:
    signer = FeedbackTokenSigner("test-secret")
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    store.initialize()

    for index in range(2):
        paper = _make_paper(title=f"Benchmark Paper {index}: Safe Agents")
        scored = _make_scored(score=8.5)
        payload = signer.loads(
            unquote(
                build_feedback_links(
                base_url="https://scout.tail.example",
                signer=signer,
                paper=paper,
                scored=scored,
                digest_date=f"2026-04-1{index + 1}",
            )["downvote_url"].split("token=", 1)[1]
            )
        )
        store.record_vote(vote_from_token_payload(payload, -1))

    model = load_preference_model(
        store,
        FeedbackConfig(enabled=True, max_adjustment=1.5, min_feature_votes=2),
    )
    adjusted = model.adjust_score(_make_paper(), _make_scored(score=8.8))

    assert adjusted.ranking_score is not None
    assert adjusted.ranking_score < adjusted.relevance_score
