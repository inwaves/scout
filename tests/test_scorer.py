from __future__ import annotations

from datetime import datetime, timezone

from paper_scout.models import Paper, ScoredPaper
from paper_scout.scorer import _calibrate_scored_paper


def _make_paper(
    *,
    title: str = "Test Paper",
    abstract: str = "Test abstract.",
) -> Paper:
    return Paper(
        arxiv_id="2604.99999",
        title=title,
        abstract=abstract,
        authors=["Alice Example"],
        categories=["cs.AI"],
        published=datetime(2026, 4, 16, 10, 0, 0, tzinfo=timezone.utc),
        url="https://arxiv.org/abs/2604.99999",
        pdf_url="https://arxiv.org/pdf/2604.99999.pdf",
    )


def test_non_breakthrough_score_can_remain_9() -> None:
    scored = ScoredPaper(
        arxiv_id="2604.99999",
        relevance_score=9.0,
        rationale="Strong and relevant paper.",
        novelty_signal="notable",
    )

    calibrated = _calibrate_scored_paper(scored, _make_paper())

    assert calibrated.relevance_score == 9.0


def test_breakthrough_score_can_remain_9() -> None:
    scored = ScoredPaper(
        arxiv_id="2604.99999",
        relevance_score=9.0,
        rationale="Exceptional result.",
        novelty_signal="breakthrough",
    )

    calibrated = _calibrate_scored_paper(scored, _make_paper())

    assert calibrated.relevance_score == 9.0


def test_survey_papers_are_capped_at_8() -> None:
    scored = ScoredPaper(
        arxiv_id="2604.99999",
        relevance_score=9.5,
        rationale="Comprehensive survey.",
        novelty_signal="breakthrough",
    )

    calibrated = _calibrate_scored_paper(
        scored,
        _make_paper(
            title="Reward Hacking in the Era of Large Models: A Survey",
            abstract="This survey reviews mechanisms and open challenges.",
        ),
    )

    assert calibrated.relevance_score == 8.0
