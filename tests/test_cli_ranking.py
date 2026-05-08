from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from paper_scout.cli import _collect_hot_alerts, _merge_seen_web_posts, _select_scored_for_digest
from paper_scout.config import (
    AlertConfig,
    AnalysisConfig,
    ArxivConfig,
    FeedbackConfig,
    KnowledgeBaseConfig,
    PaperScoutConfig,
    ProfileConfig,
    ScoringConfig,
    ScheduleConfig,
    WatchlistConfig,
)
from paper_scout.knowledge_base import KnowledgeBase
from paper_scout.models import Paper, ScoredPaper, WatchlistMatch


def _make_paper(arxiv_id: str = "2604.10000", title: str = "Test Paper") -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        title=title,
        abstract="Test abstract.",
        authors=["Alice Example"],
        categories=["cs.AI"],
        published=datetime(2026, 4, 16, 10, 0, 0, tzinfo=timezone.utc),
        url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
    )


def _make_config() -> PaperScoutConfig:
    return PaperScoutConfig(
        profile=ProfileConfig(name="Test", description="Test", scoring_rubric="Test"),
        arxiv=ArxivConfig(),
        scoring=ScoringConfig(threshold=7.0, max_papers=15),
        analysis=AnalysisConfig(),
        watchlist=WatchlistConfig(),
        alerts=AlertConfig(enabled=True, score_threshold=9.0, watchlist_score_threshold=8.0),
        knowledge_base=KnowledgeBaseConfig(),
        feedback=FeedbackConfig(),
        delivery_channels=[],
        schedule=ScheduleConfig(),
    )


def test_org_watchlist_match_does_not_bypass_digest_threshold(tmp_path: Path) -> None:
    paper = _make_paper()
    scored = ScoredPaper(
        arxiv_id=paper.arxiv_id,
        relevance_score=2.0,
        ranking_score=2.0,
        rationale="Low relevance.",
        novelty_signal="incremental",
    )
    kb = KnowledgeBase(tmp_path / "kb")
    kb.load()

    selected = _select_scored_for_digest(
        papers=[paper],
        score_by_id={paper.arxiv_id: scored},
        watchlist_matches={
            paper.arxiv_id: WatchlistMatch(match_type="organization", matched_name="Anthropic")
        },
        threshold=7.0,
        max_papers=15,
        knowledge_base=kb,
    )

    assert selected == []


def test_watchlist_match_can_trigger_alert_only_with_score_threshold(tmp_path: Path) -> None:
    paper = _make_paper()
    config = _make_config()
    kb = KnowledgeBase(tmp_path / "kb")
    kb.load()

    low_scored = ScoredPaper(
        arxiv_id=paper.arxiv_id,
        relevance_score=7.2,
        ranking_score=7.2,
        rationale="Relevant but not alert-worthy.",
        novelty_signal="notable",
    )
    low_alerts = _collect_hot_alerts(
        papers=[paper],
        score_by_id={paper.arxiv_id: low_scored},
        watchlist_matches={
            paper.arxiv_id: WatchlistMatch(match_type="organization", matched_name="Anthropic")
        },
        config=config,
        knowledge_base=kb,
    )
    assert low_alerts == []

    high_scored = ScoredPaper(
        arxiv_id=paper.arxiv_id,
        relevance_score=8.2,
        ranking_score=8.2,
        rationale="Strong watchlist paper.",
        novelty_signal="notable",
    )
    high_alerts = _collect_hot_alerts(
        papers=[paper],
        score_by_id={paper.arxiv_id: high_scored},
        watchlist_matches={
            paper.arxiv_id: WatchlistMatch(match_type="organization", matched_name="Anthropic")
        },
        config=config,
        knowledge_base=kb,
    )
    assert len(high_alerts) == 1
    assert "Watchlist match" in high_alerts[0].reason


def test_merge_seen_web_posts_records_observed_posts() -> None:
    paper = _make_paper(
        arxiv_id="openai:test-post",
        title="Test Post",
    )
    paper.source_label = "OpenAI"
    paper.url = "https://openai.com/index/test-post"
    paper.pdf_url = paper.url

    seen = _merge_seen_web_posts(
        existing={},
        observed=[paper],
        timestamp=datetime(2026, 4, 28, 9, 0, 0, tzinfo=timezone.utc),
        limit=5000,
    )

    assert seen["openai:test-post"]["url"] == "https://openai.com/index/test-post"
    assert seen["openai:test-post"]["source_label"] == "OpenAI"
    assert seen["openai:test-post"]["first_seen"] == "2026-04-28T09:00:00+00:00"
