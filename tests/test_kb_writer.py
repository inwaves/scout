from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from paper_scout.kb_writer import KBNoteWriter
from paper_scout.models import DeepReadBreakdown, DeepReadEntry, DeepReadResult, Paper


def test_deep_read_note_uses_scout_tag_and_manual_flags(tmp_path: Path) -> None:
    writer = KBNoteWriter(tmp_path)
    paper = Paper(
        arxiv_id="2604.13954",
        title="Test Deep Read",
        abstract="Abstract",
        authors=["Alice Example"],
        categories=["cs.AI"],
        published=datetime(2026, 4, 16, 10, 0, 0, tzinfo=timezone.utc),
        url="https://arxiv.org/abs/2604.13954",
        pdf_url="https://arxiv.org/pdf/2604.13954.pdf",
    )
    breakdown = DeepReadBreakdown(
        triage="Read the full paper",
        tldr=["One", "Two", "Three"],
        motivation="Motivation",
        hypothesis="Hypothesis",
        methodology="Methodology",
        results="Results",
        interpretation="Interpretation",
        context="Context",
        limitations="Limitations",
        relevance="Relevance",
    )
    result = DeepReadResult(
        entry=DeepReadEntry(
            paper=paper,
            relevance_score=8.5,
            rationale="Highly relevant.",
            novelty_signal="notable",
            breakdown=breakdown,
        ),
        topics=["agent-safety"],
    )

    path = writer.write_deep_read_note(paper, result)
    text = path.read_text(encoding="utf-8")

    assert "  - scout" in text
    assert "engaged: false" in text
    assert "insightful: false" in text
