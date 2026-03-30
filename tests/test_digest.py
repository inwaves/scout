from __future__ import annotations

from datetime import datetime, timezone

from paper_scout.digest import DigestRenderer
from paper_scout.models import DigestContext, DigestEntry, Paper


def _make_entry(
    arxiv_id: str = "2603.12345",
    title: str = "Test Paper",
    score: float = 9.0,
    novelty: str = "notable",
) -> DigestEntry:
    return DigestEntry(
        paper=Paper(
            arxiv_id=arxiv_id,
            title=title,
            abstract="Test abstract.",
            authors=["Alice", "Bob"],
            categories=["cs.AI", "cs.LG"],
            published=datetime(2026, 3, 28, 10, 0, 0, tzinfo=timezone.utc),
            url=f"https://arxiv.org/abs/{arxiv_id}",
            pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        ),
        relevance_score=score,
        rationale="Highly relevant to agent research.",
        novelty_signal=novelty,
        summary="This paper introduces a novel approach to agent planning.",
    )


class TestDigestRenderer:
    def test_render_markdown(self) -> None:
        renderer = DigestRenderer()
        context = DigestContext(
            generated_at=datetime(2026, 3, 28, 7, 0, 0, tzinfo=timezone.utc),
            total_reviewed=200,
            threshold=7.0,
            entries=[_make_entry()],
        )
        result = renderer.render(context)
        assert "Paper Scout Digest" in result.markdown
        assert "Test Paper" in result.markdown
        assert "9/10" in result.markdown
        assert "200 reviewed" in result.markdown
        assert result.subject != ""

    def test_render_html(self) -> None:
        renderer = DigestRenderer()
        context = DigestContext(
            generated_at=datetime(2026, 3, 28, 7, 0, 0, tzinfo=timezone.utc),
            total_reviewed=100,
            threshold=7.0,
            entries=[_make_entry()],
        )
        result = renderer.render(context)
        assert "<html" in result.html
        assert "Test Paper" in result.html
        assert "9/10" in result.html

    def test_empty_entries(self) -> None:
        renderer = DigestRenderer()
        context = DigestContext(
            generated_at=datetime(2026, 3, 28, 7, 0, 0, tzinfo=timezone.utc),
            total_reviewed=50,
            threshold=7.0,
            entries=[],
        )
        result = renderer.render(context)
        assert "0 papers scored" in result.markdown
        assert "No papers met your threshold" in result.markdown

    def test_subject_template(self) -> None:
        renderer = DigestRenderer()
        context = DigestContext(
            generated_at=datetime(2026, 3, 28, 7, 0, 0, tzinfo=timezone.utc),
            total_reviewed=100,
            threshold=7.0,
            entries=[_make_entry(), _make_entry(arxiv_id="2603.99999", title="Other")],
        )
        result = renderer.render(context, subject_template="Daily Digest — {date} ({count})")
        assert "2026-03-28" in result.subject
        assert "2" in result.subject

    def test_multiple_entries_ranked(self) -> None:
        renderer = DigestRenderer()
        entries = [
            _make_entry(arxiv_id="2603.00001", title="First", score=10.0),
            _make_entry(arxiv_id="2603.00002", title="Second", score=8.0),
            _make_entry(arxiv_id="2603.00003", title="Third", score=7.5),
        ]
        context = DigestContext(
            generated_at=datetime(2026, 3, 28, 7, 0, 0, tzinfo=timezone.utc),
            total_reviewed=300,
            threshold=7.0,
            entries=entries,
        )
        result = renderer.render(context)
        assert "3 papers scored" in result.markdown
        # Check ordering preserved
        first_pos = result.markdown.find("First")
        second_pos = result.markdown.find("Second")
        third_pos = result.markdown.find("Third")
        assert first_pos < second_pos < third_pos