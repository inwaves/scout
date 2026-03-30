from __future__ import annotations

from pathlib import Path

from paper_scout.delivery.discord import _chunk_text
from paper_scout.delivery.file import MarkdownFileDelivery


class TestMarkdownFileDelivery:
    def test_creates_file(self, tmp_path: Path) -> None:
        delivery = MarkdownFileDelivery(
            output_dir=str(tmp_path),
            filename_template="test-{date}.md",
        )
        delivery.deliver(
            subject="Test Subject",
            markdown_body="# Test Digest\n\nContent here.",
            html_body="<html></html>",
        )
        files = list(tmp_path.glob("test-*.md"))
        assert len(files) == 1
        content = files[0].read_text(encoding="utf-8")
        assert "Test Digest" in content

    def test_creates_output_directory(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "sub" / "dir"
        delivery = MarkdownFileDelivery(
            output_dir=str(output_dir),
            filename_template="digest-{date}.md",
        )
        delivery.deliver(
            subject="Subject",
            markdown_body="Body",
            html_body="<html></html>",
        )
        assert output_dir.exists()
        assert len(list(output_dir.glob("digest-*.md"))) == 1


class TestDiscordChunking:
    def test_short_text_single_chunk(self) -> None:
        text = "Hello world"
        chunks = _chunk_text(text, max_chars=2000)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_split(self) -> None:
        lines = [f"Line {i}\n" for i in range(100)]
        text = "".join(lines)
        chunks = _chunk_text(text, max_chars=200)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 200

    def test_empty_text(self) -> None:
        chunks = _chunk_text("", max_chars=2000)
        assert chunks == [] or chunks == [""]

    def test_exact_boundary(self) -> None:
        text = "a" * 2000
        chunks = _chunk_text(text, max_chars=2000)
        assert len(chunks) == 1