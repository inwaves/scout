from __future__ import annotations

from datetime import datetime, timezone

from paper_scout.config import ArxivConfig
from paper_scout.fetcher import ArxivFetcher


class TestArxivIdExtraction:
    def test_full_url(self) -> None:
        assert ArxivFetcher._extract_arxiv_id("http://arxiv.org/abs/2603.19461v1") == "2603.19461"

    def test_https_url(self) -> None:
        assert ArxivFetcher._extract_arxiv_id("https://arxiv.org/abs/2603.19461") == "2603.19461"

    def test_pdf_url(self) -> None:
        assert ArxivFetcher._extract_arxiv_id("https://arxiv.org/pdf/2603.19461v2") == "2603.19461"

    def test_bare_id(self) -> None:
        assert ArxivFetcher._extract_arxiv_id("2603.19461") == "2603.19461"

    def test_old_format(self) -> None:
        assert ArxivFetcher._extract_arxiv_id("http://arxiv.org/abs/cs/0601001v1") == "cs/0601001"

    def test_strip_whitespace(self) -> None:
        assert ArxivFetcher._extract_arxiv_id("  2603.19461  ") == "2603.19461"


class TestBuildQueryUrl:
    def test_url_format(self) -> None:
        config = ArxivConfig(categories=["cs.AI"], max_results_per_category=50)
        fetcher = ArxivFetcher(config)
        url = fetcher._build_query_url("cs.AI")
        assert "cat%3Acs.AI" in url or "cat:cs.AI" in url
        assert "max_results=50" in url
        assert "sortBy=submittedDate" in url


class TestParseEntry:
    def test_valid_entry(self) -> None:
        config = ArxivConfig(categories=["cs.AI"])
        fetcher = ArxivFetcher(config)

        entry = {
            "id": "http://arxiv.org/abs/2603.12345v1",
            "title": "Test Paper Title",
            "summary": "This is the abstract of the test paper.",
            "authors": [{"name": "John Doe"}, {"name": "Jane Smith"}],
            "tags": [{"term": "cs.AI"}, {"term": "cs.LG"}],
            "published": "2026-03-28T10:00:00Z",
            "links": [
                {"href": "http://arxiv.org/abs/2603.12345v1", "rel": "alternate"},
                {"href": "http://arxiv.org/pdf/2603.12345v1", "type": "application/pdf"},
            ],
        }

        paper = fetcher._parse_entry(entry, fallback_category="cs.AI")
        assert paper is not None
        assert paper.arxiv_id == "2603.12345"
        assert paper.title == "Test Paper Title"
        assert paper.abstract == "This is the abstract of the test paper."
        assert paper.authors == ["John Doe", "Jane Smith"]
        assert "cs.AI" in paper.categories
        assert "cs.LG" in paper.categories
        assert paper.published.tzinfo is not None

    def test_missing_id_returns_none(self) -> None:
        config = ArxivConfig(categories=["cs.AI"])
        fetcher = ArxivFetcher(config)
        entry = {"title": "No ID Paper", "summary": "Abstract"}
        paper = fetcher._parse_entry(entry, fallback_category="cs.AI")
        assert paper is None

    def test_missing_authors_defaults(self) -> None:
        config = ArxivConfig(categories=["cs.AI"])
        fetcher = ArxivFetcher(config)
        entry = {
            "id": "http://arxiv.org/abs/2603.99999v1",
            "title": "No Authors",
            "summary": "Abstract",
            "authors": [],
            "tags": [],
            "published": "2026-03-28T10:00:00Z",
            "links": [],
        }
        paper = fetcher._parse_entry(entry, fallback_category="cs.AI")
        assert paper is not None
        assert paper.authors == ["Unknown"]
        assert "cs.AI" in paper.categories

    def test_fallback_category_added(self) -> None:
        config = ArxivConfig(categories=["cs.MA"])
        fetcher = ArxivFetcher(config)
        entry = {
            "id": "http://arxiv.org/abs/2603.11111v1",
            "title": "Test",
            "summary": "Test",
            "authors": [{"name": "A"}],
            "tags": [{"term": "cs.AI"}],
            "published": "2026-03-28T10:00:00Z",
            "links": [],
        }
        paper = fetcher._parse_entry(entry, fallback_category="cs.MA")
        assert paper is not None
        assert "cs.MA" in paper.categories
        assert "cs.AI" in paper.categories