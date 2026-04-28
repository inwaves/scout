from __future__ import annotations

from datetime import datetime, timezone

from paper_scout.config import WebSourceConfig, WebSourcesConfig
from paper_scout.web_fetcher import BUILTIN_SOURCES, WebFetcher


def _build_fetcher(
    source_type: str = "anthropic_news",
    *,
    fetch_page_metadata: bool = False,
    max_post_age_days: int | None = 120,
) -> WebFetcher:
    config = WebSourcesConfig(
        enabled=True,
        sources=[WebSourceConfig(type=source_type)],
        query_pause_seconds=0.0,
        fetch_page_metadata=fetch_page_metadata,
        max_items_per_source=10,
        max_post_age_days=max_post_age_days,
    )
    return WebFetcher(config)


class TestParseSitemapXml:
    def test_urlset_entries_parsed(self) -> None:
        fetcher = _build_fetcher()

        xml = b"""<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
          <url>
            <loc>https://www.anthropic.com/news/claude-4</loc>
            <lastmod>2026-03-28T12:00:00Z</lastmod>
          </url>
          <url>
            <loc>https://www.anthropic.com/research/constitutional-ai</loc>
          </url>
        </urlset>
        """

        entries = fetcher._parse_sitemap_xml(xml)
        assert len(entries) == 2
        assert entries[0][0] == "https://www.anthropic.com/news/claude-4"
        assert entries[0][1] == datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc)
        assert entries[1][0] == "https://www.anthropic.com/research/constitutional-ai"
        assert entries[1][1] is None

    def test_sitemap_index_entries_parsed(self) -> None:
        fetcher = _build_fetcher("openai")

        xml = b"""<?xml version="1.0" encoding="UTF-8"?>
        <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
          <sitemap>
            <loc>https://openai.com/sitemaps/research.xml</loc>
            <lastmod>2026-03-28</lastmod>
          </sitemap>
        </sitemapindex>
        """

        entries = fetcher._parse_sitemap_xml(xml)
        assert entries == [
            ("https://openai.com/sitemaps/research.xml", datetime(2026, 3, 28, tzinfo=timezone.utc))
        ]


class TestParseAlignmentIndex:
    def test_internal_year_links_only(self) -> None:
        fetcher = _build_fetcher("anthropic_alignment")

        html = """
        <html>
          <body>
            <div class="entry">
              <a href="/2025/03/model-organisms">Model Organisms</a>
              <p>Studying alignment properties in toy models.</p>
            </div>
            <div class="entry">
              <a href="https://alignment.anthropic.com/2024/12/constitutional-classifiers">
                Constitutional Classifiers
              </a>
              <p>Safety work on classifier-based oversight.</p>
            </div>
            <div class="entry">
              <a href="https://arxiv.org/abs/2603.12345">External paper</a>
            </div>
            <div class="entry">
              <a href="/about">About</a>
            </div>
          </body>
        </html>
        """

        entries = fetcher._parse_alignment_index(html, "https://alignment.anthropic.com")
        assert len(entries) == 2
        assert entries[0] == (
            "https://alignment.anthropic.com/2025/03/model-organisms",
            "Model Organisms",
            "Studying alignment properties in toy models.",
        )
        assert entries[1] == (
            "https://alignment.anthropic.com/2024/12/constitutional-classifiers",
            "Constitutional Classifiers",
            "Safety work on classifier-based oversight.",
        )


class TestMakePaperId:
    def test_slug_generation(self) -> None:
        fetcher = _build_fetcher()

        assert (
            fetcher._make_paper_id(
                BUILTIN_SOURCES["anthropic_news"],
                "https://www.anthropic.com/news/claude-4",
            )
            == "anthropic:claude-4"
        )
        assert (
            fetcher._make_paper_id(
                BUILTIN_SOURCES["anthropic_alignment"],
                "https://alignment.anthropic.com/2025/03/model-organisms",
            )
            == "anthropic-alignment:2025-03-model-organisms"
        )


class TestBuildPaper:
    def test_build_paper_sets_web_fields(self) -> None:
        fetcher = _build_fetcher("openai")
        published = datetime(2026, 3, 28, 10, 0, 0, tzinfo=timezone.utc)

        paper = fetcher._build_paper(
            source=BUILTIN_SOURCES["openai"],
            url="https://openai.com/research/test-post",
            title="Test Post",
            abstract="A research announcement.",
            published=published,
            pdf_url=None,
        )

        assert paper.arxiv_id == "openai:test-post"
        assert paper.title == "Test Post"
        assert paper.abstract == "A research announcement."
        assert paper.authors == ["OpenAI"]
        assert paper.categories == ["web", "openai"]
        assert paper.published == published
        assert paper.url == "https://openai.com/research/test-post"
        assert paper.pdf_url == "https://openai.com/research/test-post"
        assert paper.source_label == "OpenAI"


class TestFetchSitemapSource:
    def test_filters_by_path_prefix_and_date(self) -> None:
        fetcher = _build_fetcher("anthropic_news", fetch_page_metadata=False)
        source = BUILTIN_SOURCES["anthropic_news"]
        since = datetime(2026, 3, 1, tzinfo=timezone.utc)

        fetcher._collect_sitemap_entries = lambda sitemap_url: [  # type: ignore[method-assign]
            (
                "https://www.anthropic.com/news/claude-4",
                datetime(2026, 3, 28, tzinfo=timezone.utc),
            ),
            (
                "https://www.anthropic.com/research/interp-report",
                datetime(2026, 3, 15, tzinfo=timezone.utc),
            ),
            (
                "https://www.anthropic.com/blog/ignored",
                datetime(2026, 3, 20, tzinfo=timezone.utc),
            ),
            (
                "https://www.anthropic.com/news/old-post",
                datetime(2026, 2, 1, tzinfo=timezone.utc),
            ),
            (
                "https://www.anthropic.com/research/no-lastmod",
                None,
            ),
        ]

        papers = fetcher._fetch_sitemap_source(source, since)
        paper_ids = {paper.arxiv_id for paper in papers}
        urls = {paper.url for paper in papers}

        assert paper_ids == {
            "anthropic:claude-4",
            "anthropic:interp-report",
            "anthropic:no-lastmod",
        }
        assert "https://www.anthropic.com/blog/ignored" not in urls
        assert "https://www.anthropic.com/news/old-post" not in urls


class TestFetchPageMetadata:
    def test_placeholder_title_falls_back_to_h1(self) -> None:
        fetcher = _build_fetcher(fetch_page_metadata=True)

        html = """<html><head><title>Untitled</title></head>
        <body><d-title><h1>Real Paper Title</h1></d-title>
        <a href="https://arxiv.org/abs/2602.22755">arXiv</a>
        </body></html>"""

        fetcher._fetch_text = lambda url: html  # type: ignore[method-assign]
        title, description, pdf_url, arxiv_id = fetcher._fetch_page_metadata(
            "https://arxiv.org/abs/2602.22755"
        )
        assert title == "Real Paper Title"
        assert arxiv_id == "2602.22755"

    def test_extracts_arxiv_id_from_pdf_link(self) -> None:
        fetcher = _build_fetcher(fetch_page_metadata=True)

        html = """<html><head><title>Good Title</title></head>
        <body><a href="https://arxiv.org/pdf/2604.07729v1">PDF</a></body></html>"""

        fetcher._fetch_text = lambda url: html  # type: ignore[method-assign]
        title, description, pdf_url, arxiv_id = fetcher._fetch_page_metadata(
            "https://arxiv.org/abs/2604.07729"
        )
        assert title == "Good Title"
        assert arxiv_id == "2604.07729"

    def test_ignores_arxiv_citations_on_non_arxiv_pages(self) -> None:
        fetcher = _build_fetcher(fetch_page_metadata=True)

        html = """<html><head><title>Blog Post</title></head>
        <body><a href="https://arxiv.org/abs/2211.03540">Cited paper</a></body></html>"""

        fetcher._fetch_text = lambda url: html  # type: ignore[method-assign]
        title, description, pdf_url, arxiv_id = fetcher._fetch_page_metadata(
            "https://alignment.anthropic.com/2026/some-post"
        )
        assert title == "Blog Post"
        assert arxiv_id == ""

    def test_no_arxiv_link_returns_empty_id(self) -> None:
        fetcher = _build_fetcher(fetch_page_metadata=True)

        html = """<html><head><title>Blog Post</title></head>
        <body><a href="/other-page">link</a></body></html>"""

        fetcher._fetch_text = lambda url: html  # type: ignore[method-assign]
        title, description, pdf_url, arxiv_id = fetcher._fetch_page_metadata(
            "https://example.com/page"
        )
        assert title == "Blog Post"
        assert arxiv_id == ""

    def test_extracts_published_date_from_meta(self) -> None:
        fetcher = _build_fetcher(fetch_page_metadata=True)

        html = """<html><head>
        <title>Blog Post</title>
        <meta property="article:published_time" content="2025-01-15T12:30:00Z">
        </head><body></body></html>"""

        fetcher._fetch_text = lambda url: html  # type: ignore[method-assign]
        title, description, pdf_url, arxiv_id, published = fetcher._fetch_page_metadata_details(
            "https://openai.com/index/blog-post"
        )
        assert title == "Blog Post"
        assert arxiv_id == ""
        assert published == datetime(2025, 1, 15, 12, 30, 0, tzinfo=timezone.utc)


class TestWebPostAgeGate:
    def test_page_published_date_can_drop_stale_sitemap_refresh(self) -> None:
        fetcher = _build_fetcher(
            "openai",
            fetch_page_metadata=True,
            max_post_age_days=120,
        )
        source = BUILTIN_SOURCES["openai"]
        since = datetime(2026, 4, 28, 0, 0, 0, tzinfo=timezone.utc)

        fetcher._collect_sitemap_entries = lambda sitemap_url: [  # type: ignore[method-assign]
            (
                "https://openai.com/index/old-post",
                datetime(2026, 4, 28, 0, 0, 0, tzinfo=timezone.utc),
            )
        ]
        fetcher._fetch_text = lambda url: """<html><head>
        <title>Old Post</title>
        <meta property="article:published_time" content="2025-01-15T12:30:00Z">
        </head><body></body></html>"""  # type: ignore[method-assign]

        assert fetcher._fetch_sitemap_source(source, since) == []


class TestBuildPaperArxivOverride:
    def test_arxiv_id_override_replaces_synthetic_id(self) -> None:
        fetcher = _build_fetcher()
        paper = fetcher._build_paper(
            source=BUILTIN_SOURCES["anthropic_alignment"],
            url="https://alignment.anthropic.com/2026/auditbench",
            title="AuditBench",
            abstract="A benchmark.",
            published=datetime(2026, 3, 1, tzinfo=timezone.utc),
            arxiv_id_override="2602.22755",
        )
        assert paper.arxiv_id == "2602.22755"

    def test_empty_override_uses_synthetic_id(self) -> None:
        fetcher = _build_fetcher()
        paper = fetcher._build_paper(
            source=BUILTIN_SOURCES["anthropic_news"],
            url="https://www.anthropic.com/news/some-post",
            title="Some Post",
            abstract="Description.",
            published=datetime(2026, 3, 1, tzinfo=timezone.utc),
            arxiv_id_override="",
        )
        assert paper.arxiv_id == "anthropic:some-post"
