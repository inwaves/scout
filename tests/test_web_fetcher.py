from __future__ import annotations

from datetime import datetime, timezone

from paper_scout.config import WebSourceConfig, WebSourcesConfig
from paper_scout.web_fetcher import BUILTIN_SOURCES, WebFetcher


def _build_fetcher(
    source_type: str = "anthropic_news",
    *,
    fetch_page_metadata: bool = False,
) -> WebFetcher:
    config = WebSourcesConfig(
        enabled=True,
        sources=[WebSourceConfig(type=source_type)],
        query_pause_seconds=0.0,
        fetch_page_metadata=fetch_page_metadata,
        max_items_per_source=10,
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