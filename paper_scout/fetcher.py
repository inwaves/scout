from __future__ import annotations

import logging
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

import feedparser
from dateutil import parser as date_parser

from .config import ArxivConfig
from .models import Paper

ARXIV_API_URL = "http://export.arxiv.org/api/query"
_ARXIV_VERSION_RE = re.compile(r"v\d+$")
LOGGER = logging.getLogger(__name__)


class ArxivFetchError(RuntimeError):
    """Raised when arXiv feed retrieval fails."""


class ArxivFetcher:
    """Fetch recent papers from arXiv Atom feeds."""

    def __init__(self, config: ArxivConfig, logger: logging.Logger | None = None) -> None:
        self.config = config
        self._logger = logger or LOGGER

    def fetch_new_papers(self, since: datetime) -> list[Paper]:
        """
        Fetch papers newer than `since` and deduplicate across watched categories.
        """
        since_utc = since.astimezone(timezone.utc) if since.tzinfo else since.replace(
            tzinfo=timezone.utc
        )
        deduped: dict[str, Paper] = {}

        for index, category in enumerate(self.config.categories):
            try:
                entries = self._fetch_category_entries(category)
            except Exception as exc:
                self._logger.error("Failed to fetch category %s: %s", category, exc)
                continue

            for entry in entries:
                paper = self._parse_entry(entry, fallback_category=category)
                if paper is None:
                    continue
                if paper.published < since_utc:
                    continue

                existing = deduped.get(paper.arxiv_id)
                if existing:
                    existing.categories = sorted(
                        set(existing.categories).union(paper.categories)
                    )
                else:
                    deduped[paper.arxiv_id] = paper

            if index < len(self.config.categories) - 1 and self.config.query_pause_seconds > 0:
                time.sleep(self.config.query_pause_seconds)

        papers = sorted(deduped.values(), key=lambda item: item.published, reverse=True)
        self._logger.info(
            "Fetched %d unique papers across %d categories since %s.",
            len(papers),
            len(self.config.categories),
            since_utc.isoformat(),
        )
        return papers

    def _fetch_category_entries(self, category: str) -> list[Any]:
        url = self._build_query_url(category)
        last_error: Exception | None = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                # Fetch raw XML via urllib with configurable timeout, then
                # pass the content to feedparser for parsing. This ensures
                # request_timeout_seconds is actually enforced (feedparser's
                # own parse(url) does not accept a timeout parameter).
                request = urllib.request.Request(
                    url,
                    headers={"User-Agent": "paper-scout/0.1 (+https://arxiv.org)"},
                )
                with urllib.request.urlopen(
                    request, timeout=self.config.request_timeout_seconds
                ) as response:
                    status = response.status
                    if status >= 400:
                        raise ArxivFetchError(f"HTTP status {status} from arXiv.")
                    raw_xml = response.read()

                feed = feedparser.parse(raw_xml)
                if getattr(feed, "bozo", False) and not getattr(feed, "entries", None):
                    raise ArxivFetchError(f"Malformed feed payload: {feed.bozo_exception}")

                if getattr(feed, "bozo", False):
                    self._logger.warning(
                        "arXiv feed parser reported a recoverable issue for %s: %s",
                        category,
                        getattr(feed, "bozo_exception", "unknown parser issue"),
                    )

                return list(feed.entries)
            except ArxivFetchError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    break
                sleep_seconds = self.config.retry_backoff_seconds * attempt
                self._logger.warning(
                    "Fetch attempt %d/%d failed for category %s: %s (retry in %.1fs)",
                    attempt,
                    self.config.max_retries,
                    category,
                    exc,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)

        raise ArxivFetchError(
            f"Failed to fetch arXiv feed for category {category}: {last_error}"
        )

    def _build_query_url(self, category: str) -> str:
        params = {
            "search_query": f"cat:{category}",
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": str(self.config.max_results_per_category),
        }
        return f"{ARXIV_API_URL}?{urllib.parse.urlencode(params)}"

    def _parse_entry(self, entry: Any, fallback_category: str) -> Paper | None:
        try:
            raw_id = str(entry.get("id", "")).strip()
            arxiv_id = self._extract_arxiv_id(raw_id)
            if not arxiv_id:
                raise ValueError("missing arXiv identifier")

            title = _normalize_whitespace(str(entry.get("title", "")).strip()) or "(untitled)"
            abstract = _normalize_whitespace(str(entry.get("summary", "")).strip())

            authors = [
                str(author.get("name", "")).strip()
                for author in entry.get("authors", [])
                if str(author.get("name", "")).strip()
            ]
            if not authors:
                authors = ["Unknown"]

            categories = [
                str(tag.get("term", "")).strip()
                for tag in entry.get("tags", [])
                if str(tag.get("term", "")).strip()
            ]
            if fallback_category not in categories:
                categories.append(fallback_category)

            published_raw = entry.get("published") or entry.get("updated")
            if not published_raw:
                raise ValueError("missing published timestamp")

            published = date_parser.parse(str(published_raw))
            if published.tzinfo is None:
                published = published.replace(tzinfo=timezone.utc)
            else:
                published = published.astimezone(timezone.utc)

            url = f"https://arxiv.org/abs/{arxiv_id}"
            pdf_url = self._extract_pdf_url(entry, arxiv_id)

            return Paper(
                arxiv_id=arxiv_id,
                title=title,
                abstract=abstract,
                authors=authors,
                categories=sorted(set(categories)),
                published=published,
                url=url,
                pdf_url=pdf_url,
            )
        except Exception as exc:
            self._logger.debug("Skipping invalid arXiv entry: %s", exc, exc_info=True)
            return None

    @staticmethod
    def _extract_arxiv_id(raw_id: str) -> str:
        candidate = raw_id.strip()
        for marker in ("/abs/", "/pdf/"):
            if marker in candidate:
                candidate = candidate.split(marker, 1)[1]
        candidate = candidate.strip().removesuffix(".pdf").strip("/")
        candidate = _ARXIV_VERSION_RE.sub("", candidate)
        return candidate

    @staticmethod
    def _extract_pdf_url(entry: Any, arxiv_id: str) -> str:
        for link in entry.get("links", []):
            href = str(link.get("href", "")).strip()
            if not href:
                continue
            if "/pdf/" in href:
                if href.startswith("http"):
                    return href
                return f"https://arxiv.org{href}"
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())