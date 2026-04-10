from __future__ import annotations

import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from bs4 import BeautifulSoup
from dateutil import parser as date_parser

from .config import WebSourcesConfig
from .models import Paper

LOGGER = logging.getLogger(__name__)
_USER_AGENT = (
    "Mozilla/5.0 (compatible; scout/0.1; +https://github.com/inwaves/scout)"
)
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_SPACE_RE = re.compile(r"\s+")
_YEAR_PATH_RE = re.compile(r"^/20\d{2}/")


class WebFetchError(RuntimeError):
    """Raised when a web source cannot be fetched or parsed."""


@dataclass(slots=True)
class WebSourceDef:
    name: str
    source_type: str
    base_url: str
    index_url: str
    path_prefixes: list[str]
    org_name: str
    id_prefix: str
    source_label: str


BUILTIN_SOURCES: dict[str, WebSourceDef] = {
    "anthropic_alignment": WebSourceDef(
        name="Anthropic Alignment Blog",
        source_type="html_index",
        base_url="https://alignment.anthropic.com",
        index_url="https://alignment.anthropic.com",
        path_prefixes=["/2024/", "/2025/", "/2026/"],
        org_name="Anthropic",
        id_prefix="anthropic-alignment",
        source_label="Anthropic Alignment",
    ),
    "anthropic_news": WebSourceDef(
        name="Anthropic News & Research",
        source_type="sitemap",
        base_url="https://www.anthropic.com",
        index_url="https://www.anthropic.com/sitemap.xml",
        path_prefixes=["/news/", "/research/"],
        org_name="Anthropic",
        id_prefix="anthropic",
        source_label="Anthropic",
    ),
    "openai": WebSourceDef(
        name="OpenAI Research & Index",
        source_type="sitemap",
        base_url="https://openai.com",
        index_url="https://openai.com/sitemap.xml",
        path_prefixes=["/index/", "/research/"],
        org_name="OpenAI",
        id_prefix="openai",
        source_label="OpenAI",
    ),
    "deepmind": WebSourceDef(
        name="Google DeepMind Blog",
        source_type="sitemap",
        base_url="https://deepmind.google",
        index_url="https://deepmind.google/sitemap.xml",
        path_prefixes=["/discover/blog/", "/research/publications/"],
        org_name="Google DeepMind",
        id_prefix="deepmind",
        source_label="Google DeepMind",
    ),
}


class WebFetcher:
    """Fetch recent web posts from configured built-in sources."""

    def __init__(
        self,
        config: WebSourcesConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self._logger = logger or LOGGER
        self._last_request_started_at = 0.0

    def fetch_new_posts(self, since: datetime) -> list[Paper]:
        if not self.config.enabled:
            return []

        since_utc = _ensure_utc(since)
        enabled_sources = [source for source in self.config.sources if source.enabled]
        if not enabled_sources:
            self._logger.info("Web sources enabled but no source types are configured.")
            return []

        deduped: dict[str, Paper] = {}

        for configured_source in enabled_sources:
            source = BUILTIN_SOURCES.get(configured_source.type)
            if source is None:
                self._logger.warning("Skipping unknown web source type: %s", configured_source.type)
                continue

            try:
                if source.source_type == "sitemap":
                    fetched = self._fetch_sitemap_source(source, since_utc)
                elif source.source_type == "html_index":
                    fetched = self._fetch_html_index_source(source, since_utc)
                else:
                    raise WebFetchError(f"Unsupported web source type: {source.source_type}")
            except Exception as exc:
                self._logger.error("Web source %s failed: %s", source.name, exc)
                continue

            for paper in fetched:
                deduped.setdefault(paper.arxiv_id, paper)

            self._logger.info("Web source %s: %d post(s) fetched.", source.name, len(fetched))

        papers = sorted(deduped.values(), key=lambda item: item.published, reverse=True)
        self._logger.info(
            "Fetched %d unique web post(s) across %d configured source(s).",
            len(papers),
            len(enabled_sources),
        )
        return papers

    def _fetch_sitemap_source(self, source: WebSourceDef, since: datetime) -> list[Paper]:
        entries = self._collect_sitemap_entries(source.index_url)

        candidates: list[tuple[str, datetime, datetime]] = []
        seen_urls: set[str] = set()

        for url, lastmod in entries:
            canonical_url = _canonicalize_url(url)
            if not canonical_url or canonical_url in seen_urls:
                continue
            seen_urls.add(canonical_url)

            if not _url_matches_source(canonical_url, source):
                continue

            include, published, sort_key = self._select_candidate_timestamp(
                canonical_url,
                lastmod,
                since,
            )
            if not include:
                continue

            candidates.append((canonical_url, published, sort_key))

        candidates.sort(key=lambda item: item[2], reverse=True)
        candidates = candidates[: self.config.max_items_per_source]

        papers: list[Paper] = []
        for url, published, _sort_key in candidates:
            title = _title_from_url(url)
            abstract = ""
            pdf_url: str | None = None

            if self.config.fetch_page_metadata:
                try:
                    meta_title, meta_description, meta_pdf_url = self._fetch_page_metadata(url)
                except Exception as exc:
                    self._logger.debug("Page metadata fetch failed for %s: %s", url, exc)
                else:
                    title = meta_title or title
                    abstract = meta_description or abstract
                    pdf_url = meta_pdf_url

            papers.append(
                self._build_paper(
                    source=source,
                    url=url,
                    title=title,
                    abstract=abstract,
                    published=published,
                    pdf_url=pdf_url,
                )
            )

        return papers

    def _fetch_html_index_source(self, source: WebSourceDef, since: datetime) -> list[Paper]:
        html = self._fetch_text(source.index_url)
        discovered = self._parse_alignment_index(html, source.base_url)

        candidates: list[tuple[str, str, str, datetime, datetime]] = []
        seen_urls: set[str] = set()

        for url, title_hint, description_hint in discovered:
            canonical_url = _canonicalize_url(url)
            if not canonical_url or canonical_url in seen_urls:
                continue
            seen_urls.add(canonical_url)

            if not _url_matches_source(canonical_url, source):
                continue

            include, published, sort_key = self._select_candidate_timestamp(
                canonical_url,
                None,
                since,
            )
            if not include:
                continue

            candidates.append((canonical_url, title_hint, description_hint, published, sort_key))

        candidates.sort(key=lambda item: item[4], reverse=True)
        candidates = candidates[: self.config.max_items_per_source]

        papers: list[Paper] = []
        for url, title_hint, description_hint, published, _sort_key in candidates:
            title = title_hint or _title_from_url(url)
            abstract = description_hint or ""
            pdf_url: str | None = None

            if self.config.fetch_page_metadata:
                try:
                    meta_title, meta_description, meta_pdf_url = self._fetch_page_metadata(url)
                except Exception as exc:
                    self._logger.debug("Page metadata fetch failed for %s: %s", url, exc)
                else:
                    title = meta_title or title
                    abstract = meta_description or abstract
                    pdf_url = meta_pdf_url

            papers.append(
                self._build_paper(
                    source=source,
                    url=url,
                    title=title,
                    abstract=abstract,
                    published=published,
                    pdf_url=pdf_url,
                )
            )

        return papers

    def _collect_sitemap_entries(self, sitemap_url: str) -> list[tuple[str, datetime | None]]:
        return self._collect_sitemap_entries_recursive(sitemap_url, seen_urls=set(), depth=0)

    def _collect_sitemap_entries_recursive(
        self,
        sitemap_url: str,
        *,
        seen_urls: set[str],
        depth: int,
    ) -> list[tuple[str, datetime | None]]:
        if depth > 5:
            raise WebFetchError(f"Sitemap recursion depth exceeded while fetching {sitemap_url}")

        canonical_sitemap_url = _canonicalize_url(sitemap_url)
        if canonical_sitemap_url in seen_urls:
            return []
        seen_urls.add(canonical_sitemap_url)

        xml_bytes = self._fetch_bytes(sitemap_url)
        root_kind, entries = self._parse_sitemap_document(xml_bytes)

        if root_kind == "sitemapindex":
            collected: list[tuple[str, datetime | None]] = []
            for nested_sitemap_url, _nested_lastmod in entries:
                try:
                    collected.extend(
                        self._collect_sitemap_entries_recursive(
                            nested_sitemap_url,
                            seen_urls=seen_urls,
                            depth=depth + 1,
                        )
                    )
                except Exception as exc:
                    self._logger.warning(
                        "Skipping nested sitemap %s due to error: %s",
                        nested_sitemap_url,
                        exc,
                    )
            return collected

        if root_kind != "urlset":
            raise WebFetchError(f"Unsupported sitemap root element: {root_kind}")

        return entries

    def _parse_sitemap_xml(self, xml_bytes: bytes) -> list[tuple[str, datetime | None]]:
        return self._parse_sitemap_document(xml_bytes)[1]

    def _parse_sitemap_document(
        self,
        xml_bytes: bytes,
    ) -> tuple[str, list[tuple[str, datetime | None]]]:
        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError as exc:
            raise WebFetchError(f"Invalid sitemap XML: {exc}") from exc

        root_kind = _local_name(root.tag)
        if root_kind not in {"urlset", "sitemapindex"}:
            raise WebFetchError(f"Unexpected sitemap root element: {root_kind}")

        item_tag = "url" if root_kind == "urlset" else "sitemap"
        entries: list[tuple[str, datetime | None]] = []

        for item in root:
            if _local_name(item.tag) != item_tag:
                continue

            loc = ""
            lastmod: datetime | None = None

            for child in item:
                child_name = _local_name(child.tag)
                text = (child.text or "").strip()
                if child_name == "loc":
                    loc = text
                elif child_name == "lastmod" and text:
                    lastmod = _parse_datetime(text)

            if loc:
                entries.append((_canonicalize_url(loc), lastmod))

        return root_kind, entries

    def _parse_alignment_index(self, html: str, base_url: str) -> list[tuple[str, str, str]]:
        soup = BeautifulSoup(html, "html.parser")
        base_host = urllib.parse.urlparse(base_url).netloc.lower()

        results: list[tuple[str, str, str]] = []
        seen_urls: set[str] = set()

        for link in soup.find_all("a", href=True):
            href = str(link.get("href", "")).strip()
            if not href:
                continue

            absolute_url = _canonicalize_url(urllib.parse.urljoin(base_url, href))
            parsed = urllib.parse.urlparse(absolute_url)

            if parsed.netloc.lower() != base_host:
                continue
            if not _YEAR_PATH_RE.match(parsed.path or "/"):
                continue
            if absolute_url in seen_urls:
                continue

            seen_urls.add(absolute_url)
            title = _normalize_whitespace(link.get_text(" ", strip=True)) or _title_from_url(
                absolute_url
            )
            description = _extract_link_description(link, title)
            results.append((absolute_url, title, description))

        return results

    def _fetch_page_metadata(self, url: str) -> tuple[str, str, str | None]:
        html = self._fetch_text(url)
        soup = BeautifulSoup(html, "html.parser")

        title = _find_first_meta_content(
            soup,
            [
                {"property": "og:title"},
                {"name": "twitter:title"},
            ],
        )
        if not title:
            title_tag = soup.find("title")
            if title_tag is not None:
                title = _normalize_whitespace(title_tag.get_text(" ", strip=True))

        description = _find_first_meta_content(
            soup,
            [
                {"name": "description"},
                {"property": "og:description"},
                {"name": "twitter:description"},
            ],
        )

        pdf_url: str | None = None
        for tag in soup.find_all(["a", "link"], href=True):
            href = str(tag.get("href", "")).strip()
            if not href:
                continue
            absolute_href = urllib.parse.urljoin(url, href)
            if _looks_like_pdf_url(absolute_href):
                pdf_url = _canonicalize_url(absolute_href)
                break

        return title, description, pdf_url

    def _make_paper_id(self, source: WebSourceDef, url: str) -> str:
        parsed = urllib.parse.urlparse(url)
        path = parsed.path or "/"
        segments = [urllib.parse.unquote(segment) for segment in path.split("/") if segment]

        matched_prefix_segments: list[str] = []
        for prefix in source.path_prefixes:
            normalized_prefix = prefix.rstrip("/")
            if not normalized_prefix:
                continue
            if _path_has_prefix(path, prefix):
                matched_prefix_segments = [
                    urllib.parse.unquote(segment) for segment in normalized_prefix.split("/") if segment
                ]
                break

        slug_segments = list(segments)
        if matched_prefix_segments:
            year_only_prefix = (
                len(matched_prefix_segments) == 1
                and matched_prefix_segments[0].isdigit()
                and len(matched_prefix_segments[0]) == 4
            )
            if not year_only_prefix and len(segments) > len(matched_prefix_segments):
                slug_segments = segments[len(matched_prefix_segments) :]

        slug = _slugify("-".join(slug_segments))
        if not slug:
            slug = _slugify(parsed.netloc) or "web-post"

        return f"{source.id_prefix}:{slug}"

    def _build_paper(
        self,
        *,
        source: WebSourceDef,
        url: str,
        title: str,
        abstract: str,
        published: datetime | None,
        pdf_url: str | None = None,
    ) -> Paper:
        normalized_url = _canonicalize_url(url)
        normalized_pdf_url = _canonicalize_url(pdf_url) if pdf_url else normalized_url
        published_utc = _ensure_utc(published) if published is not None else datetime.now(
            timezone.utc
        )
        normalized_title = _normalize_whitespace(title) or _title_from_url(normalized_url)
        if not _normalize_whitespace(title):
            self._logger.warning(
                "No title extracted for %s — using fallback: %s", normalized_url, normalized_title,
            )
        normalized_abstract = _normalize_whitespace(abstract) or (
            f"Web post from {source.source_label or source.org_name}."
        )

        return Paper(
            arxiv_id=self._make_paper_id(source, normalized_url),
            title=normalized_title,
            abstract=normalized_abstract,
            authors=[source.org_name],
            categories=["web", source.id_prefix],
            published=published_utc,
            url=normalized_url,
            pdf_url=normalized_pdf_url,
            source_label=source.source_label,
        )

    def _select_candidate_timestamp(
        self,
        url: str,
        lastmod: datetime | None,
        since: datetime,
    ) -> tuple[bool, datetime, datetime]:
        if lastmod is not None:
            lastmod_utc = _ensure_utc(lastmod)
            return lastmod_utc >= since, lastmod_utc, lastmod_utc

        inferred_range = _infer_date_range_from_url(url)
        if inferred_range is not None:
            published, latest_possible = inferred_range
            return latest_possible >= since, published, latest_possible

        assumed_recent = datetime.now(timezone.utc)
        return True, assumed_recent, assumed_recent

    def _fetch_bytes(self, url: str) -> bytes:
        last_error: Exception | None = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                self._respect_rate_limit()
                request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
                with urllib.request.urlopen(
                    request,
                    timeout=self.config.request_timeout_seconds,
                ) as response:
                    status = getattr(response, "status", 200)
                    if status >= 400:
                        raise urllib.error.HTTPError(
                            url,
                            status,
                            f"HTTP {status}",
                            getattr(response, "headers", {}),
                            None,
                        )
                    return response.read()
            except urllib.error.HTTPError as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    break
                sleep_seconds = self.config.retry_backoff_seconds * attempt
                if exc.code == 429:
                    sleep_seconds *= 2
                sleep_seconds = min(120.0, sleep_seconds)
                self._logger.warning(
                    "Web fetch attempt %d/%d failed for %s: HTTP %s (retry in %.1fs)",
                    attempt,
                    self.config.max_retries,
                    url,
                    exc.code,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)
            except Exception as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    break
                sleep_seconds = min(120.0, self.config.retry_backoff_seconds * attempt)
                self._logger.warning(
                    "Web fetch attempt %d/%d failed for %s: %s (retry in %.1fs)",
                    attempt,
                    self.config.max_retries,
                    url,
                    exc,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)

        raise WebFetchError(
            f"Failed to fetch {url} after {self.config.max_retries} attempt(s): {last_error}"
        )

    def _fetch_text(self, url: str) -> str:
        return self._fetch_bytes(url).decode("utf-8", errors="replace")

    def _respect_rate_limit(self) -> None:
        pause_seconds = max(0.0, float(self.config.query_pause_seconds))
        if pause_seconds <= 0.0:
            self._last_request_started_at = time.monotonic()
            return

        now = time.monotonic()
        if self._last_request_started_at > 0.0:
            elapsed = now - self._last_request_started_at
            if elapsed < pause_seconds:
                time.sleep(pause_seconds - elapsed)

        self._last_request_started_at = time.monotonic()


def _normalize_whitespace(text: str) -> str:
    return _SPACE_RE.sub(" ", str(text).strip()).strip()


def _slugify(text: str) -> str:
    return _SLUG_RE.sub("-", text.strip().lower()).strip("-")


def _local_name(tag: str) -> str:
    return str(tag).rsplit("}", 1)[-1].lower()


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_datetime(value: str) -> datetime | None:
    text = str(value).strip()
    if not text:
        return None

    try:
        parsed = date_parser.parse(text)
    except (TypeError, ValueError, OverflowError):
        return None

    return _ensure_utc(parsed)


def _path_has_prefix(path: str, prefix: str) -> bool:
    normalized_path = path or "/"
    normalized_prefix = prefix.rstrip("/")
    if not normalized_prefix:
        return True
    return normalized_path == normalized_prefix or normalized_path.startswith(
        normalized_prefix + "/"
    )


def _url_matches_source(url: str, source: WebSourceDef) -> bool:
    parsed = urllib.parse.urlparse(url)
    base = urllib.parse.urlparse(source.base_url)
    if parsed.netloc.lower() != base.netloc.lower():
        return False
    return any(_path_has_prefix(parsed.path or "/", prefix) for prefix in source.path_prefixes)


def _infer_date_range_from_url(url: str) -> tuple[datetime, datetime] | None:
    parsed = urllib.parse.urlparse(url)
    segments = [urllib.parse.unquote(segment) for segment in parsed.path.split("/") if segment]

    for index, segment in enumerate(segments):
        if not (segment.isdigit() and len(segment) == 4):
            continue

        year = int(segment)
        if year < 2000 or year > 2100:
            continue

        month: int | None = None
        day: int | None = None

        if index + 1 < len(segments) and segments[index + 1].isdigit():
            month_value = int(segments[index + 1])
            if 1 <= month_value <= 12:
                month = month_value

        if month is not None and index + 2 < len(segments) and segments[index + 2].isdigit():
            day_value = int(segments[index + 2])
            if 1 <= day_value <= 31:
                day = day_value

        try:
            if month is None:
                start = datetime(year, 1, 1, tzinfo=timezone.utc)
                end = datetime(year + 1, 1, 1, tzinfo=timezone.utc) - timedelta(
                    microseconds=1
                )
                return start, end

            if day is None:
                start = datetime(year, month, 1, tzinfo=timezone.utc)
                if month == 12:
                    next_month = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
                else:
                    next_month = datetime(year, month + 1, 1, tzinfo=timezone.utc)
                end = next_month - timedelta(microseconds=1)
                return start, end

            start = datetime(year, month, day, tzinfo=timezone.utc)
            end = start + timedelta(days=1) - timedelta(microseconds=1)
            return start, end
        except ValueError:
            continue

    return None


def _title_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    segments = [
        urllib.parse.unquote(seg).strip()
        for seg in (parsed.path or "/").rstrip("/").split("/")
        if seg.strip()
    ]

    # Walk segments from the end, skip pure year/number segments.
    slug = ""
    for seg in reversed(segments):
        cleaned = seg.removesuffix(".pdf").removesuffix(".html")
        if cleaned and not cleaned.isdigit():
            slug = cleaned
            break

    if slug:
        words = [part for part in re.split(r"[-_]+", slug) if part]
        if words:
            return " ".join(word.capitalize() for word in words)

    # Last resort: use the domain name so the alert is at least identifiable.
    domain = parsed.netloc.lower().removeprefix("www.")
    return f"Post on {domain}" if domain else "Untitled Web Post"


def _canonicalize_url(url: str | None) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""

    parsed = urllib.parse.urlparse(raw)
    if not parsed.scheme and not parsed.netloc:
        path = parsed.path or "/"
        if path != "/":
            path = path.rstrip("/") or "/"
        return urllib.parse.urlunparse(("", "", path, parsed.params, parsed.query, ""))

    scheme = parsed.scheme or "https"
    netloc = parsed.netloc.lower()
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/") or "/"

    return urllib.parse.urlunparse((scheme, netloc, path, parsed.params, parsed.query, ""))


def _looks_like_pdf_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(str(url).strip())
    return parsed.path.lower().endswith(".pdf")


def _extract_link_description(link: object, title: str) -> str:
    parent = getattr(link, "parent", None)
    if parent is not None and hasattr(parent, "get_text"):
        parent_text = _normalize_whitespace(parent.get_text(" ", strip=True))
        if parent_text:
            cleaned = parent_text.replace(title, "", 1).strip(" -–—:|")
            if cleaned and cleaned != title:
                return cleaned

    sibling = getattr(link, "next_sibling", None)
    while sibling is not None:
        if hasattr(sibling, "get_text"):
            text = _normalize_whitespace(sibling.get_text(" ", strip=True))
        else:
            text = _normalize_whitespace(str(sibling))
        cleaned = text.strip(" -–—:|")
        if cleaned and cleaned != title:
            return cleaned
        sibling = getattr(sibling, "next_sibling", None)

    return ""


def _find_first_meta_content(soup: BeautifulSoup, attr_options: list[dict[str, str]]) -> str:
    for attrs in attr_options:
        tag = soup.find("meta", attrs=attrs)
        if tag is None:
            continue
        content = _normalize_whitespace(str(tag.get("content", "")))
        if content:
            return content
    return ""