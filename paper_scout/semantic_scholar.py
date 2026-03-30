from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from .models import S2Paper, S2Reference

_ARXIV_VERSION_RE = re.compile(r"v\d+$")


class SemanticScholarError(RuntimeError):
    """Raised when Semantic Scholar API calls fail unexpectedly."""


class SemanticScholarClient:
    BASE_URL = "https://api.semanticscholar.org"

    def __init__(
        self,
        request_timeout_seconds: float = 15.0,
        pause_seconds: float = 0.05,
    ) -> None:
        self.request_timeout_seconds = max(1.0, request_timeout_seconds)
        self.pause_seconds = max(0.0, pause_seconds)
        self._rate_lock = threading.Lock()
        self._last_request_ts = 0.0

    def get_paper(self, arxiv_id: str) -> S2Paper | None:
        normalized_id = _normalize_arxiv_id(arxiv_id)
        if not normalized_id:
            return None

        payload = self._request_json(
            f"/graph/v1/paper/ArXiv:{normalized_id}",
            params={
                "fields": ",".join(
                    [
                        "paperId",
                        "title",
                        "abstract",
                        "citationCount",
                        "referenceCount",
                        "authors.name",
                        "authors.affiliations",
                    ]
                )
            },
        )
        if payload is None:
            return None
        return _parse_s2_paper(payload)

    def get_references(self, arxiv_id: str) -> list[S2Reference]:
        normalized_id = _normalize_arxiv_id(arxiv_id)
        if not normalized_id:
            return []

        payload = self._request_json(
            f"/graph/v1/paper/ArXiv:{normalized_id}/references",
            params={"fields": "paperId,title,abstract,authors,year"},
        )
        if payload is None:
            return []

        data = payload.get("data") if isinstance(payload, dict) else payload
        if not isinstance(data, list):
            return []

        references: list[S2Reference] = []
        for item in data:
            target = (
                item.get("citedPaper") if isinstance(item, dict) and "citedPaper" in item else item
            )
            parsed = _parse_s2_reference(target)
            if parsed is not None:
                references.append(parsed)
        return references

    def get_author_affiliations(self, arxiv_id: str) -> dict[str, list[str]]:
        paper = self.get_paper(arxiv_id)
        if paper is None:
            return {}

        affiliations: dict[str, list[str]] = {}
        for author in paper.authors:
            if not isinstance(author, dict):
                continue
            name = str(author.get("name", "")).strip()
            if not name:
                continue
            raw_affiliations = author.get("affiliations", [])
            clean_affiliations = _coerce_string_list(raw_affiliations)
            if clean_affiliations:
                affiliations[name] = clean_affiliations
        return affiliations

    def get_reference_details(self, paper_id: str) -> S2Reference | None:
        normalized = paper_id.strip()
        if not normalized:
            return None

        payload = self._request_json(
            f"/graph/v1/paper/{urllib.parse.quote(normalized)}",
            params={"fields": "paperId,title,abstract,authors.name,year"},
        )
        if payload is None:
            return None
        return _parse_s2_reference(payload)

    def search_paper(self, query: str) -> S2Paper | None:
        clean_query = query.strip()
        if not clean_query:
            return None

        payload = self._request_json(
            "/graph/v1/paper/search",
            params={
                "query": clean_query,
                "limit": "1",
                "fields": ",".join(
                    [
                        "paperId",
                        "title",
                        "abstract",
                        "citationCount",
                        "referenceCount",
                        "authors.name",
                        "authors.affiliations",
                    ]
                ),
            },
        )
        if not isinstance(payload, dict):
            return None

        data = payload.get("data", [])
        if not isinstance(data, list) or not data:
            return None

        first = data[0]
        if not isinstance(first, dict):
            return None

        return _parse_s2_paper(first)

    def _request_json(self, path: str, params: dict[str, str] | None = None) -> Any | None:
        query = urllib.parse.urlencode(params or {})
        url = f"{self.BASE_URL}{path}"
        if query:
            url = f"{url}?{query}"

        self._wait_for_rate_slot()

        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "paper-scout/0.2 (+https://api.semanticscholar.org)",
                "Accept": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout_seconds) as response:
                status = getattr(response, "status", 200)
                if status == 404:
                    return None
                if status >= 400:
                    raise SemanticScholarError(f"Semantic Scholar HTTP status {status} for {url}.")
                body = response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise SemanticScholarError(
                f"Semantic Scholar request failed ({exc.code}) for {url}: {exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise SemanticScholarError(f"Semantic Scholar request failed for {url}: {exc}") from exc

        try:
            return json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SemanticScholarError(
                f"Semantic Scholar response was not valid JSON for {url}: {exc}"
            ) from exc

    def _wait_for_rate_slot(self) -> None:
        if self.pause_seconds <= 0:
            return

        with self._rate_lock:
            now = time.monotonic()
            elapsed = now - self._last_request_ts
            if elapsed < self.pause_seconds:
                time.sleep(self.pause_seconds - elapsed)
            self._last_request_ts = time.monotonic()


def _normalize_arxiv_id(arxiv_id: str) -> str:
    candidate = arxiv_id.strip()
    if not candidate:
        return ""
    for marker in ("/abs/", "/pdf/"):
        if marker in candidate:
            candidate = candidate.split(marker, 1)[1]
    candidate = candidate.removesuffix(".pdf").strip().strip("/")
    candidate = _ARXIV_VERSION_RE.sub("", candidate)
    return candidate


def _coerce_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        if isinstance(value, str):
            stripped = value.strip()
            return [stripped] if stripped else []
        return []

    cleaned: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            cleaned.append(text)
    return cleaned


def _coerce_authors(raw: Any) -> list[dict]:
    if not isinstance(raw, list):
        return []

    authors: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        authors.append(
            {
                "name": name,
                "affiliations": _coerce_string_list(item.get("affiliations", [])),
            }
        )
    return authors


def _extract_author_names(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    names: list[str] = []
    for item in raw:
        if isinstance(item, dict):
            name = str(item.get("name", "")).strip()
        else:
            name = str(item).strip()
        if name:
            names.append(name)
    return names


def _parse_s2_paper(payload: Any) -> S2Paper | None:
    if not isinstance(payload, dict):
        return None

    paper_id = str(payload.get("paperId", "")).strip()
    title = str(payload.get("title", "")).strip()
    if not paper_id or not title:
        return None

    abstract_raw = payload.get("abstract")
    abstract = (
        str(abstract_raw).strip()
        if isinstance(abstract_raw, str) and abstract_raw.strip()
        else None
    )

    return S2Paper(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        citation_count=_coerce_int(payload.get("citationCount"), default=0),
        reference_count=_coerce_int(payload.get("referenceCount"), default=0),
        authors=_coerce_authors(payload.get("authors", [])),
    )


def _parse_s2_reference(payload: Any) -> S2Reference | None:
    if not isinstance(payload, dict):
        return None

    paper_id = str(payload.get("paperId", "")).strip()
    title = str(payload.get("title", "")).strip()
    if not paper_id or not title:
        return None

    abstract_raw = payload.get("abstract")
    abstract = (
        str(abstract_raw).strip()
        if isinstance(abstract_raw, str) and abstract_raw.strip()
        else None
    )

    return S2Reference(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        authors=_extract_author_names(payload.get("authors", [])),
        year=_coerce_optional_int(payload.get("year")),
    )


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None