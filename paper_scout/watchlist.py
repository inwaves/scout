from __future__ import annotations

import re

from .config import WatchlistConfig
from .models import Paper, WatchlistMatch

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


class WatchlistMatcher:
    def __init__(self, config: WatchlistConfig) -> None:
        self.config = config
        self._author_patterns: list[tuple[str, str]] = [
            (author, _normalize(author)) for author in config.authors if _normalize(author)
        ]
        self._organization_patterns: list[tuple[str, bool, list[str]]] = []
        for org in config.organizations:
            aliases = [org.name, *org.aliases]
            normalized_aliases = _dedupe(
                [_normalize(alias) for alias in aliases if _normalize(alias)]
            )
            if normalized_aliases:
                self._organization_patterns.append(
                    (org.name, org.always_include, normalized_aliases)
                )

    def match_paper(
        self,
        paper: Paper,
        affiliations: dict[str, list[str]] | None = None,
    ) -> WatchlistMatch | None:
        normalized_paper_authors = [_normalize(author) for author in paper.authors if _normalize(author)]
        for watched_name, watched_normalized in self._author_patterns:
            if any(
                _name_matches(author_normalized, watched_normalized)
                for author_normalized in normalized_paper_authors
            ):
                return WatchlistMatch(match_type="author", matched_name=watched_name)

        if affiliations:
            flattened_affiliations: list[str] = []
            for items in affiliations.values():
                for item in items:
                    normalized = _normalize(item)
                    if normalized:
                        flattened_affiliations.append(normalized)

            for org_name, always_include, aliases in self._organization_patterns:
                for affiliation in flattened_affiliations:
                    if any(alias in affiliation or affiliation in alias for alias in aliases):
                        return WatchlistMatch(
                            match_type="organization",
                            matched_name=org_name,
                        )

        return None


def _normalize(value: str) -> str:
    lowered = value.strip().lower()
    if not lowered:
        return ""
    return " ".join(_NON_ALNUM_RE.sub(" ", lowered).split())


def _name_matches(candidate: str, target: str) -> bool:
    if not candidate or not target:
        return False
    if candidate == target:
        return True
    if candidate in target or target in candidate:
        return True

    candidate_tokens = candidate.split()
    target_tokens = target.split()
    return candidate_tokens == target_tokens


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped