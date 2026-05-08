from __future__ import annotations

import base64
import hmac
import html
import json
import logging
import re
import sqlite3
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import parse_qs, quote, urlsplit

from .config import FeedbackConfig
from .models import Paper, ScoredPaper

LOGGER = logging.getLogger(__name__)

_TOKEN_VERSION = 1
_TOKEN_SEPARATOR = "."
_STOPWORDS = {
    "about",
    "across",
    "after",
    "against",
    "agent",
    "agents",
    "among",
    "analysis",
    "approach",
    "based",
    "benchmark",
    "benchmarks",
    "can",
    "data",
    "from",
    "into",
    "large",
    "language",
    "learning",
    "llm",
    "llms",
    "model",
    "models",
    "paper",
    "study",
    "their",
    "these",
    "this",
    "using",
    "with",
}
_WORD_RE = re.compile(r"[a-z0-9]{4,}")


class FeedbackError(RuntimeError):
    """Raised when feedback configuration or persistence fails."""


@dataclass(slots=True)
class FeedbackVote:
    token_id: str
    paper_id: str
    title: str
    url: str
    source_label: str
    categories: list[str]
    authors: list[str]
    base_score: float
    novelty_signal: str
    digest_date: str
    vote: int
    updated_at: str


@dataclass(slots=True)
class FeedbackStats:
    positive_votes: int
    negative_votes: int
    total_votes: int


class FeedbackTokenSigner:
    def __init__(self, secret: str) -> None:
        normalized_secret = secret.strip()
        if not normalized_secret:
            raise FeedbackError("Feedback signing secret must not be empty.")
        self._secret = normalized_secret.encode("utf-8")

    def dumps(self, payload: dict[str, Any]) -> str:
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        signature = hmac.new(self._secret, body, sha256).digest()
        return (
            _urlsafe_b64encode(body)
            + _TOKEN_SEPARATOR
            + _urlsafe_b64encode(signature)
        )

    def loads(self, token: str) -> dict[str, Any]:
        parts = token.split(_TOKEN_SEPARATOR, 1)
        if len(parts) != 2:
            raise FeedbackError("Malformed feedback token.")

        body = _urlsafe_b64decode(parts[0])
        signature = _urlsafe_b64decode(parts[1])
        expected = hmac.new(self._secret, body, sha256).digest()
        if not hmac.compare_digest(signature, expected):
            raise FeedbackError("Invalid feedback token signature.")

        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise FeedbackError("Feedback token payload is not valid JSON.") from exc

        if not isinstance(payload, dict):
            raise FeedbackError("Feedback token payload must be a JSON object.")

        version = payload.get("v")
        if version != _TOKEN_VERSION:
            raise FeedbackError(f"Unsupported feedback token version: {version!r}")
        return payload


class FeedbackStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()

    def initialize(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS feedback_votes (
                        token_id TEXT PRIMARY KEY,
                        paper_id TEXT NOT NULL,
                        title TEXT NOT NULL,
                        url TEXT NOT NULL,
                        source_label TEXT NOT NULL,
                        categories_json TEXT NOT NULL,
                        authors_json TEXT NOT NULL,
                        base_score REAL NOT NULL,
                        novelty_signal TEXT NOT NULL,
                        digest_date TEXT NOT NULL,
                        vote INTEGER NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
        except OSError as exc:
            raise FeedbackError(f"Failed to initialize feedback store at {self.path}: {exc}") from exc

    def record_vote(self, vote: FeedbackVote) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO feedback_votes (
                    token_id,
                    paper_id,
                    title,
                    url,
                    source_label,
                    categories_json,
                    authors_json,
                    base_score,
                    novelty_signal,
                    digest_date,
                    vote,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(token_id) DO UPDATE SET
                    paper_id=excluded.paper_id,
                    title=excluded.title,
                    url=excluded.url,
                    source_label=excluded.source_label,
                    categories_json=excluded.categories_json,
                    authors_json=excluded.authors_json,
                    base_score=excluded.base_score,
                    novelty_signal=excluded.novelty_signal,
                    digest_date=excluded.digest_date,
                    vote=excluded.vote,
                    updated_at=excluded.updated_at
                """,
                (
                    vote.token_id,
                    vote.paper_id,
                    vote.title,
                    vote.url,
                    vote.source_label,
                    json.dumps(vote.categories),
                    json.dumps(vote.authors),
                    vote.base_score,
                    vote.novelty_signal,
                    vote.digest_date,
                    vote.vote,
                    vote.updated_at,
                ),
            )

    def list_votes(self) -> list[FeedbackVote]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    token_id,
                    paper_id,
                    title,
                    url,
                    source_label,
                    categories_json,
                    authors_json,
                    base_score,
                    novelty_signal,
                    digest_date,
                    vote,
                    updated_at
                FROM feedback_votes
                ORDER BY updated_at DESC
                """
            ).fetchall()

        votes: list[FeedbackVote] = []
        for row in rows:
            votes.append(
                FeedbackVote(
                    token_id=str(row["token_id"]),
                    paper_id=str(row["paper_id"]),
                    title=str(row["title"]),
                    url=str(row["url"]),
                    source_label=str(row["source_label"]),
                    categories=_safe_json_list(row["categories_json"]),
                    authors=_safe_json_list(row["authors_json"]),
                    base_score=float(row["base_score"]),
                    novelty_signal=str(row["novelty_signal"]),
                    digest_date=str(row["digest_date"]),
                    vote=int(row["vote"]),
                    updated_at=str(row["updated_at"]),
                )
            )
        return votes

    def stats(self) -> FeedbackStats:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    SUM(CASE WHEN vote > 0 THEN 1 ELSE 0 END) AS positive_votes,
                    SUM(CASE WHEN vote < 0 THEN 1 ELSE 0 END) AS negative_votes,
                    COUNT(*) AS total_votes
                FROM feedback_votes
                """
            ).fetchone()

        return FeedbackStats(
            positive_votes=int(row["positive_votes"] or 0),
            negative_votes=int(row["negative_votes"] or 0),
            total_votes=int(row["total_votes"] or 0),
        )

    def recent_votes(self, limit: int = 20) -> list[FeedbackVote]:
        return self.list_votes()[: max(1, limit)]

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn


class FeedbackPreferenceModel:
    def __init__(
        self,
        votes: Sequence[FeedbackVote],
        *,
        max_adjustment: float,
        min_feature_votes: int,
    ) -> None:
        self._votes = list(votes)
        self._max_adjustment = max(0.0, float(max_adjustment))
        self._min_feature_votes = max(1, int(min_feature_votes))
        self._feature_stats: dict[str, tuple[int, int]] = {}
        self._paper_votes: dict[str, int] = {}

        for vote in self._votes:
            self._paper_votes[vote.paper_id] = vote.vote
            for feature in _vote_features(vote):
                pos, neg = self._feature_stats.get(feature, (0, 0))
                if vote.vote > 0:
                    pos += 1
                elif vote.vote < 0:
                    neg += 1
                self._feature_stats[feature] = (pos, neg)

    @property
    def total_votes(self) -> int:
        return len(self._votes)

    def adjust_score(self, paper: Paper, scored: ScoredPaper) -> ScoredPaper:
        if self._max_adjustment <= 0.0 or not self._votes:
            ranking_score = scored.ranking_score if scored.ranking_score is not None else scored.relevance_score
            if ranking_score == scored.ranking_score:
                return scored
            return replace(scored, ranking_score=ranking_score)

        raw_adjustment = 0.0
        paper_vote = self._paper_votes.get(paper.arxiv_id)
        if paper_vote is not None:
            raw_adjustment += paper_vote * self._max_adjustment

        feature_adjustments: list[float] = []
        for feature in _paper_features(paper, scored):
            stats = self._feature_stats.get(feature)
            if not stats:
                continue
            pos, neg = stats
            count = pos + neg
            if count < self._min_feature_votes:
                continue
            feature_adjustments.append((pos - neg) / (count + 2))

        if feature_adjustments:
            raw_adjustment += (sum(feature_adjustments) / len(feature_adjustments)) * (
                self._max_adjustment * 0.7
            )

        adjustment = max(-self._max_adjustment, min(self._max_adjustment, raw_adjustment))
        ranking_score = max(1.0, min(10.0, scored.relevance_score + adjustment))

        if scored.ranking_score is not None and abs(scored.ranking_score - ranking_score) < 1e-9:
            return scored
        return replace(scored, ranking_score=ranking_score)


def build_feedback_token(
    signer: FeedbackTokenSigner,
    *,
    paper: Paper,
    scored: ScoredPaper,
    digest_date: str,
) -> str:
    token_id = f"{digest_date}:{paper.arxiv_id}"
    payload = {
        "v": _TOKEN_VERSION,
        "tid": token_id,
        "pid": paper.arxiv_id,
        "t": paper.title,
        "u": paper.url,
        "s": paper.source_label,
        "c": list(paper.categories),
        "a": list(paper.authors),
        "b": float(scored.relevance_score),
        "n": scored.novelty_signal,
        "d": digest_date,
    }
    return signer.dumps(payload)


def build_feedback_links(
    *,
    base_url: str,
    signer: FeedbackTokenSigner,
    paper: Paper,
    scored: ScoredPaper,
    digest_date: str,
) -> dict[str, str]:
    token = build_feedback_token(
        signer,
        paper=paper,
        scored=scored,
        digest_date=digest_date,
    )
    normalized_base = base_url.rstrip("/")
    encoded = quote(token, safe="")
    return {
        "upvote_url": f"{normalized_base}/feedback/up?token={encoded}",
        "downvote_url": f"{normalized_base}/feedback/down?token={encoded}",
    }


def vote_from_token_payload(payload: dict[str, Any], vote: int) -> FeedbackVote:
    now = datetime.now(timezone.utc).isoformat()
    return FeedbackVote(
        token_id=str(payload.get("tid") or ""),
        paper_id=str(payload.get("pid") or ""),
        title=str(payload.get("t") or ""),
        url=str(payload.get("u") or ""),
        source_label=str(payload.get("s") or ""),
        categories=[str(item) for item in payload.get("c", []) if str(item).strip()],
        authors=[str(item) for item in payload.get("a", []) if str(item).strip()],
        base_score=float(payload.get("b") or 0.0),
        novelty_signal=str(payload.get("n") or "incremental"),
        digest_date=str(payload.get("d") or ""),
        vote=1 if vote > 0 else -1,
        updated_at=now,
    )


def load_preference_model(
    store: FeedbackStore,
    config: FeedbackConfig,
) -> FeedbackPreferenceModel:
    return FeedbackPreferenceModel(
        store.list_votes(),
        max_adjustment=config.max_adjustment,
        min_feature_votes=config.min_feature_votes,
    )


def run_feedback_server(
    *,
    store: FeedbackStore,
    signer: FeedbackTokenSigner,
    host: str,
    port: int,
    logger: logging.Logger | None = None,
) -> None:
    server_logger = logger or LOGGER

    class FeedbackHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlsplit(self.path)
            if parsed.path == "/healthz":
                self._send_html(HTTPStatus.OK, "<p>ok</p>")
                return
            if parsed.path in {"/", ""}:
                self._send_index()
                return
            if parsed.path not in {"/feedback/up", "/feedback/down"}:
                self._send_html(HTTPStatus.NOT_FOUND, "<p>Not found.</p>")
                return

            vote_value = 1 if parsed.path.endswith("/up") else -1
            params = parse_qs(parsed.query)
            token = params.get("token", [""])[0]
            if not token:
                self._send_html(HTTPStatus.BAD_REQUEST, "<p>Missing feedback token.</p>")
                return

            try:
                payload = signer.loads(token)
                vote = vote_from_token_payload(payload, vote_value)
                if not vote.token_id or not vote.paper_id or not vote.title:
                    raise FeedbackError("Feedback token payload is incomplete.")
                store.record_vote(vote)
            except FeedbackError as exc:
                self._send_html(
                    HTTPStatus.BAD_REQUEST,
                    f"<p>Could not record feedback: {html.escape(str(exc))}</p>",
                )
                return
            except Exception as exc:  # pragma: no cover - defensive server path
                server_logger.exception("Unexpected feedback error: %s", exc)
                self._send_html(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "<p>Unexpected error while recording feedback.</p>",
                )
                return

            verdict = "Interesting" if vote.vote > 0 else "Not interesting"
            body = (
                f"<p><strong>{html.escape(verdict)}</strong> recorded for:</p>"
                f"<p>{html.escape(vote.title)}</p>"
                "<p>You can close this tab.</p>"
            )
            self._send_html(HTTPStatus.OK, body)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            server_logger.info("feedback-server | " + format, *args)

        def _send_index(self) -> None:
            stats = store.stats()
            rows = []
            for vote in store.recent_votes(limit=20):
                direction = "interesting" if vote.vote > 0 else "not interesting"
                rows.append(
                    "<li>"
                    f"{html.escape(vote.updated_at)} · "
                    f"{html.escape(direction)} · "
                    f"{html.escape(vote.title)}"
                    "</li>"
                )
            body = [
                "<h1>Scout Feedback</h1>",
                f"<p>Total votes: {stats.total_votes} "
                f"(+{stats.positive_votes} / -{stats.negative_votes})</p>",
                "<p><a href=\"/healthz\">healthz</a></p>",
                "<h2>Recent Votes</h2>",
                "<ul>",
                *rows,
                "</ul>",
            ]
            self._send_html(HTTPStatus.OK, "".join(body))

        def _send_html(self, status: HTTPStatus, body: str) -> None:
            payload = (
                "<!doctype html><html><head><meta charset=\"utf-8\">"
                "<title>Scout Feedback</title></head><body>"
                f"{body}"
                "</body></html>"
            ).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    with ThreadingHTTPServer((host, port), FeedbackHandler) as server:
        server_logger.info("Scout feedback server listening on http://%s:%d", host, port)
        server.serve_forever()


def _paper_features(paper: Paper, scored: ScoredPaper) -> list[str]:
    features: list[str] = []
    if paper.source_label:
        features.append(f"source:{paper.source_label.strip().lower()}")
    for category in paper.categories:
        normalized = category.strip().lower()
        if normalized:
            features.append(f"category:{normalized}")
    features.append(f"novelty:{scored.novelty_signal}")
    features.extend(_paper_kind_features(paper.title))
    features.extend(_keyword_features(paper.title))
    return _dedupe(features)


def _vote_features(vote: FeedbackVote) -> list[str]:
    features: list[str] = []
    if vote.source_label:
        features.append(f"source:{vote.source_label.strip().lower()}")
    for category in vote.categories:
        normalized = category.strip().lower()
        if normalized:
            features.append(f"category:{normalized}")
    if vote.novelty_signal:
        features.append(f"novelty:{vote.novelty_signal.strip().lower()}")
    features.extend(_paper_kind_features(vote.title))
    features.extend(_keyword_features(vote.title))
    return _dedupe(features)


def _paper_kind_features(title: str) -> list[str]:
    lowered = title.lower()
    kinds: list[str] = []
    if "benchmark" in lowered or "bench" in lowered:
        kinds.append("kind:benchmark")
    if any(term in lowered for term in ("survey", "review", "taxonomy", "overview")):
        kinds.append("kind:survey")
    if any(term in lowered for term in ("framework", "architecture", "protocol")):
        kinds.append("kind:framework")
    if any(term in lowered for term in ("eval", "evaluation", "audit", "monitoring")):
        kinds.append("kind:evaluation")
    if any(term in lowered for term in ("jailbreak", "attack", "backdoor", "exploit")):
        kinds.append("kind:attack")
    if any(term in lowered for term in ("defend", "defense", "guard", "safe", "safety")):
        kinds.append("kind:defense")
    if any(term in lowered for term in ("interpretability", "mechanistic", "probe", "steering")):
        kinds.append("kind:interpretability")
    return kinds


def _keyword_features(text: str) -> list[str]:
    features: list[str] = []
    for match in _WORD_RE.findall(text.lower()):
        if match in _STOPWORDS:
            continue
        features.append(f"kw:{match}")
        if len(features) >= 6:
            break
    return features


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _urlsafe_b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _urlsafe_b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii"))


def _safe_json_list(value: Any) -> list[str]:
    try:
        parsed = json.loads(value)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if str(item).strip()]
