from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


class StateError(RuntimeError):
    """Raised when state file cannot be read or written."""


def load_last_run(state_path: str | Path) -> datetime | None:
    path = Path(state_path).expanduser()
    payload = _load_state_payload(path)
    if payload is None:
        return None

    raw = payload.get("last_successful_run")
    if not isinstance(raw, str) or not raw.strip():
        return None

    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise StateError(f"Invalid timestamp in state file {path}: {raw}") from exc

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def load_cumulative_cost(state_path: str | Path) -> float:
    path = Path(state_path).expanduser()
    payload = _load_state_payload(path)
    if payload is None:
        return 0.0

    raw = payload.get("cumulative_cost_usd", 0.0)
    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, parsed)


def load_seen_web_posts(state_path: str | Path) -> dict[str, dict[str, str]]:
    path = Path(state_path).expanduser()
    payload = _load_state_payload(path)
    if payload is None:
        return {}

    raw = payload.get("seen_web_posts", {})
    if isinstance(raw, list):
        return {
            item.strip(): {}
            for item in raw
            if isinstance(item, str) and item.strip()
        }
    if not isinstance(raw, dict):
        return {}

    seen: dict[str, dict[str, str]] = {}
    for raw_id, raw_record in raw.items():
        post_id = str(raw_id).strip()
        if not post_id:
            continue
        if isinstance(raw_record, dict):
            seen[post_id] = {
                str(key): str(value)
                for key, value in raw_record.items()
                if value is not None
            }
        else:
            seen[post_id] = {}
    return seen


def save_last_run(
    state_path: str | Path,
    timestamp: datetime,
    cumulative_cost_usd: float = 0.0,
    seen_web_posts: Mapping[str, Mapping[str, str]] | None = None,
) -> None:
    path = Path(state_path).expanduser()
    timestamp_utc = timestamp.astimezone(timezone.utc) if timestamp.tzinfo else timestamp.replace(
        tzinfo=timezone.utc
    )
    try:
        cumulative_cost = max(0.0, float(cumulative_cost_usd))
    except (TypeError, ValueError):
        cumulative_cost = 0.0

    try:
        payload: dict[str, Any] = _load_state_payload(path) or {}
    except StateError:
        payload = {}

    payload["last_successful_run"] = timestamp_utc.isoformat()
    payload["cumulative_cost_usd"] = cumulative_cost
    if seen_web_posts is not None:
        payload["seen_web_posts"] = {
            str(post_id): {
                str(key): str(value)
                for key, value in record.items()
                if value is not None
            }
            for post_id, record in seen_web_posts.items()
            if str(post_id).strip()
        }

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        raise StateError(f"Failed to write state file {path}: {exc}") from exc


def determine_since(
    last_run: datetime | None,
    lookback_hours: int,
    now: datetime | None = None,
) -> datetime:
    if now is None:
        now_utc = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now_utc = now.replace(tzinfo=timezone.utc)
    else:
        now_utc = now.astimezone(timezone.utc)

    minimum_since = now_utc - timedelta(hours=lookback_hours)

    if last_run is not None:
        if last_run.tzinfo is None:
            last_run_utc = last_run.replace(tzinfo=timezone.utc)
        else:
            last_run_utc = last_run.astimezone(timezone.utc)
        return min(last_run_utc, minimum_since, now_utc)

    return minimum_since


def _load_state_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StateError(f"Failed to read state file {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise StateError(f"Invalid state payload in {path}: expected JSON object.")
    return payload
