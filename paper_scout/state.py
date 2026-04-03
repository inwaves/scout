from __future__ import annotations

import json
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


def save_last_run(
    state_path: str | Path,
    timestamp: datetime,
    cumulative_cost_usd: float = 0.0,
) -> None:
    path = Path(state_path).expanduser()
    timestamp_utc = timestamp.astimezone(timezone.utc) if timestamp.tzinfo else timestamp.replace(
        tzinfo=timezone.utc
    )
    try:
        cumulative_cost = max(0.0, float(cumulative_cost_usd))
    except (TypeError, ValueError):
        cumulative_cost = 0.0

    payload: dict[str, Any] = {
        "last_successful_run": timestamp_utc.isoformat(),
        "cumulative_cost_usd": cumulative_cost,
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

    if last_run is not None:
        if last_run.tzinfo is None:
            return min(last_run.replace(tzinfo=timezone.utc), now_utc)
        return min(last_run.astimezone(timezone.utc), now_utc)

    return now_utc - timedelta(hours=lookback_hours)


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