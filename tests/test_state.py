from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from paper_scout.state import (
    StateError,
    determine_since,
    load_last_run,
    load_seen_web_posts,
    save_last_run,
)


class TestLoadLastRun:
    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        result = load_last_run(tmp_path / "nonexistent.json")
        assert result is None

    def test_valid_file(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        path.write_text(
            json.dumps({"last_successful_run": "2026-03-28T07:00:00+00:00"}),
            encoding="utf-8",
        )
        result = load_last_run(path)
        assert result is not None
        assert result.year == 2026
        assert result.month == 3
        assert result.day == 28
        assert result.tzinfo is not None

    def test_naive_timestamp_gets_utc(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        path.write_text(
            json.dumps({"last_successful_run": "2026-03-28T07:00:00"}),
            encoding="utf-8",
        )
        result = load_last_run(path)
        assert result is not None
        assert result.tzinfo == timezone.utc

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        path.write_text("not json", encoding="utf-8")
        with pytest.raises(StateError):
            load_last_run(path)

    def test_missing_timestamp_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        path.write_text(json.dumps({}), encoding="utf-8")
        result = load_last_run(path)
        assert result is None


class TestSaveLastRun:
    def test_creates_file(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        now = datetime.now(timezone.utc)
        save_last_run(path, now)
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "last_successful_run" in data

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "sub" / "dir" / "state.json"
        now = datetime.now(timezone.utc)
        save_last_run(path, now)
        assert path.exists()

    def test_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        original = datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc)
        save_last_run(path, original)
        loaded = load_last_run(path)
        assert loaded is not None
        assert abs((loaded - original).total_seconds()) < 1

    def test_preserves_seen_web_posts(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        path.write_text(
            json.dumps(
                {
                    "last_successful_run": "2026-03-27T12:00:00+00:00",
                    "seen_web_posts": {
                        "openai:test-post": {
                            "url": "https://openai.com/index/test-post",
                            "first_seen": "2026-03-27T12:00:00+00:00",
                        }
                    },
                    "custom": "kept",
                }
            ),
            encoding="utf-8",
        )

        save_last_run(
            path,
            datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc),
            cumulative_cost_usd=1.25,
        )

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["custom"] == "kept"
        assert payload["seen_web_posts"]["openai:test-post"]["url"] == (
            "https://openai.com/index/test-post"
        )


class TestLoadSeenWebPosts:
    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert load_seen_web_posts(tmp_path / "missing.json") == {}

    def test_loads_mapping(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        path.write_text(
            json.dumps(
                {
                    "seen_web_posts": {
                        "openai:test-post": {
                            "url": "https://openai.com/index/test-post",
                            "title": "Test Post",
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        seen = load_seen_web_posts(path)
        assert seen["openai:test-post"]["title"] == "Test Post"

    def test_loads_legacy_list(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        path.write_text(json.dumps({"seen_web_posts": ["openai:old-post"]}), encoding="utf-8")
        assert load_seen_web_posts(path) == {"openai:old-post": {}}


class TestDetermineSince:
    def test_minimum_lookback_enforced(self) -> None:
        """Even with a recent last_run, since is at least lookback_hours ago."""
        last = datetime(2026, 3, 27, 12, 0, 0, tzinfo=timezone.utc)
        now = datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc)
        since = determine_since(last, lookback_hours=28, now=now)
        expected_min = now - timedelta(hours=28)
        # since should be the earlier of last_run and now-28h
        assert since == expected_min  # now-28h = Mar 27 08:00, earlier than last_run Mar 27 12:00

    def test_last_run_used_when_earlier_than_lookback(self) -> None:
        """If last_run is older than lookback window, use last_run (wider window)."""
        last = datetime(2026, 3, 20, 12, 0, 0, tzinfo=timezone.utc)
        now = datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc)
        since = determine_since(last, lookback_hours=28, now=now)
        assert since == last  # last_run is 8 days ago, much wider than 28h

    def test_falls_back_to_lookback_hours(self) -> None:
        now = datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc)
        since = determine_since(None, lookback_hours=24, now=now)
        expected = now - timedelta(hours=24)
        assert abs((since - expected).total_seconds()) < 1

    def test_future_last_run_uses_minimum_lookback(self) -> None:
        """A future last_run should not prevent looking back at least lookback_hours."""
        future = datetime(2099, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        now = datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc)
        since = determine_since(future, lookback_hours=28, now=now)
        expected_min = now - timedelta(hours=28)
        assert since == expected_min
