from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from paper_scout.config import (
    ConfigError,
    PaperScoutConfig,
    load_config,
    substitute_env_vars,
)


@pytest.fixture()
def minimal_config_file(tmp_path: Path) -> Path:
    content = textwrap.dedent("""\
        profile:
          name: "Test Profile"
          description: "A test research profile."
          scoring_rubric: "Score 1-10."
        arxiv:
          categories:
            - "cs.AI"
        scoring:
          model: "claude-sonnet-4-6"
          threshold: 7
        delivery:
          channels:
            - type: "markdown"
              output_dir: "./test-digests"
    """)
    path = tmp_path / "scout.yml"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture()
def full_config_file(tmp_path: Path) -> Path:
    content = textwrap.dedent("""\
        anthropic_api_key: "sk-test-key"
        state_file: "./test-state.json"

        profile:
          name: "AI Agents Research"
          description: "Agent architectures, multi-agent systems."
          scoring_rubric: |
            Score 1-10:
            - 9-10: Directly relevant.
            - 1-2: Not relevant.

        arxiv:
          categories:
            - "cs.AI"
            - "cs.MA"
          max_results_per_category: 100
          lookback_hours: 24
          request_timeout_seconds: 15.0
          max_retries: 2
          retry_backoff_seconds: 1.0
          query_pause_seconds: 0.3

        scoring:
          model: "claude-sonnet-4-6"
          batch_size: 5
          threshold: 8.0
          max_papers: 10
          use_batch_api: false
          temperature: 0.0
          max_output_tokens: 2048
          batch_poll_interval_seconds: 3
          batch_poll_timeout_seconds: 600
          summary_words: 100

        delivery:
          channels:
            - type: "markdown"
              output_dir: "./digests"
              filename_template: "digest-{date}.md"

        schedule:
          cron: "0 8 * * *"
    """)
    path = tmp_path / "scout.yml"
    path.write_text(content, encoding="utf-8")
    return path


class TestLoadConfig:
    def test_load_minimal_config(self, minimal_config_file: Path) -> None:
        config = load_config(minimal_config_file)
        assert isinstance(config, PaperScoutConfig)
        assert config.profile.name == "Test Profile"
        assert config.arxiv.categories == ["cs.AI"]
        assert config.scoring.threshold == 7.0
        assert len(config.delivery_channels) == 1
        assert config.delivery_channels[0].type == "markdown"

    def test_load_full_config(self, full_config_file: Path) -> None:
        config = load_config(full_config_file)
        assert config.anthropic_api_key == "sk-test-key"
        assert config.profile.name == "AI Agents Research"
        assert config.arxiv.categories == ["cs.AI", "cs.MA"]
        assert config.arxiv.max_results_per_category == 100
        assert config.arxiv.lookback_hours == 24
        assert config.arxiv.request_timeout_seconds == 15.0
        assert config.scoring.batch_size == 5
        assert config.scoring.threshold == 8.0
        assert config.scoring.use_batch_api is False
        assert config.scoring.summary_words == 100
        assert config.schedule.cron == "0 8 * * *"

    def test_missing_config_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="Config file not found"):
            load_config(tmp_path / "nonexistent.yml")

    def test_missing_profile_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yml"
        path.write_text("arxiv:\n  categories: ['cs.AI']\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="profile"):
            load_config(path)

    def test_missing_profile_name_raises(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            profile:
              description: "test"
              scoring_rubric: "test"
        """)
        path = tmp_path / "bad.yml"
        path.write_text(content, encoding="utf-8")
        with pytest.raises(ConfigError, match="profile.name"):
            load_config(path)

    def test_defaults_applied(self, minimal_config_file: Path) -> None:
        config = load_config(minimal_config_file)
        assert config.arxiv.max_results_per_category == 200
        assert config.arxiv.lookback_hours == 28
        assert config.scoring.batch_size == 10
        assert config.scoring.max_papers == 15
        assert config.scoring.use_batch_api is True
        assert config.state_file == "last_run.json"


class TestEnvVarSubstitution:
    def test_substitute_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_VAR", "hello")
        result = substitute_env_vars("prefix-${TEST_VAR}-suffix")
        assert result == "prefix-hello-suffix"

    def test_substitute_nested(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_KEY", "secret123")
        data = {"outer": {"inner": "${MY_KEY}"}, "list": ["${MY_KEY}", "static"]}
        result = substitute_env_vars(data)
        assert result == {"outer": {"inner": "secret123"}, "list": ["secret123", "static"]}

    def test_missing_env_var_raises(self) -> None:
        env_var = "DEFINITELY_NOT_SET_PAPER_SCOUT_TEST"
        if env_var in os.environ:
            del os.environ[env_var]
        with pytest.raises(ConfigError, match=env_var):
            substitute_env_vars(f"${{{env_var}}}")

    def test_non_string_passthrough(self) -> None:
        assert substitute_env_vars(42) == 42
        assert substitute_env_vars(True) is True
        assert substitute_env_vars(None) is None

    def test_env_var_in_config_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_API_KEY", "sk-from-env")
        content = textwrap.dedent("""\
            anthropic_api_key: "${TEST_API_KEY}"
            profile:
              name: "Test"
              description: "Test"
              scoring_rubric: "Test"
        """)
        path = tmp_path / "scout.yml"
        path.write_text(content, encoding="utf-8")
        config = load_config(path)
        assert config.anthropic_api_key == "sk-from-env"


class TestBoolRejectionInIntFields:
    def test_bool_in_int_field_raises(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            profile:
              name: "Test"
              description: "Test"
              scoring_rubric: "Test"
            scoring:
              batch_size: true
        """)
        path = tmp_path / "bad.yml"
        path.write_text(content, encoding="utf-8")
        with pytest.raises(ConfigError, match="must be an integer, not a boolean"):
            load_config(path)

    def test_bool_in_float_field_raises(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            profile:
              name: "Test"
              description: "Test"
              scoring_rubric: "Test"
            scoring:
              threshold: false
        """)
        path = tmp_path / "bad.yml"
        path.write_text(content, encoding="utf-8")
        with pytest.raises(ConfigError, match="must be a number, not a boolean"):
            load_config(path)


class TestDeliveryChannelValidation:
    def test_email_missing_fields_raises(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            profile:
              name: "Test"
              description: "Test"
              scoring_rubric: "Test"
            delivery:
              channels:
                - type: "email"
        """)
        path = tmp_path / "bad.yml"
        path.write_text(content, encoding="utf-8")
        with pytest.raises(ConfigError, match="smtp_host"):
            load_config(path)

    def test_slack_missing_webhook_raises(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            profile:
              name: "Test"
              description: "Test"
              scoring_rubric: "Test"
            delivery:
              channels:
                - type: "slack"
        """)
        path = tmp_path / "bad.yml"
        path.write_text(content, encoding="utf-8")
        with pytest.raises(ConfigError, match="webhook_url"):
            load_config(path)

    def test_unknown_channel_type_raises(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            profile:
              name: "Test"
              description: "Test"
              scoring_rubric: "Test"
            delivery:
              channels:
                - type: "telegram"
        """)
        path = tmp_path / "bad.yml"
        path.write_text(content, encoding="utf-8")
        with pytest.raises(ConfigError, match="must be one of"):
            load_config(path)