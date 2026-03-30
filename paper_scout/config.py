from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_FILE = "scout.yml"
DEFAULT_SUBJECT_TEMPLATE = "Scout Digest — {date} ({count} papers)"
DEFAULT_ARXIV_CATEGORIES = ["cs.AI", "cs.MA", "cs.CL", "cs.LG", "cs.SE"]
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ConfigError(ValueError):
    """Raised when configuration is missing or invalid."""


@dataclass(slots=True)
class ProfileConfig:
    name: str
    description: str
    scoring_rubric: str


@dataclass(slots=True)
class ArxivConfig:
    categories: list[str] = field(default_factory=lambda: DEFAULT_ARXIV_CATEGORIES.copy())
    max_results_per_category: int = 200
    lookback_hours: int = 28
    request_timeout_seconds: float = 20.0
    max_retries: int = 3
    retry_backoff_seconds: float = 1.5
    query_pause_seconds: float = 0.5


@dataclass(slots=True)
class ScoringConfig:
    model: str = "claude-sonnet-4-6"
    batch_size: int = 10
    threshold: float = 7.0
    max_papers: int = 15
    use_batch_api: bool = True
    temperature: float = 0.0
    max_output_tokens: int = 4096
    batch_poll_interval_seconds: int = 5
    batch_poll_timeout_seconds: int = 1800
    summary_words: int = 150


@dataclass(slots=True)
class DeliveryChannelConfig:
    type: str
    enabled: bool = True
    output_dir: str | None = None
    filename_template: str = "scout-{date}.md"
    webhook_url: str | None = None
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_username: str | None = None
    smtp_password: str | None = None
    smtp_starttls: bool = True
    from_address: str | None = None
    to_address: str | None = None
    subject_template: str = DEFAULT_SUBJECT_TEMPLATE


@dataclass(slots=True)
class ScheduleConfig:
    cron: str = "0 7 * * 1-5"


@dataclass(slots=True)
class PaperScoutConfig:
    profile: ProfileConfig
    arxiv: ArxivConfig
    scoring: ScoringConfig
    delivery_channels: list[DeliveryChannelConfig] = field(default_factory=list)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    anthropic_api_key: str | None = None
    state_file: str = "last_run.json"
    config_path: Path = Path(DEFAULT_CONFIG_FILE)


def load_config(config_path: str | Path | None = None) -> PaperScoutConfig:
    """
    Load, validate, and return application configuration.

    Resolution order:
      1) explicit config_path argument
      2) SCOUT_CONFIG environment variable
      3) ./scout.yml
    """
    path = _resolve_config_path(config_path)
    raw = _load_yaml(path)
    substituted = substitute_env_vars(raw)
    return _parse_config(substituted, path)


def describe_config(config: PaperScoutConfig) -> str:
    """Render a short human-readable summary of the active configuration."""
    channels = ", ".join(
        f"{channel.type}{'' if channel.enabled else ' (disabled)'}"
        for channel in config.delivery_channels
    ) or "none"

    return "\n".join(
        [
            f"Config path: {config.config_path}",
            f"Profile: {config.profile.name}",
            f"arXiv categories: {', '.join(config.arxiv.categories)}",
            f"arXiv max per category: {config.arxiv.max_results_per_category}",
            f"Lookback hours: {config.arxiv.lookback_hours}",
            f"Scoring model: {config.scoring.model}",
            f"LLM batch size: {config.scoring.batch_size}",
            f"Threshold: {config.scoring.threshold}",
            f"Max papers in digest: {config.scoring.max_papers}",
            f"Use Anthropic batch API: {config.scoring.use_batch_api}",
            f"Delivery channels: {channels}",
            f"State file: {config.state_file}",
            f"Schedule cron: {config.schedule.cron}",
            f"Anthropic API key configured: {'yes' if config.anthropic_api_key else 'no'}",
        ]
    )


def substitute_env_vars(value: Any) -> Any:
    """Recursively replace ${VAR_NAME} placeholders with environment values."""
    if isinstance(value, str):
        return _substitute_string(value)
    if isinstance(value, list):
        return [substitute_env_vars(item) for item in value]
    if isinstance(value, dict):
        return {key: substitute_env_vars(item) for key, item in value.items()}
    return value


def _resolve_config_path(config_path: str | Path | None) -> Path:
    candidate = config_path or os.environ.get("SCOUT_CONFIG") or DEFAULT_CONFIG_FILE
    path = Path(candidate).expanduser()
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    return path


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Unable to read config file {path}: {exc}") from exc

    try:
        loaded = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in config file {path}: {exc}") from exc

    if not isinstance(loaded, dict):
        raise ConfigError("Top-level configuration must be a YAML mapping/object.")
    return loaded


def _substitute_string(text: str) -> str:
    def replacer(match: re.Match[str]) -> str:
        var_name = match.group(1)
        if var_name not in os.environ:
            raise ConfigError(
                f"Environment variable '{var_name}' referenced in config is not set."
            )
        return os.environ[var_name]

    return _ENV_PATTERN.sub(replacer, text)


def _parse_config(raw: dict[str, Any], path: Path) -> PaperScoutConfig:
    profile = _parse_profile(raw.get("profile"))
    arxiv = _parse_arxiv(raw.get("arxiv", {}))
    scoring = _parse_scoring(raw.get("scoring", {}))
    delivery_channels = _parse_delivery(raw.get("delivery", {}))
    schedule = _parse_schedule(raw.get("schedule", {}))
    anthropic_api_key = _optional_str(raw.get("anthropic_api_key")) or os.environ.get(
        "ANTHROPIC_API_KEY"
    )
    state_file = _optional_str(raw.get("state_file")) or "last_run.json"

    return PaperScoutConfig(
        profile=profile,
        arxiv=arxiv,
        scoring=scoring,
        delivery_channels=delivery_channels,
        schedule=schedule,
        anthropic_api_key=anthropic_api_key,
        state_file=state_file,
        config_path=path,
    )


def _parse_profile(section: Any) -> ProfileConfig:
    section_dict = _require_dict(section, "profile")
    return ProfileConfig(
        name=_as_non_empty_str(section_dict.get("name"), "profile.name"),
        description=_as_non_empty_str(section_dict.get("description"), "profile.description"),
        scoring_rubric=_as_non_empty_str(
            section_dict.get("scoring_rubric"), "profile.scoring_rubric"
        ),
    )


def _parse_arxiv(section: Any) -> ArxivConfig:
    section_dict = _ensure_dict(section, "arxiv")
    defaults = ArxivConfig()

    categories_raw = section_dict.get("categories", defaults.categories)
    if not isinstance(categories_raw, list) or not categories_raw:
        raise ConfigError("arxiv.categories must be a non-empty list of category strings.")

    categories: list[str] = []
    for index, category in enumerate(categories_raw):
        categories.append(_as_non_empty_str(category, f"arxiv.categories[{index}]"))

    return ArxivConfig(
        categories=categories,
        max_results_per_category=_as_int(
            section_dict.get("max_results_per_category", defaults.max_results_per_category),
            "arxiv.max_results_per_category",
            min_value=1,
        ),
        lookback_hours=_as_int(
            section_dict.get("lookback_hours", defaults.lookback_hours),
            "arxiv.lookback_hours",
            min_value=1,
        ),
        request_timeout_seconds=_as_float(
            section_dict.get("request_timeout_seconds", defaults.request_timeout_seconds),
            "arxiv.request_timeout_seconds",
            min_value=1.0,
        ),
        max_retries=_as_int(
            section_dict.get("max_retries", defaults.max_retries),
            "arxiv.max_retries",
            min_value=1,
            max_value=10,
        ),
        retry_backoff_seconds=_as_float(
            section_dict.get("retry_backoff_seconds", defaults.retry_backoff_seconds),
            "arxiv.retry_backoff_seconds",
            min_value=0.1,
        ),
        query_pause_seconds=_as_float(
            section_dict.get("query_pause_seconds", defaults.query_pause_seconds),
            "arxiv.query_pause_seconds",
            min_value=0.0,
        ),
    )


def _parse_scoring(section: Any) -> ScoringConfig:
    section_dict = _ensure_dict(section, "scoring")
    defaults = ScoringConfig()

    return ScoringConfig(
        model=_optional_str(section_dict.get("model")) or defaults.model,
        batch_size=_as_int(
            section_dict.get("batch_size", defaults.batch_size), "scoring.batch_size", min_value=1
        ),
        threshold=_as_float(
            section_dict.get("threshold", defaults.threshold),
            "scoring.threshold",
            min_value=1.0,
            max_value=10.0,
        ),
        max_papers=_as_int(
            section_dict.get("max_papers", defaults.max_papers), "scoring.max_papers", min_value=1
        ),
        use_batch_api=_as_bool(
            section_dict.get("use_batch_api", defaults.use_batch_api), "scoring.use_batch_api"
        ),
        temperature=_as_float(
            section_dict.get("temperature", defaults.temperature),
            "scoring.temperature",
            min_value=0.0,
            max_value=1.0,
        ),
        max_output_tokens=_as_int(
            section_dict.get("max_output_tokens", defaults.max_output_tokens),
            "scoring.max_output_tokens",
            min_value=256,
        ),
        batch_poll_interval_seconds=_as_int(
            section_dict.get(
                "batch_poll_interval_seconds", defaults.batch_poll_interval_seconds
            ),
            "scoring.batch_poll_interval_seconds",
            min_value=1,
        ),
        batch_poll_timeout_seconds=_as_int(
            section_dict.get(
                "batch_poll_timeout_seconds", defaults.batch_poll_timeout_seconds
            ),
            "scoring.batch_poll_timeout_seconds",
            min_value=30,
        ),
        summary_words=_as_int(
            section_dict.get("summary_words", defaults.summary_words),
            "scoring.summary_words",
            min_value=50,
        ),
    )


def _parse_delivery(section: Any) -> list[DeliveryChannelConfig]:
    section_dict = _ensure_dict(section, "delivery")
    channels_raw = section_dict.get("channels")

    if channels_raw is None:
        return [DeliveryChannelConfig(type="markdown", output_dir="./digests")]
    if not isinstance(channels_raw, list):
        raise ConfigError("delivery.channels must be a list.")

    channels = [_parse_delivery_channel(item, index) for index, item in enumerate(channels_raw)]
    if not channels:
        channels.append(DeliveryChannelConfig(type="markdown", output_dir="./digests"))
    return channels


def _parse_delivery_channel(section: Any, index: int) -> DeliveryChannelConfig:
    section_dict = _require_dict(section, f"delivery.channels[{index}]")
    field_prefix = f"delivery.channels[{index}]"

    channel_type = _as_non_empty_str(section_dict.get("type"), f"{field_prefix}.type").lower()
    enabled = _as_bool(section_dict.get("enabled", True), f"{field_prefix}.enabled")

    from_address = _optional_str(section_dict.get("from_address")) or _optional_str(
        section_dict.get("from")
    )
    to_address = _optional_str(section_dict.get("to_address")) or _optional_str(section_dict.get("to"))

    config = DeliveryChannelConfig(
        type=channel_type,
        enabled=enabled,
        output_dir=_optional_str(section_dict.get("output_dir")),
        filename_template=_optional_str(section_dict.get("filename_template"))
        or "scout-{date}.md",
        webhook_url=_optional_str(section_dict.get("webhook_url")),
        smtp_host=_optional_str(section_dict.get("smtp_host")),
        smtp_port=_as_int(
            section_dict.get("smtp_port", 587),
            f"{field_prefix}.smtp_port",
            min_value=1,
            max_value=65535,
        ),
        smtp_username=_optional_str(section_dict.get("smtp_username")),
        smtp_password=_optional_str(section_dict.get("smtp_password")),
        smtp_starttls=_as_bool(
            section_dict.get("smtp_starttls", True), f"{field_prefix}.smtp_starttls"
        ),
        from_address=from_address,
        to_address=to_address,
        subject_template=_optional_str(section_dict.get("subject_template"))
        or DEFAULT_SUBJECT_TEMPLATE,
    )

    if channel_type == "email":
        missing: list[str] = []
        if not config.smtp_host:
            missing.append("smtp_host")
        if not config.from_address:
            missing.append("from")
        if not config.to_address:
            missing.append("to")
        if missing:
            raise ConfigError(
                f"{field_prefix} (email) missing required fields: {', '.join(missing)}"
            )
    elif channel_type in {"slack", "discord"}:
        if not config.webhook_url:
            raise ConfigError(f"{field_prefix}.{channel_type} requires webhook_url.")
    elif channel_type == "markdown":
        if not config.output_dir:
            config.output_dir = "./digests"
    else:
        raise ConfigError(
            f"{field_prefix}.type must be one of: email, slack, discord, markdown."
        )

    return config


def _parse_schedule(section: Any) -> ScheduleConfig:
    section_dict = _ensure_dict(section, "schedule")
    defaults = ScheduleConfig()
    cron = _optional_str(section_dict.get("cron")) or defaults.cron
    return ScheduleConfig(cron=cron)


def _require_dict(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{field_name} must be a mapping/object in YAML.")
    return value


def _ensure_dict(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{field_name} must be a mapping/object in YAML.")
    return value


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    return str(value)


def _as_non_empty_str(value: Any, field_name: str) -> str:
    parsed = _optional_str(value)
    if parsed is None:
        raise ConfigError(f"{field_name} is required and must be a non-empty string.")
    return parsed


def _as_int(
    value: Any,
    field_name: str,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    # Reject booleans explicitly since bool is a subclass of int in Python.
    # YAML true/false would otherwise silently map to 1/0 for integer fields.
    if isinstance(value, bool):
        raise ConfigError(f"{field_name} must be an integer, not a boolean.")

    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field_name} must be an integer.") from exc

    if min_value is not None and parsed < min_value:
        raise ConfigError(f"{field_name} must be >= {min_value}.")
    if max_value is not None and parsed > max_value:
        raise ConfigError(f"{field_name} must be <= {max_value}.")
    return parsed


def _as_float(
    value: Any,
    field_name: str,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{field_name} must be a number, not a boolean.")

    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field_name} must be a number.") from exc

    if min_value is not None and parsed < min_value:
        raise ConfigError(f"{field_name} must be >= {min_value}.")
    if max_value is not None and parsed > max_value:
        raise ConfigError(f"{field_name} must be <= {max_value}.")
    return parsed


def _as_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    raise ConfigError(f"{field_name} must be a boolean.")