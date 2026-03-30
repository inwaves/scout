from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from .base import DeliveryError

LOGGER = logging.getLogger(__name__)


class MarkdownFileDelivery:
    channel_type = "markdown"

    def __init__(
        self,
        output_dir: str,
        filename_template: str = "scout-{date}.md",
        logger: logging.Logger | None = None,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser()
        self.filename_template = filename_template
        self._logger = logger or LOGGER

    def deliver(self, *, subject: str, markdown_body: str, html_body: str) -> None:
        del subject, html_body  # unused for file channel

        now = datetime.now(timezone.utc)
        filename = self.filename_template.format(
            date=now.strftime("%Y-%m-%d"),
            timestamp=now.strftime("%Y%m%d-%H%M%S"),
        )
        path = self.output_dir / filename

        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(markdown_body, encoding="utf-8")
        except OSError as exc:
            raise DeliveryError(f"Failed to write markdown digest to {path}: {exc}") from exc

        self._logger.info("Wrote markdown digest to %s", path)