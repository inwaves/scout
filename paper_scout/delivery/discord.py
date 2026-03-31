from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from .base import DeliveryError

LOGGER = logging.getLogger(__name__)


class DiscordWebhookDelivery:
    channel_type = "discord"

    def __init__(
        self,
        webhook_url: str,
        timeout_seconds: int = 20,
        max_message_chars: int = 2000,
        logger: logging.Logger | None = None,
    ) -> None:
        self.webhook_url = webhook_url
        self.timeout_seconds = timeout_seconds
        self.max_message_chars = max_message_chars
        self._logger = logger or LOGGER

    def deliver(self, *, subject: str, markdown_body: str, html_body: str) -> None:
        del html_body  # Discord webhook uses text content.
        full_message = f"**{subject}**\n\n{markdown_body}"

        for chunk in _chunk_text(full_message, self.max_message_chars):
            self._post_json({"content": chunk})

        self._logger.info("Discord digest delivered successfully.")

    def _post_json(self, payload: dict[str, str]) -> None:
        request = urllib.request.Request(
            self.webhook_url,
            method="POST",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                status = getattr(response, "status", 200)
                if status >= 400:
                    body = response.read().decode("utf-8", errors="replace")
                    raise DeliveryError(f"Discord webhook returned HTTP {status}: {body}")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise DeliveryError(f"Discord webhook HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise DeliveryError(f"Discord webhook connection failed: {exc}") from exc


def _chunk_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current = ""

    for line in text.splitlines(keepends=True):
        if len(line) > max_chars:
            if current:
                chunks.append(current.rstrip())
                current = ""
            for index in range(0, len(line), max_chars):
                chunks.append(line[index : index + max_chars].rstrip())
            continue

        if len(current) + len(line) > max_chars:
            chunks.append(current.rstrip())
            current = line
        else:
            current += line

    if current:
        chunks.append(current.rstrip())

    return [chunk for chunk in chunks if chunk]