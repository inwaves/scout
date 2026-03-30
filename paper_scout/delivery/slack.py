from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from .base import DeliveryError

LOGGER = logging.getLogger(__name__)


class SlackWebhookDelivery:
    channel_type = "slack"

    def __init__(
        self,
        webhook_url: str,
        timeout_seconds: int = 20,
        logger: logging.Logger | None = None,
    ) -> None:
        self.webhook_url = webhook_url
        self.timeout_seconds = timeout_seconds
        self._logger = logger or LOGGER

    def deliver(self, *, subject: str, markdown_body: str, html_body: str) -> None:
        del html_body  # Slack incoming webhooks accept text payload.
        text = f"*{subject}*\n\n{markdown_body}"

        payload = {"text": text}
        self._post_json(payload)

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
                    raise DeliveryError(f"Slack webhook returned HTTP {status}: {body}")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise DeliveryError(f"Slack webhook HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise DeliveryError(f"Slack webhook connection failed: {exc}") from exc

        self._logger.info("Slack digest delivered successfully.")