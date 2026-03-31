from __future__ import annotations

from typing import Protocol


class DeliveryError(RuntimeError):
    """Raised when a delivery channel fails."""


class DeliveryChannel(Protocol):
    channel_type: str

    def deliver(self, *, subject: str, markdown_body: str, html_body: str) -> None:
        """Deliver a rendered digest."""