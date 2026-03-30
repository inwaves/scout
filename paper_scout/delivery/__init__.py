from __future__ import annotations

import logging
from typing import Iterable

from ..config import DeliveryChannelConfig
from .base import DeliveryChannel, DeliveryError
from .discord import DiscordWebhookDelivery
from .email_delivery import EmailDelivery
from .file import MarkdownFileDelivery
from .slack import SlackWebhookDelivery

LOGGER = logging.getLogger(__name__)


def build_delivery_channels(
    channel_configs: Iterable[DeliveryChannelConfig],
    logger: logging.Logger | None = None,
) -> list[DeliveryChannel]:
    channels: list[DeliveryChannel] = []
    active_logger = logger or LOGGER

    for config in channel_configs:
        if not config.enabled:
            continue

        channel_type = config.type.lower()

        if channel_type == "markdown":
            channels.append(
                MarkdownFileDelivery(
                    output_dir=config.output_dir or "./digests",
                    filename_template=config.filename_template,
                    logger=active_logger,
                )
            )
        elif channel_type == "email":
            if not config.smtp_host or not config.from_address or not config.to_address:
                raise DeliveryError(
                    "Email channel configuration is incomplete (smtp_host/from/to required)."
                )
            channels.append(
                EmailDelivery(
                    smtp_host=config.smtp_host,
                    smtp_port=config.smtp_port,
                    from_address=config.from_address,
                    to_address=config.to_address,
                    smtp_username=config.smtp_username,
                    smtp_password=config.smtp_password,
                    smtp_starttls=config.smtp_starttls,
                    logger=active_logger,
                )
            )
        elif channel_type == "slack":
            if not config.webhook_url:
                raise DeliveryError("Slack channel requires webhook_url.")
            channels.append(
                SlackWebhookDelivery(
                    webhook_url=config.webhook_url,
                    logger=active_logger,
                )
            )
        elif channel_type == "discord":
            if not config.webhook_url:
                raise DeliveryError("Discord channel requires webhook_url.")
            channels.append(
                DiscordWebhookDelivery(
                    webhook_url=config.webhook_url,
                    logger=active_logger,
                )
            )
        else:
            raise DeliveryError(f"Unsupported delivery channel type: {config.type}")

    return channels


__all__ = [
    "DeliveryChannel",
    "DeliveryError",
    "build_delivery_channels",
    "MarkdownFileDelivery",
    "EmailDelivery",
    "SlackWebhookDelivery",
    "DiscordWebhookDelivery",
]