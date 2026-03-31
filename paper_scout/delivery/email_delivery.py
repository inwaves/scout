from __future__ import annotations

import logging
import smtplib
import ssl
from email.message import EmailMessage

from .base import DeliveryError

LOGGER = logging.getLogger(__name__)


class EmailDelivery:
    channel_type = "email"

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        from_address: str,
        to_address: str,
        smtp_username: str | None = None,
        smtp_password: str | None = None,
        smtp_starttls: bool = True,
        logger: logging.Logger | None = None,
    ) -> None:
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.from_address = from_address
        self.to_address = to_address
        self.smtp_username = smtp_username
        self.smtp_password = smtp_password
        self.smtp_starttls = smtp_starttls
        self._logger = logger or LOGGER

    def deliver(self, *, subject: str, markdown_body: str, html_body: str) -> None:
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = self.from_address
        message["To"] = self.to_address
        message.set_content(markdown_body)
        message.add_alternative(html_body, subtype="html")

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=30) as smtp:
                smtp.ehlo()
                if self.smtp_starttls:
                    smtp.starttls(context=ssl.create_default_context())
                    smtp.ehlo()

                if self.smtp_username:
                    if self.smtp_password is None:
                        raise DeliveryError(
                            "SMTP username configured but SMTP password is missing."
                        )
                    smtp.login(self.smtp_username, self.smtp_password)

                smtp.send_message(message)
        except DeliveryError:
            raise
        except Exception as exc:
            raise DeliveryError(f"Failed to send digest email via SMTP: {exc}") from exc

        self._logger.info("Email digest delivered to %s", self.to_address)