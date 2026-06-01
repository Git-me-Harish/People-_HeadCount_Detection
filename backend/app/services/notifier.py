"""Notifier service — multi-channel alert dispatch with retry.

Channels supported: inbox (in-app), email, Slack, MS Teams, webhook.
Each delivery attempt is persisted as a Notification row.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import httpx
from sqlalchemy.orm import Session

from ..config import get_settings
from ..models import Notification, NotificationChannel, NotificationStatus

logger = logging.getLogger(__name__)

_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF_S = [1, 3, 5]  # seconds between retries (sync; use background task for async)


def _http_post(url: str, payload: dict, headers: dict | None = None) -> None:
    """Synchronous HTTP POST with linear retry."""
    last_exc: Exception | None = None
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            with httpx.Client(timeout=8.0) as client:
                resp = client.post(url, json=payload, headers=headers or {})
                resp.raise_for_status()
                return
        except Exception as exc:
            last_exc = exc
            logger.warning("Dispatch attempt %d/%d failed: %s", attempt + 1, _RETRY_ATTEMPTS, exc)
    raise last_exc  # type: ignore[misc]


def dispatch(
    db: Session,
    *,
    organization_id: int,
    user_id: int | None,
    title: str,
    body: str,
    channels: list[NotificationChannel],
    source_type: str | None = None,
    source_id: int | None = None,
    # Channel-specific targets
    email_to: str | None = None,
    slack_webhook: str | None = None,
    teams_webhook: str | None = None,
    extra_webhook_url: str | None = None,
) -> list[Notification]:
    """Create Notification rows and attempt delivery for each channel."""
    settings = get_settings()
    created: list[Notification] = []

    for channel in channels:
        notif = Notification(
            organization_id=organization_id,
            user_id=user_id,
            channel=channel,
            status=NotificationStatus.pending,
            title=title,
            body=body,
            source_type=source_type,
            source_id=source_id,
        )
        db.add(notif)
        db.flush()  # get ID before delivery attempt

        error: str | None = None

        try:
            if channel == NotificationChannel.inbox:
                # Inbox delivery = just persisting the row; no external call needed
                pass

            elif channel == NotificationChannel.email:
                # NOTE: Real email requires SMTP/SES config. Logs for now; wire up in Tier 2.
                logger.info(
                    "EMAIL → %s | %s | %s",
                    email_to or "no-recipient",
                    title,
                    body[:120],
                )

            elif channel == NotificationChannel.slack and slack_webhook:
                _http_post(
                    slack_webhook,
                    {"text": f"*{title}*\n{body}"},
                )

            elif channel == NotificationChannel.teams and teams_webhook:
                _http_post(
                    teams_webhook,
                    {
                        "@type": "MessageCard",
                        "@context": "http://schema.org/extensions",
                        "summary": title,
                        "themeColor": "6366f1",
                        "sections": [{"activityTitle": title, "activityText": body}],
                    },
                )

            elif channel == NotificationChannel.webhook and extra_webhook_url:
                _http_post(
                    extra_webhook_url,
                    {
                        "title": title,
                        "body": body,
                        "source_type": source_type,
                        "source_id": source_id,
                        "sent_at": datetime.now(timezone.utc).isoformat(),
                    },
                )

            notif.status = NotificationStatus.sent

        except Exception as exc:
            error = str(exc)
            notif.status = NotificationStatus.failed
            notif.error_detail = error
            logger.error("Notification delivery failed [channel=%s]: %s", channel, error)

        created.append(notif)

    db.commit()
    return created
