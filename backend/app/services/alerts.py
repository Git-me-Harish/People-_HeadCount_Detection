"""Alert evaluation and webhook dispatch."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx
from sqlalchemy.orm import Session

from ..models import Alert, NotificationChannel
from ..services.notifier import dispatch as notifier_dispatch

logger = logging.getLogger(__name__)


def evaluate_alerts(
    db: Session,
    *,
    organization_id: int,
    camera_id: int | None,
    person_count: int,
) -> list[Alert]:
    """Find matching enabled alerts whose threshold is breached and fire them.

    Skips alerts still within their cooldown window to prevent webhook spam.
    Dispatches in-app inbox notification + any configured webhook per alert.
    """
    now = datetime.now(timezone.utc)

    query = db.query(Alert).filter(
        Alert.organization_id == organization_id,
        Alert.is_enabled.is_(True),
        Alert.threshold <= person_count,
    )
    if camera_id is None:
        query = query.filter(Alert.camera_id.is_(None))
    else:
        query = query.filter((Alert.camera_id == camera_id) | (Alert.camera_id.is_(None)))

    fired: list[Alert] = []
    for alert in query.all():
        # Cooldown guard — skip if fired within cooldown_minutes
        if alert.last_triggered_at is not None:
            elapsed_s = (now - alert.last_triggered_at).total_seconds()
            if elapsed_s < alert.cooldown_minutes * 60:
                logger.debug(
                    "Alert %s suppressed — cooldown %.0fs remaining",
                    alert.id,
                    alert.cooldown_minutes * 60 - elapsed_s,
                )
                continue

        alert.last_triggered_at = now
        fired.append(alert)

        # Raw webhook (backward-compat)
        _dispatch_webhook(alert, person_count=person_count, camera_id=camera_id)

        # In-app + channel notifications via notifier service
        title = f"Alert: {alert.name}"
        body = (
            f"Threshold {alert.threshold} exceeded — "
            f"current count is {person_count}"
            + (f" (camera {camera_id})" if camera_id else "")
        )
        channels: list[NotificationChannel] = [NotificationChannel.inbox]
        notifier_dispatch(
            db,
            organization_id=organization_id,
            user_id=None,
            title=title,
            body=body,
            channels=channels,
            source_type="alert",
            source_id=alert.id,
        )

    if fired:
        db.commit()
    return fired


def _dispatch_webhook(alert: Alert, *, person_count: int, camera_id: int | None) -> None:
    if not alert.webhook_url:
        return
    payload = {
        "alert_id": alert.id,
        "alert_name": alert.name,
        "threshold": alert.threshold,
        "observed_count": person_count,
        "camera_id": camera_id,
        "triggered_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        with httpx.Client(timeout=5.0) as client:
            client.post(alert.webhook_url, json=payload)
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("Failed to dispatch webhook for alert %s: %s", alert.id, exc)