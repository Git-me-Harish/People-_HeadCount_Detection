"""Tests for notifications inbox."""

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models import Notification, NotificationChannel, NotificationStatus


def _seed_notification(
    db: Session, org_id: int, user_id: int, is_read: bool = False
) -> Notification:
    n = Notification(
        organization_id=org_id,
        user_id=user_id,
        channel=NotificationChannel.inbox,
        status=NotificationStatus.sent,
        title="Test Alert",
        body="Threshold exceeded",
        is_read=is_read,
    )
    db.add(n)
    db.commit()
    db.refresh(n)
    return n


def test_list_notifications_empty(client: TestClient, auth_headers: dict) -> None:
    resp = client.get("/api/v1/notifications", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json() == []


def test_unread_count(client: TestClient, auth_headers: dict, db: Session, test_user) -> None:
    _seed_notification(db, test_user.organization_id, test_user.id)
    resp = client.get("/api/v1/notifications/count-unread", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["unread_count"] == 1


def test_mark_all_read(client: TestClient, auth_headers: dict, db: Session, test_user) -> None:
    _seed_notification(db, test_user.organization_id, test_user.id)
    _seed_notification(db, test_user.organization_id, test_user.id)
    resp = client.post("/api/v1/notifications/mark-all-read", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["marked_read"] == 2
    # confirm count is now 0
    count_resp = client.get("/api/v1/notifications/count-unread", headers=auth_headers)
    assert count_resp.json()["unread_count"] == 0
