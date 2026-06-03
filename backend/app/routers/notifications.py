"""Notifications router — in-app inbox CRUD."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import get_current_user
from ..models import Notification, NotificationChannel, User

router = APIRouter(prefix="/notifications", tags=["notifications"])


class NotificationRead(BaseModel):
    id: int
    channel: str
    status: str
    title: str
    body: str
    source_type: str | None
    source_id: int | None
    is_read: bool
    created_at: datetime
    read_at: datetime | None

    model_config = {"from_attributes": True}


class MarkReadRequest(BaseModel):
    notification_ids: list[int]


@router.get("", response_model=list[NotificationRead])
def list_notifications(
    unread_only: bool = Query(False),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[Notification]:
    """Return in-app inbox notifications for the current user."""
    q = db.query(Notification).filter(
        Notification.organization_id == user.organization_id,
        Notification.channel == NotificationChannel.inbox,
    )
    if unread_only:
        q = q.filter(Notification.is_read.is_(False))
    return q.order_by(Notification.created_at.desc()).limit(limit).all()


@router.get("/count-unread")
def count_unread(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    count = (
        db.query(Notification)
        .filter(
            Notification.organization_id == user.organization_id,
            Notification.channel == NotificationChannel.inbox,
            Notification.is_read.is_(False),
        )
        .count()
    )
    return {"unread_count": count}


@router.post("/mark-read", status_code=status.HTTP_200_OK)
def mark_read(
    payload: MarkReadRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    """Mark specific notification IDs as read."""
    now = datetime.now(timezone.utc)
    updated = (
        db.query(Notification)
        .filter(
            Notification.id.in_(payload.notification_ids),
            Notification.organization_id == user.organization_id,
        )
        .all()
    )
    for n in updated:
        n.is_read = True
        n.read_at = now
    db.commit()
    return {"marked_read": len(updated)}


@router.post("/mark-all-read", status_code=status.HTTP_200_OK)
def mark_all_read(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    notifications = (
        db.query(Notification)
        .filter(
            Notification.organization_id == user.organization_id,
            Notification.channel == NotificationChannel.inbox,
            Notification.is_read.is_(False),
        )
        .all()
    )
    now = datetime.now(timezone.utc)
    for n in notifications:
        n.is_read = True
        n.read_at = now
    db.commit()
    return {"marked_read": len(notifications)}


@router.delete("/{notification_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_notification(
    notification_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> None:
    notif = (
        db.query(Notification)
        .filter(
            Notification.id == notification_id,
            Notification.organization_id == user.organization_id,
        )
        .first()
    )
    if notif is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Notification not found")
    db.delete(notif)
    db.commit()
