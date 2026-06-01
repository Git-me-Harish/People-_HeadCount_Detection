"""Audit log router — read-only audit trail for admins."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import get_current_user
from ..models import AuditLog, User, UserRole

router = APIRouter(prefix="/audit", tags=["audit"])


class AuditLogRead(BaseModel):
    id: int
    actor_email: str | None
    action: str
    resource_type: str
    resource_id: str | None
    ip_address: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


def _require_admin(user: User) -> User:
    if user.role != UserRole.admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required to view audit logs",
        )
    return user


@router.get("", response_model=list[AuditLogRead])
def list_audit_logs(
    limit: int = Query(100, ge=1, le=500),
    resource_type: str | None = Query(None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[AuditLog]:
    _require_admin(user)
    q = db.query(AuditLog).filter(AuditLog.organization_id == user.organization_id)
    if resource_type:
        q = q.filter(AuditLog.resource_type == resource_type)
    return q.order_by(AuditLog.created_at.desc()).limit(limit).all()
