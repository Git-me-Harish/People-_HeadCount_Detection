"""Alert CRUD endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import get_current_user
from ..models import Alert, User
from ..schemas.alert import AlertCreate, AlertRead, AlertUpdate

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("", response_model=list[AlertRead])
def list_alerts(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[Alert]:
    return (
        db.query(Alert)
        .filter(Alert.organization_id == user.organization_id)
        .order_by(Alert.created_at.desc())
        .all()
    )


@router.post("", response_model=AlertRead, status_code=status.HTTP_201_CREATED)
def create_alert(
    payload: AlertCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Alert:
    alert = Alert(organization_id=user.organization_id, **payload.model_dump())
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return alert


@router.patch("/{alert_id}", response_model=AlertRead)
def update_alert(
    alert_id: int,
    payload: AlertUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Alert:
    alert = db.get(Alert, alert_id)
    if alert is None or alert.organization_id != user.organization_id:
        raise HTTPException(status_code=404, detail="Alert not found")
    for k, v in payload.model_dump(exclude_unset=True).items():
        setattr(alert, k, v)
    db.commit()
    db.refresh(alert)
    return alert


@router.delete("/{alert_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_alert(
    alert_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> None:
    alert = db.get(Alert, alert_id)
    if alert is None or alert.organization_id != user.organization_id:
        raise HTTPException(status_code=404, detail="Alert not found")
    db.delete(alert)
    db.commit()
