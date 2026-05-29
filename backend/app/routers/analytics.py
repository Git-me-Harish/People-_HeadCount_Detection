"""Analytics endpoints — time series, summary KPIs, recent detections."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import get_current_user
from ..models import DetectionRecord, User
from ..schemas.detection import DetectionRecordRead

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/summary")
def summary(
    days: int = Query(7, ge=1, le=365),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    base = db.query(DetectionRecord).filter(
        DetectionRecord.organization_id == user.organization_id,
        DetectionRecord.created_at >= since,
    )
    total = base.count()
    peak = (
        db.query(func.max(DetectionRecord.person_count))
        .filter(
            DetectionRecord.organization_id == user.organization_id,
            DetectionRecord.created_at >= since,
        )
        .scalar()
        or 0
    )
    avg = (
        db.query(func.avg(DetectionRecord.person_count))
        .filter(
            DetectionRecord.organization_id == user.organization_id,
            DetectionRecord.created_at >= since,
        )
        .scalar()
        or 0
    )
    latest = (
        db.query(DetectionRecord)
        .filter(DetectionRecord.organization_id == user.organization_id)
        .order_by(DetectionRecord.created_at.desc())
        .first()
    )
    return {
        "window_days": days,
        "total_detections": int(total),
        "peak_person_count": int(peak),
        "average_person_count": round(float(avg), 2),
        "current_count": int(latest.person_count) if latest else 0,
        "last_seen_at": latest.created_at.isoformat() if latest else None,
    }


@router.get("/timeseries")
def timeseries(
    days: int = Query(7, ge=1, le=365),
    bucket_minutes: int = Query(60, ge=1, le=24 * 60),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[dict]:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    records = (
        db.query(DetectionRecord)
        .filter(
            DetectionRecord.organization_id == user.organization_id,
            DetectionRecord.created_at >= since,
        )
        .order_by(DetectionRecord.created_at.asc())
        .all()
    )
    buckets: dict[datetime, list[int]] = {}
    for r in records:
        anchor = r.created_at - timedelta(
            minutes=r.created_at.minute % bucket_minutes,
            seconds=r.created_at.second,
            microseconds=r.created_at.microsecond,
        )
        buckets.setdefault(anchor, []).append(r.person_count)
    return [
        {
            "timestamp": ts.isoformat(),
            "count": int(sum(counts) / len(counts)),
            "peak": int(max(counts)),
            "samples": len(counts),
        }
        for ts, counts in sorted(buckets.items())
    ]


@router.get("/records", response_model=list[DetectionRecordRead])
def list_records(
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[DetectionRecord]:
    return (
        db.query(DetectionRecord)
        .filter(DetectionRecord.organization_id == user.organization_id)
        .order_by(DetectionRecord.created_at.desc())
        .limit(limit)
        .all()
    )
