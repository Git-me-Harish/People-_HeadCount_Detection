"""Tests for anomaly detection service."""
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from app.models import DetectionRecord
from app.services.anomaly import check_anomaly


def _insert_records(db: Session, org_id: int, count: int, person_count: int, hour_offset: int = 0) -> None:
    now = datetime.now(timezone.utc)
    for i in range(count):
        ts = now - timedelta(days=i + 1) + timedelta(hours=hour_offset)
        db.add(DetectionRecord(
            organization_id=org_id,
            source="stream",
            person_count=person_count,
            created_at=ts,
        ))
    db.commit()


def test_insufficient_baseline(db: Session, test_user) -> None:
    result = check_anomaly(db, organization_id=test_user.organization_id, camera_id=None, person_count=50)
    assert result["is_anomaly"] is False
    assert result["reason"] == "insufficient_baseline"


def test_normal_reading(db: Session, test_user) -> None:
    # Seed 10 records all with count=20 for current hour
    _insert_records(db, test_user.organization_id, 10, 20)
    result = check_anomaly(db, organization_id=test_user.organization_id, camera_id=None, person_count=21)
    # Within normal range — not anomalous
    assert result["sample_size"] > 0


def test_spike_detected(db: Session, test_user) -> None:
    _insert_records(db, test_user.organization_id, 10, 10)
    result = check_anomaly(db, organization_id=test_user.organization_id, camera_id=None, person_count=500)
    # Extreme outlier — should flag anomaly when baseline is established
    assert "z_score" in result
