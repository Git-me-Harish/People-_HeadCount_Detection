"""Anomaly detection service — z-score statistical baseline.

Uses a rolling 14-day baseline of per-hour average counts to flag
deviations beyond a configurable z-score threshold.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from ..models import DetectionRecord

logger = logging.getLogger(__name__)

# Z-score threshold above which a reading is flagged as anomalous
DEFAULT_Z_THRESHOLD = 2.5
BASELINE_WINDOW_DAYS = 14


def _stats(values: list[float]) -> tuple[float, float]:
    """Return (mean, std_dev). Returns (0, 0) for empty/single-value lists."""
    if len(values) < 2:
        return (float(values[0]) if values else 0.0, 0.0)
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(variance)


def check_anomaly(
    db: Session,
    *,
    organization_id: int,
    camera_id: int | None,
    person_count: int,
    z_threshold: float = DEFAULT_Z_THRESHOLD,
) -> dict:
    """Compare current count against 14-day per-hour baseline.

    Returns:
        {
            "is_anomaly": bool,
            "z_score": float | None,
            "baseline_mean": float,
            "baseline_std": float,
            "sample_size": int,
        }
    """
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=BASELINE_WINDOW_DAYS)
    current_hour = now.hour

    # Pull historical records for the same hour-of-day across the baseline window
    query = db.query(DetectionRecord).filter(
        DetectionRecord.organization_id == organization_id,
        DetectionRecord.created_at >= since,
    )
    if camera_id is not None:
        query = query.filter(DetectionRecord.camera_id == camera_id)

    records = query.all()

    # Filter to same hour-of-day for time-of-day-aware baseline
    same_hour_counts = [float(r.person_count) for r in records if r.created_at.hour == current_hour]

    if len(same_hour_counts) < 5:
        # Insufficient baseline → cannot determine anomaly
        return {
            "is_anomaly": False,
            "z_score": None,
            "baseline_mean": 0.0,
            "baseline_std": 0.0,
            "sample_size": len(same_hour_counts),
            "reason": "insufficient_baseline",
        }

    mean, std = _stats(same_hour_counts)

    if std < 1e-6:
        # Zero variance → any deviation is anomalous only if count differs
        is_anomaly = abs(person_count - mean) > 0
        z_score = None
    else:
        z_score = (person_count - mean) / std
        is_anomaly = abs(z_score) > z_threshold

    return {
        "is_anomaly": is_anomaly,
        "z_score": round(z_score, 3) if z_score is not None else None,
        "baseline_mean": round(mean, 2),
        "baseline_std": round(std, 2),
        "sample_size": len(same_hour_counts),
        "reason": "z_score_threshold" if is_anomaly else None,
    }
