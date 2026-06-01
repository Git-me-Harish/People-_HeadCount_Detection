"""Heatmap router — per-camera density grid endpoints."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import get_current_user
from ..models import Camera, HeatmapSnapshot, User
from ..services.heatmap import GRID_COLS, GRID_ROWS, get_normalized_grid

router = APIRouter(prefix="/heatmaps", tags=["heatmaps"])


class HeatmapResponse(BaseModel):
    camera_id: int
    camera_name: str
    bucket_hour: datetime
    grid: list[float]  # 144 normalised floats
    grid_cols: int
    grid_rows: int
    sample_count: int
    peak_count: int


@router.get("/camera/{camera_id}/latest", response_model=HeatmapResponse)
def get_latest_heatmap(
    camera_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> HeatmapResponse:
    """Return the most recent heatmap snapshot for a camera."""
    cam = db.query(Camera).filter(
        Camera.id == camera_id, Camera.organization_id == user.organization_id
    ).first()
    if cam is None:
        raise HTTPException(status_code=404, detail="Camera not found")

    snap = (
        db.query(HeatmapSnapshot)
        .filter(
            HeatmapSnapshot.camera_id == camera_id,
            HeatmapSnapshot.organization_id == user.organization_id,
        )
        .order_by(HeatmapSnapshot.bucket_hour.desc())
        .first()
    )
    if snap is None:
        raise HTTPException(status_code=404, detail="No heatmap data available yet")

    return HeatmapResponse(
        camera_id=cam.id,
        camera_name=cam.name,
        bucket_hour=snap.bucket_hour,
        grid=get_normalized_grid(snap),
        grid_cols=GRID_COLS,
        grid_rows=GRID_ROWS,
        sample_count=snap.sample_count,
        peak_count=snap.peak_count,
    )


@router.get("/camera/{camera_id}/history", response_model=list[HeatmapResponse])
def get_heatmap_history(
    camera_id: int,
    hours: int = Query(24, ge=1, le=168),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[HeatmapResponse]:
    """Return hourly heatmap snapshots for the last N hours."""
    cam = db.query(Camera).filter(
        Camera.id == camera_id, Camera.organization_id == user.organization_id
    ).first()
    if cam is None:
        raise HTTPException(status_code=404, detail="Camera not found")

    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    snaps = (
        db.query(HeatmapSnapshot)
        .filter(
            HeatmapSnapshot.camera_id == camera_id,
            HeatmapSnapshot.organization_id == user.organization_id,
            HeatmapSnapshot.bucket_hour >= since,
        )
        .order_by(HeatmapSnapshot.bucket_hour.asc())
        .all()
    )
    return [
        HeatmapResponse(
            camera_id=cam.id,
            camera_name=cam.name,
            bucket_hour=s.bucket_hour,
            grid=get_normalized_grid(s),
            grid_cols=GRID_COLS,
            grid_rows=GRID_ROWS,
            sample_count=s.sample_count,
            peak_count=s.peak_count,
        )
        for s in snaps
    ]
