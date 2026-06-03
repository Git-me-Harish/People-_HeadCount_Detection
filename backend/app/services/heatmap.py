"""Heatmap service — aggregate bounding boxes into a density grid.

Grid: 16 columns × 9 rows = 144 cells.
Each detection bbox contributes its centroid to the grid cell.
Grid values are cumulative counts (not normalized) — normalization
happens at read time so historical comparisons stay accurate.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from ..models import HeatmapSnapshot
from ..schemas.common import BoundingBox

logger = logging.getLogger(__name__)

GRID_COLS = 16
GRID_ROWS = 9
GRID_SIZE = GRID_COLS * GRID_ROWS  # 144


def _cell_index(x_norm: float, y_norm: float) -> int:
    """Convert normalised (0-1) coordinates to flat grid index."""
    col = min(int(x_norm * GRID_COLS), GRID_COLS - 1)
    row = min(int(y_norm * GRID_ROWS), GRID_ROWS - 1)
    return row * GRID_COLS + col


def update_heatmap(
    db: Session,
    *,
    organization_id: int,
    camera_id: int,
    bboxes: list[BoundingBox],
    frame_width: int,
    frame_height: int,
) -> HeatmapSnapshot:
    """Upsert the current-hour heatmap snapshot for a camera.

    Args:
        bboxes: List of bounding boxes from the detector (pixel coords).
        frame_width / frame_height: Image dimensions for normalisation.
    """
    now = datetime.now(timezone.utc)
    bucket = now.replace(minute=0, second=0, microsecond=0)

    snap = (
        db.query(HeatmapSnapshot)
        .filter(
            HeatmapSnapshot.organization_id == organization_id,
            HeatmapSnapshot.camera_id == camera_id,
            HeatmapSnapshot.bucket_hour == bucket,
        )
        .first()
    )

    if snap is None:
        snap = HeatmapSnapshot(
            organization_id=organization_id,
            camera_id=camera_id,
            bucket_hour=bucket,
            grid_json=json.dumps([0.0] * GRID_SIZE),
            sample_count=0,
            peak_count=0,
        )
        db.add(snap)

    grid: list[float] = json.loads(snap.grid_json)
    if len(grid) != GRID_SIZE:
        grid = [0.0] * GRID_SIZE

    if frame_width > 0 and frame_height > 0:
        for bbox in bboxes:
            cx = ((bbox.x1 + bbox.x2) / 2) / frame_width
            cy = ((bbox.y1 + bbox.y2) / 2) / frame_height
            idx = _cell_index(cx, cy)
            grid[idx] += 1.0

    snap.grid_json = json.dumps(grid)
    snap.sample_count += 1
    snap.peak_count = max(snap.peak_count, len(bboxes))
    db.commit()
    db.refresh(snap)
    return snap


def get_normalized_grid(snapshot: HeatmapSnapshot) -> list[float]:
    """Return grid normalized 0-1 relative to max cell value."""
    grid: list[float] = json.loads(snapshot.grid_json)
    max_val = max(grid) if grid else 0.0
    if max_val == 0:
        return grid
    return [round(v / max_val, 4) for v in grid]
