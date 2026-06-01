"""HeatmapSnapshot — aggregated density grid for a camera at a point in time."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from ..db import Base


class HeatmapSnapshot(Base):
    """Stores a serialized density grid (JSON array of floats) per camera per hour.

    Grid is a flattened 16×9 matrix of cumulative person-density scores,
    normalized 0-1. Rows = vertical bands, cols = horizontal bands.
    Aggregated from bbox centroids during detection.
    """

    __tablename__ = "heatmap_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    organization_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    camera_id: Mapped[int] = mapped_column(
        ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # ISO-truncated to hour — one snapshot per camera per hour
    bucket_hour: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    # JSON: flat list of 144 floats (16*9 grid)
    grid_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    peak_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
