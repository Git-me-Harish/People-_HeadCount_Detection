"""Job model for async detection workloads (video processing, batch jobs)."""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from ..db import Base


class JobStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class JobType(str, enum.Enum):
    video = "video"
    batch_image = "batch_image"


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    organization_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    job_type: Mapped[JobType] = mapped_column(
        Enum(JobType, native_enum=False, length=20), nullable=False
    )
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, native_enum=False, length=20),
        default=JobStatus.pending,
        nullable=False,
    )
    progress: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    input_path: Mapped[str] = mapped_column(String(512), nullable=False)
    output_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    summary_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
