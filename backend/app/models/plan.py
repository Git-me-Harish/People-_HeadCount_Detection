"""Plan + UsageCounter — subscription tier and usage metering."""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from ..db import Base


class PlanTier(str, enum.Enum):
    free = "free"
    pro = "pro"
    enterprise = "enterprise"


class Plan(Base):
    """Defines limits for each subscription tier."""

    __tablename__ = "plans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tier: Mapped[PlanTier] = mapped_column(
        Enum(PlanTier, native_enum=False, length=20), unique=True, nullable=False
    )
    display_name: Mapped[str] = mapped_column(String(50), nullable=False)
    max_cameras: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    max_alerts: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    max_api_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=2)
    # -1 = unlimited
    max_frames_per_month: Mapped[int] = mapped_column(Integer, nullable=False, default=10_000)
    retention_days: Mapped[int] = mapped_column(Integer, nullable=False, default=30)
    can_export_pdf: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    can_use_public_page: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    price_usd_monthly: Mapped[int] = mapped_column(Integer, nullable=False, default=0)  # cents


class UsageCounter(Base):
    """Monthly rolling usage per organization."""

    __tablename__ = "usage_counters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    organization_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    plan_tier: Mapped[PlanTier] = mapped_column(
        Enum(PlanTier, native_enum=False, length=20), nullable=False, default=PlanTier.free
    )
    cameras_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    frames_processed_month: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    alerts_sent_month: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    period_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
