"""IndustryTemplate — pre-configured vertical setups for onboarding."""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import DateTime, Enum, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from ..db import Base


class Vertical(str, enum.Enum):
    religious = "religious"
    transit = "transit"
    retail = "retail"
    hospital = "hospital"
    education = "education"
    stadium = "stadium"
    workplace = "workplace"
    tourism = "tourism"


class IndustryTemplate(Base):
    """Seeded read-only templates; applied to an org via routers/templates.py."""

    __tablename__ = "industry_templates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    vertical: Mapped[Vertical] = mapped_column(
        Enum(Vertical, native_enum=False, length=30), nullable=False, unique=True, index=True
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    description: Mapped[str] = mapped_column(String(500), nullable=False)
    icon: Mapped[str] = mapped_column(String(50), nullable=False)  # emoji or icon key
    # JSON strings — kept as Text to avoid dialect-specific JSON column
    default_cameras_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    default_alerts_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
