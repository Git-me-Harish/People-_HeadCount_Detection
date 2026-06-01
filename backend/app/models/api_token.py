"""APIToken — customer-managed bearer tokens for programmatic API access.

Design:
  - Prefix stored plaintext (ps_<8chars>) for identification in logs.
  - Full token = prefix + secret; only secret is bcrypt-hashed at rest.
  - Secret shown exactly once on creation; thereafter only prefix visible.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from ..db import Base


class APIToken(Base):
    __tablename__ = "api_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    organization_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    created_by_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    prefix: Mapped[str] = mapped_column(String(20), nullable=False, unique=True, index=True)
    hashed_secret: Mapped[str] = mapped_column(String(255), nullable=False)
    # Comma-separated scope strings e.g. "read:detections,read:analytics"
    scopes: Mapped[str] = mapped_column(Text, nullable=False, default="read:all")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
