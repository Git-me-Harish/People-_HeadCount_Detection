"""Alert schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class AlertBase(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    threshold: int = Field(ge=1)
    camera_id: int | None = None
    webhook_url: str | None = None
    is_enabled: bool = True


class AlertCreate(AlertBase):
    pass


class AlertUpdate(BaseModel):
    name: str | None = None
    threshold: int | None = None
    camera_id: int | None = None
    webhook_url: str | None = None
    is_enabled: bool | None = None


class AlertRead(AlertBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    organization_id: int
    last_triggered_at: datetime | None
    created_at: datetime
