"""Detection record schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DetectionRecordRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    organization_id: int
    camera_id: int | None
    source: str
    person_count: int
    unique_people: int | None
    avg_confidence: float | None
    artifact_path: str | None
    created_at: datetime
