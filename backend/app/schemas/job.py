"""Job schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from ..models.job import JobStatus, JobType


class JobRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    organization_id: int
    user_id: int
    job_type: JobType
    status: JobStatus
    progress: float
    input_path: str
    output_path: str | None
    summary_json: str | None
    error_message: str | None
    created_at: datetime
    completed_at: datetime | None
