"""Shared schemas for detection results and job summaries."""

from __future__ import annotations

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    track_id: int | None = None


class DetectionResult(BaseModel):
    person_count: int
    detections: list[Detection]
    annotated_image_b64: str | None = None
    avg_confidence: float | None = None
    width: int | None = None
    height: int | None = None


class JobSummary(BaseModel):
    frames_processed: int = 0
    total_frames: int = 0
    average_person_count: float = 0.0
    peak_person_count: int = 0
    unique_people: int = 0
    per_frame: list[int] = Field(default_factory=list)
    duration_seconds: float | None = None
