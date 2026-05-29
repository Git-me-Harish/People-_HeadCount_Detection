"""Image & video detection endpoints."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import get_db
from ..deps import get_current_user
from ..models import DetectionRecord, Job, JobStatus, JobType, User
from ..schemas.common import DetectionResult
from ..schemas.job import JobRead
from ..services.alerts import evaluate_alerts
from ..services.detector import get_detector
from ..workers.jobs import run_video_job

router = APIRouter(prefix="/detect", tags=["detect"])

ALLOWED_IMAGE = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp"}
ALLOWED_VIDEO = {"video/mp4", "video/quicktime", "video/x-msvideo", "video/webm"}


def _validate_size(upload: UploadFile, max_mb: int) -> None:
    if upload.size is not None and upload.size > max_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large (max {max_mb} MB)")


@router.post("/image", response_model=DetectionResult)
async def detect_image(
    file: UploadFile = File(...),
    confidence: float | None = Form(None),
    annotate: bool = Form(True),
    camera_id: int | None = Form(None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> DetectionResult:
    settings = get_settings()
    if file.content_type not in ALLOWED_IMAGE:
        raise HTTPException(status_code=400, detail=f"Unsupported image type: {file.content_type}")
    _validate_size(file, settings.max_upload_size_mb)

    data = await file.read()
    detector = get_detector()
    result = detector.detect_image_bytes(data, conf=confidence, annotate=annotate)

    record = DetectionRecord(
        organization_id=user.organization_id,
        camera_id=camera_id,
        source="image",
        person_count=result.person_count,
        unique_people=result.person_count,
        avg_confidence=result.avg_confidence,
    )
    db.add(record)
    db.commit()

    evaluate_alerts(
        db,
        organization_id=user.organization_id,
        camera_id=camera_id,
        person_count=result.person_count,
    )
    return result


@router.post("/video", response_model=JobRead, status_code=status.HTTP_202_ACCEPTED)
async def detect_video(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Job:
    settings = get_settings()
    if file.content_type and file.content_type not in ALLOWED_VIDEO:
        raise HTTPException(status_code=400, detail=f"Unsupported video type: {file.content_type}")
    _validate_size(file, settings.max_upload_size_mb)

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "input.mp4").suffix or ".mp4"
    dest = settings.upload_dir / f"{uuid.uuid4().hex}{suffix}"
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    job = Job(
        organization_id=user.organization_id,
        user_id=user.id,
        job_type=JobType.video,
        status=JobStatus.pending,
        input_path=str(dest),
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    background.add_task(run_video_job, job.id)
    return job


@router.get("/status", tags=["meta"])
def detector_status() -> dict:
    return get_detector().status
