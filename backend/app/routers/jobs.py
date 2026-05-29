"""Job listing / polling / artifact download endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import get_current_user
from ..models import Job, User
from ..schemas.job import JobRead

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("", response_model=list[JobRead])
def list_jobs(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[Job]:
    return (
        db.query(Job)
        .filter(Job.organization_id == user.organization_id)
        .order_by(Job.created_at.desc())
        .limit(100)
        .all()
    )


@router.get("/{job_id}", response_model=JobRead)
def get_job(
    job_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Job:
    job = db.get(Job, job_id)
    if job is None or job.organization_id != user.organization_id:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/{job_id}/artifact")
def download_artifact(
    job_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    job = db.get(Job, job_id)
    if job is None or job.organization_id != user.organization_id:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.output_path:
        raise HTTPException(status_code=404, detail="Artifact not ready")
    path = Path(job.output_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact missing on disk")
    return FileResponse(path, media_type="video/mp4", filename=path.name)
