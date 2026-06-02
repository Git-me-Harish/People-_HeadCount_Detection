"""Job listing / polling / artifact download / cancellation endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import get_current_user
from ..models import Job, JobStatus, User, UserRole
from ..schemas.job import JobRead
from ..workers.jobs import request_cancel

router = APIRouter(prefix="/jobs", tags=["jobs"])


def _get_job_or_404(job_id: int, user: User, db: Session) -> Job:
    """Fetch job, enforce org-scoped ownership."""
    job = db.get(Job, job_id)
    if job is None or job.organization_id != user.organization_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return job


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
    return _get_job_or_404(job_id, user, db)


@router.delete("/{job_id}", status_code=status.HTTP_200_OK)
def cancel_job(
    job_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    """Cancel a pending or running job.

    - **pending** jobs: status flipped to cancelled immediately — the
      background thread hasn't started yet so no signal is needed.
    - **running** jobs: a threading.Event is set; the video processor's
      progress callback checks it on every frame and raises
      ``JobCancelledError``, which the worker catches and sets
      status = cancelled cleanly.
    - **completed / failed / cancelled**: returns 409 — nothing to cancel.

    Only admins and members can cancel; viewers are rejected with 403.
    """
    job = _get_job_or_404(job_id, user, db)

    if user.role == UserRole.viewer:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Viewers cannot cancel jobs",
        )

    if job.status == JobStatus.pending:
        # Background thread hasn't started yet — flip status directly
        job.status = JobStatus.cancelled
        job.error_message = "Cancelled by user before processing started"
        db.commit()
        return {"job_id": job_id, "status": "cancelled", "method": "immediate"}

    if job.status == JobStatus.running:
        signalled = request_cancel(job_id)
        if not signalled:
            # Race: thread just finished between our check and signal attempt
            db.refresh(job)
            if job.status in (JobStatus.completed, JobStatus.failed, JobStatus.cancelled):
                return {"job_id": job_id, "status": job.status, "method": "already_finished"}
            # Event not registered for some reason — mark failed-safe
            job.status = JobStatus.cancelled
            job.error_message = "Cancelled (worker handle unavailable)"
            db.commit()
            return {"job_id": job_id, "status": "cancelled", "method": "fallback"}
        return {
            "job_id": job_id,
            "status": "cancellation_requested",
            "method": "signal",
            "note": "Worker will stop at next frame boundary; poll GET /jobs/{id} for final status.",
        }

    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail=f"Job is already {job.status} — nothing to cancel",
    )


@router.get("/{job_id}/artifact")
def download_artifact(
    job_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> FileResponse:
    job = _get_job_or_404(job_id, user, db)
    if not job.output_path:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not ready")
    path = Path(job.output_path)
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact missing on disk")
    return FileResponse(path, media_type="video/mp4", filename=path.name)
