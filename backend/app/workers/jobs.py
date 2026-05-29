"""Background worker entrypoints for async detection jobs."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from ..db import SessionLocal
from ..models import DetectionRecord, Job, JobStatus
from ..services.alerts import evaluate_alerts
from ..services.video_processor import process_video, summary_to_json

logger = logging.getLogger(__name__)


def run_video_job(job_id: int) -> None:
    """Background task: process a video job end-to-end."""
    db = SessionLocal()
    try:
        job = db.get(Job, job_id)
        if job is None:
            logger.warning("Job %s not found", job_id)
            return
        job.status = JobStatus.running
        job.progress = 0.0
        db.commit()

        def update_progress(pct: float, processed: int, total: int) -> None:
            job.progress = round(pct, 4)
            db.commit()
            logger.debug("Job %s progress: %.2f%% (%d/%d)", job_id, pct * 100, processed, total)

        output_path = Path(job.input_path).with_suffix(".annotated.mp4")
        summary = process_video(Path(job.input_path), output_path, progress_cb=update_progress)

        job.output_path = str(output_path)
        job.summary_json = summary_to_json(summary)
        job.progress = 1.0
        job.status = JobStatus.completed
        job.completed_at = datetime.now(timezone.utc)

        record = DetectionRecord(
            organization_id=job.organization_id,
            source="video",
            person_count=summary.peak_person_count,
            unique_people=summary.unique_people,
            avg_confidence=None,
            artifact_path=str(output_path),
        )
        db.add(record)
        db.commit()

        evaluate_alerts(
            db,
            organization_id=job.organization_id,
            camera_id=None,
            person_count=summary.peak_person_count,
        )
    except Exception as exc:  # pragma: no cover - background-only
        logger.exception("Job %s failed", job_id)
        job = db.get(Job, job_id)
        if job is not None:
            job.status = JobStatus.failed
            job.error_message = str(exc)
            db.commit()
    finally:
        db.close()
