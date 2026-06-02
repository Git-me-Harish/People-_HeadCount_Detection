"""Background worker entrypoints for async detection jobs.

Cancellation contract
---------------------
Each running job registers a ``threading.Event`` in ``_cancel_events``.
Calling ``request_cancel(job_id)`` sets that event. The video processor's
``progress_cb`` checks the event on every frame and raises
``JobCancelledError`` when set — which unwinds to the except block below,
setting status = cancelled cleanly.

The Event is removed from the registry when the job finishes (any outcome).
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

from ..db import SessionLocal
from ..models import DetectionRecord, Job, JobStatus, NotificationChannel
from ..services.alerts import evaluate_alerts
from ..services.anomaly import check_anomaly
from ..services.notifier import dispatch as notifier_dispatch
from ..services.video_processor import process_video, summary_to_json

logger = logging.getLogger(__name__)

# Cancellation registry:
# job_id → Event; set() = "please stop"
_cancel_events: dict[int, threading.Event] = {}
_registry_lock = threading.Lock()


class JobCancelledError(Exception):
    """Raised by progress_cb when cancellation is requested."""


def request_cancel(job_id: int) -> bool:
    """Signal a running job to stop. Returns True if the signal was sent."""
    with _registry_lock:
        event = _cancel_events.get(job_id)
        if event is None:
            return False
        event.set()
        return True


def _register(job_id: int) -> threading.Event:
    event = threading.Event()
    with _registry_lock:
        _cancel_events[job_id] = event
    return event


def _unregister(job_id: int) -> None:
    with _registry_lock:
        _cancel_events.pop(job_id, None)


def is_cancellable(job_id: int) -> bool:
    """True if the job is currently running and has a cancel handle."""
    with _registry_lock:
        return job_id in _cancel_events


# Worker:

def run_video_job(job_id: int) -> None:
    """Background task: process a video job end-to-end."""
    db = SessionLocal()
    cancel_event = _register(job_id)
    try:
        job = db.get(Job, job_id)
        if job is None:
            logger.warning("Job %s not found", job_id)
            return

        job.status = JobStatus.running
        job.progress = 0.0
        db.commit()

        def update_progress(pct: float, processed: int, total: int) -> None:
            # Check for cancellation on every progress tick (every N frames via caller)
            if cancel_event.is_set():
                raise JobCancelledError(f"Job {job_id} cancelled by user request")
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

        anomaly = check_anomaly(
            db,
            organization_id=job.organization_id,
            camera_id=None,
            person_count=summary.peak_person_count,
        )
        if anomaly.get("is_anomaly"):
            notifier_dispatch(
                db,
                organization_id=job.organization_id,
                user_id=job.user_id,
                title="Anomaly detected in video",
                body=(
                    f"Peak count {summary.peak_person_count} in job {job_id} is unusual "
                    f"(z-score: {anomaly.get('z_score')}, baseline mean: {anomaly.get('baseline_mean')})"
                ),
                channels=[NotificationChannel.inbox],
                source_type="job",
                source_id=job_id,
            )

    except JobCancelledError:
        logger.info("Job %s was cancelled", job_id)
        job = db.get(Job, job_id)
        if job is not None:
            job.status = JobStatus.cancelled
            job.error_message = "Cancelled by user"
            job.completed_at = datetime.now(timezone.utc)
            db.commit()

    except Exception as exc:  # pragma: no cover - background-only
        logger.exception("Job %s failed", job_id)
        job = db.get(Job, job_id)
        if job is not None:
            job.status = JobStatus.failed
            job.error_message = str(exc)
            db.commit()

    finally:
        _unregister(job_id)
        db.close()