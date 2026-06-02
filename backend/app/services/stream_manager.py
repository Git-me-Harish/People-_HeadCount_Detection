"""CameraStreamManager — server-side RTSP / HTTP stream puller.

Architecture
------------
One ``threading.Thread`` per active camera.  Each thread:

  1. Opens the stream URL via ``cv2.VideoCapture``.
  2. Reads frames in a tight loop, throttled by ``sample_interval_s``
     so we don't saturate the GPU/CPU at 30fps.
  3. Runs YOLOv8 inference on every sampled frame.
  4. Calls ``evaluate_alerts``, ``update_heatmap``, ``check_anomaly``
     using a **thread-local** SQLAlchemy session (never shared).
  5. Persists a ``DetectionRecord`` every ``persist_every_n`` samples.
  6. On any read failure, enters an **exponential-backoff reconnect loop**
     capped at ``reconnect_max_backoff_s``.  Permanent stop only after
     ``max_consecutive_errors`` consecutive failures without a successful
     frame between them.

Lifecycle
---------
  CameraStreamManager.get()            — process singleton
  manager.start_camera(camera_id, org_id, stream_url)
  manager.stop_camera(camera_id)
  manager.restart_camera(camera_id, org_id, stream_url)
  manager.start_all_active_cameras()   — called on app boot from lifespan
  manager.stop_all()                   — called on app shutdown

Thread safety
-------------
  ``_threads`` and ``_stop_events`` are protected by ``_lock`` for all
  mutations.  Status reads are lock-free (dict lookup is GIL-safe in CPython).

Multi-worker note
-----------------
  Each uvicorn worker process runs its own CameraStreamManager instance.
  With ``--workers N``, N copies of every camera thread run in parallel.
  For production multi-worker deployments, run stream pulling in a
  dedicated single-process worker (Celery / ARQ / standalone script)
  and remove stream management from the web process lifespan.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


# ── Thread state ──────────────────────────────────────────────────────────────

class StreamState(str, Enum):
    idle      = "idle"
    starting  = "starting"
    running   = "running"
    reconnecting = "reconnecting"
    stopping  = "stopping"
    stopped   = "stopped"
    error     = "error"


@dataclass
class StreamStatus:
    camera_id: int
    state: StreamState = StreamState.idle
    last_frame_at: datetime | None = None
    last_person_count: int = 0
    frames_processed: int = 0
    consecutive_errors: int = 0
    reconnect_attempts: int = 0
    error_message: str | None = None
    started_at: datetime | None = None


# ── Per-camera worker thread ──────────────────────────────────────────────────

def _stream_worker(
    camera_id: int,
    org_id: int,
    stream_url: str,
    stop_event: threading.Event,
    status: StreamStatus,
) -> None:
    """Main loop for a single camera stream thread.

    Runs until stop_event is set or max_consecutive_errors is reached.
    Each iteration: open capture → read frames → infer → persist → repeat.
    On cv2 failure, backoff and retry.
    """
    from ..config import get_settings
    from ..db import SessionLocal
    from ..models import DetectionRecord, NotificationChannel
    from ..services.alerts import evaluate_alerts
    from ..services.anomaly import check_anomaly
    from ..services.detector import get_detector
    from ..services.heatmap import update_heatmap
    from ..services.notifier import dispatch as notifier_dispatch

    settings   = get_settings()
    detector   = get_detector()
    sample_interval   = settings.stream_sample_interval_s
    persist_every     = settings.stream_persist_every_n_samples
    max_backoff       = settings.stream_reconnect_max_backoff_s
    max_errors        = settings.stream_max_consecutive_errors

    # Local imports so cv2 absence doesn't crash the whole app on import
    try:
        import cv2  # type: ignore
    except ImportError:
        logger.error(
            "Camera %s: cv2 (opencv-python-headless) not importable — "
            "cannot start RTSP puller.", camera_id
        )
        status.state = StreamState.error
        status.error_message = "cv2 not available"
        return

    status.state = StreamState.starting
    status.started_at = datetime.now(timezone.utc)
    backoff_s = 2.0        # initial reconnect wait
    sample_count = 0       # samples since last DB write

    logger.info("Camera %s: stream thread started (%s)", camera_id, stream_url)

    while not stop_event.is_set():
        # ── Open capture ──────────────────────────────────────────────────────
        status.state = StreamState.reconnecting if status.reconnect_attempts > 0 else StreamState.starting
        cap = cv2.VideoCapture(stream_url)

        if not cap.isOpened():
            status.consecutive_errors += 1
            status.reconnect_attempts += 1
            status.error_message = f"cv2.VideoCapture could not open: {stream_url}"
            logger.warning(
                "Camera %s: could not open stream (attempt %d). Retrying in %.1fs",
                camera_id, status.reconnect_attempts, backoff_s,
            )
            if status.consecutive_errors >= max_errors:
                logger.error(
                    "Camera %s: %d consecutive failures — giving up.",
                    camera_id, max_errors,
                )
                status.state = StreamState.error
                return
            _interruptible_sleep(backoff_s, stop_event)
            backoff_s = min(backoff_s * 2, max_backoff)
            continue

        # Opened successfully — reset backoff
        backoff_s = 2.0
        status.state = StreamState.running
        status.error_message = None
        logger.info("Camera %s: stream opened successfully", camera_id)

        last_sample_time = 0.0  # epoch seconds of last inference

        # ── Frame read loop ───────────────────────────────────────────────────
        while not stop_event.is_set():
            ret, frame = cap.read()

            if not ret or frame is None:
                status.consecutive_errors += 1
                logger.warning(
                    "Camera %s: frame read failed (%d consecutive).",
                    camera_id, status.consecutive_errors,
                )
                if status.consecutive_errors >= max_errors:
                    logger.error(
                        "Camera %s: %d consecutive frame errors — reconnecting.",
                        camera_id, status.consecutive_errors,
                    )
                    break  # break inner loop → outer loop will reconnect

                # Brief pause before next read attempt — avoid busy-spin on
                # a degraded stream without blocking the stop_event check
                _interruptible_sleep(0.5, stop_event)
                continue

            # ── Throttle — only infer every sample_interval_s ────────────────
            now_s = time.monotonic()
            if now_s - last_sample_time < sample_interval:
                # Read and discard — keeps the decoder buffer from backing up
                # (VideoCapture buffers frames internally; if we don't read
                # them the buffer fills and we'd get stale frames on next read)
                continue

            last_sample_time = now_s
            status.consecutive_errors = 0  # successful frame resets error counter
            status.last_frame_at = datetime.now(timezone.utc)
            status.frames_processed += 1

            # ── Inference ─────────────────────────────────────────────────────
            try:
                result = detector.detect_array(frame, annotate=False)
            except Exception as exc:
                logger.warning("Camera %s: inference error: %s", camera_id, exc)
                continue

            status.last_person_count = result.person_count
            sample_count += 1

            # ── DB work — own session per write cycle ─────────────────────────
            if sample_count >= persist_every:
                sample_count = 0
                db = SessionLocal()
                try:
                    # Persist DetectionRecord
                    record = DetectionRecord(
                        organization_id=org_id,
                        camera_id=camera_id,
                        source="rtsp",
                        person_count=result.person_count,
                        unique_people=result.person_count,
                        avg_confidence=result.avg_confidence,
                    )
                    db.add(record)
                    db.commit()

                    # Alerts
                    evaluate_alerts(
                        db,
                        organization_id=org_id,
                        camera_id=camera_id,
                        person_count=result.person_count,
                    )

                    # Heatmap
                    if result.detections:
                        try:
                            update_heatmap(
                                db,
                                organization_id=org_id,
                                camera_id=camera_id,
                                bboxes=[d.bbox for d in result.detections],
                                frame_width=result.width or frame.shape[1],
                                frame_height=result.height or frame.shape[0],
                            )
                        except Exception as exc:
                            logger.warning("Camera %s: heatmap update failed: %s", camera_id, exc)

                    # Anomaly
                    try:
                        anomaly = check_anomaly(
                            db,
                            organization_id=org_id,
                            camera_id=camera_id,
                            person_count=result.person_count,
                        )
                        if anomaly.get("is_anomaly"):
                            notifier_dispatch(
                                db,
                                organization_id=org_id,
                                user_id=None,
                                title="Anomaly detected",
                                body=(
                                    f"Unusual crowd count {result.person_count} on camera {camera_id} "
                                    f"(z-score: {anomaly.get('z_score')}, "
                                    f"baseline mean: {anomaly.get('baseline_mean')})"
                                ),
                                channels=[NotificationChannel.inbox],
                                source_type="anomaly",
                                source_id=camera_id,
                            )
                    except Exception as exc:
                        logger.warning("Camera %s: anomaly check failed: %s", camera_id, exc)

                except Exception as exc:
                    logger.exception("Camera %s: DB write error: %s", camera_id, exc)
                    try:
                        db.rollback()
                    except Exception:
                        pass
                finally:
                    db.close()

        # ── Inner loop exited — release capture before reconnect ──────────────
        try:
            cap.release()
        except Exception:
            pass

        if stop_event.is_set():
            break

        # Reconnect with backoff
        status.reconnect_attempts += 1
        logger.info(
            "Camera %s: reconnecting in %.1fs (attempt %d)",
            camera_id, backoff_s, status.reconnect_attempts,
        )
        _interruptible_sleep(backoff_s, stop_event)
        backoff_s = min(backoff_s * 2, max_backoff)

    # ── Thread exiting ────────────────────────────────────────────────────────
    status.state = StreamState.stopped
    logger.info(
        "Camera %s: stream thread stopped (frames_processed=%d)",
        camera_id, status.frames_processed,
    )


def _interruptible_sleep(seconds: float, stop_event: threading.Event) -> None:
    """Sleep for ``seconds`` but wake immediately if stop_event is set.

    Polls in 0.25s increments so the thread reacts to stop requests
    within 250ms regardless of the requested sleep duration.
    """
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if stop_event.is_set():
            return
        time.sleep(min(0.25, deadline - time.monotonic()))


# ── Manager ───────────────────────────────────────────────────────────────────

class CameraStreamManager:
    """Process-singleton manager for all camera stream threads.

    Thread safety: all mutations to ``_threads``, ``_stop_events``,
    ``_statuses`` are protected by ``_lock``.
    """

    _instance: CameraStreamManager | None = None
    _instance_lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()
        self._threads: dict[int, threading.Thread]       = {}
        self._stop_events: dict[int, threading.Event]    = {}
        self._statuses: dict[int, StreamStatus]          = {}

    @classmethod
    def get(cls) -> CameraStreamManager:
        """Return the process-singleton instance."""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # ── Public API ────────────────────────────────────────────────────────────

    def start_camera(self, camera_id: int, org_id: int, stream_url: str) -> bool:
        """Start a stream thread for the given camera.

        Returns False if a thread is already running for this camera_id.
        Idempotent: calling start on an already-running camera is a no-op.
        """
        with self._lock:
            existing = self._threads.get(camera_id)
            if existing is not None and existing.is_alive():
                logger.debug("Camera %s: start_camera called but thread already alive", camera_id)
                return False

            stop_event = threading.Event()
            status = StreamStatus(camera_id=camera_id)
            thread = threading.Thread(
                target=_stream_worker,
                args=(camera_id, org_id, stream_url, stop_event, status),
                name=f"rtsp-cam-{camera_id}",
                daemon=True,  # die with the process — no orphaned threads on hard kill
            )

            self._stop_events[camera_id] = stop_event
            self._statuses[camera_id] = status
            self._threads[camera_id] = thread

        thread.start()
        logger.info("Camera %s: stream thread launched (org=%s url=%s)", camera_id, org_id, stream_url)
        return True

    def stop_camera(self, camera_id: int, timeout_s: float = 5.0) -> bool:
        """Signal a camera's thread to stop and wait up to timeout_s.

        Returns True if the thread was stopped, False if it wasn't running.
        """
        with self._lock:
            stop_event = self._stop_events.get(camera_id)
            thread = self._threads.get(camera_id)

        if stop_event is None or thread is None:
            return False

        stop_event.set()
        thread.join(timeout=timeout_s)

        if thread.is_alive():
            logger.warning(
                "Camera %s: thread did not stop within %.1fs — it will die when the process exits.",
                camera_id, timeout_s,
            )
        else:
            logger.info("Camera %s: stream thread stopped cleanly.", camera_id)

        with self._lock:
            self._threads.pop(camera_id, None)
            self._stop_events.pop(camera_id, None)
            if camera_id in self._statuses:
                self._statuses[camera_id].state = StreamState.stopped

        return True

    def restart_camera(self, camera_id: int, org_id: int, stream_url: str) -> None:
        """Stop (if running) then start a camera stream."""
        self.stop_camera(camera_id)
        self.start_camera(camera_id, org_id, stream_url)

    def is_running(self, camera_id: int) -> bool:
        with self._lock:
            t = self._threads.get(camera_id)
            return t is not None and t.is_alive()

    def get_status(self, camera_id: int) -> StreamStatus | None:
        return self._statuses.get(camera_id)

    def all_statuses(self) -> dict[int, StreamStatus]:
        return dict(self._statuses)

    def start_all_active_cameras(self) -> int:
        """Boot-time call: query DB and start threads for all active cameras with a stream_url.

        Returns the number of threads started.
        """
        # Local imports to avoid circular deps at module load time
        from ..db import SessionLocal
        from ..models import Camera

        db = SessionLocal()
        started = 0
        try:
            cameras = (
                db.query(Camera)
                .filter(Camera.is_active.is_(True), Camera.stream_url.isnot(None))
                .all()
            )
            for cam in cameras:
                if cam.stream_url:
                    ok = self.start_camera(cam.id, cam.organization_id, cam.stream_url)
                    if ok:
                        started += 1
        except Exception:
            logger.exception("start_all_active_cameras: DB query failed")
        finally:
            db.close()

        logger.info("CameraStreamManager: started %d camera thread(s) on boot", started)
        return started

    def stop_all(self, timeout_s: float = 5.0) -> None:
        """Stop all running camera threads (called on app shutdown)."""
        with self._lock:
            camera_ids = list(self._threads.keys())

        logger.info("CameraStreamManager: stopping %d thread(s)…", len(camera_ids))
        for cid in camera_ids:
            self.stop_camera(cid, timeout_s=timeout_s)
        logger.info("CameraStreamManager: all threads stopped.")
