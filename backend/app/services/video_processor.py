"""Async video processing service.

Reads frames from a video file, runs detection on (every Nth) frame, writes an
annotated MP4, and emits a JSON summary. Used by the background job worker.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..config import get_settings
from ..schemas.common import JobSummary
from .detector import get_detector

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, int, int], None]


def process_video(
    input_path: Path,
    output_path: Path,
    progress_cb: ProgressCallback | None = None,
    *,
    conf: float | None = None,
) -> JobSummary:
    """Process a video and return a summary. Annotated video written to output_path."""
    settings = get_settings()
    detector = get_detector()

    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for video processing") from exc

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    stride = max(1, int(settings.video_frame_stride))
    per_frame_counts: list[int] = []
    frame_idx = 0
    processed = 0
    started = time.time()
    last_annotated = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % stride == 0:
                result = detector.detect_array(frame, conf=conf, annotate=True)
                last_annotated = _decode_annotated(result.annotated_image_b64, frame)
                per_frame_counts.append(result.person_count)
                processed += 1
            else:
                # Reuse the last annotated frame to keep output framerate constant
                last_annotated = last_annotated if last_annotated is not None else frame

            writer.write(last_annotated if last_annotated is not None else frame)
            frame_idx += 1

            if progress_cb and (frame_idx % 10 == 0 or frame_idx == total_frames):
                pct = (frame_idx / total_frames) if total_frames > 0 else 0.0
                progress_cb(min(pct, 0.99), frame_idx, total_frames)
    finally:
        cap.release()
        writer.release()

    duration = time.time() - started
    counts = np.array(per_frame_counts, dtype=int) if per_frame_counts else np.zeros(0, dtype=int)
    summary = JobSummary(
        frames_processed=processed,
        total_frames=total_frames,
        average_person_count=float(counts.mean()) if counts.size else 0.0,
        peak_person_count=int(counts.max()) if counts.size else 0,
        unique_people=int(counts.max()) if counts.size else 0,
        per_frame=counts.tolist(),
        duration_seconds=duration,
    )
    if progress_cb:
        progress_cb(1.0, frame_idx, total_frames)
    return summary


def summary_to_json(summary: JobSummary) -> str:
    return json.dumps(summary.model_dump())


def _decode_annotated(b64: str | None, fallback):
    if not b64:
        return fallback
    try:
        import base64

        import cv2  # type: ignore

        data = base64.b64decode(b64)
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img if img is not None else fallback
    except Exception:  # pragma: no cover
        return fallback
