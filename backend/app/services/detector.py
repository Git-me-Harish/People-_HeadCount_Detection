"""YOLOv8-backed people detector wrapped behind a clean interface.

The detector is lazy-loaded so we don't pay the model-load cost during tests
or simple health-check requests. It also degrades gracefully if `ultralytics`
isn't installed — the API still responds, just with zero detections — which
keeps the CI test suite hermetic.
"""

from __future__ import annotations

import base64
import logging
import threading
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

from ..config import get_settings
from ..schemas.common import BoundingBox, Detection, DetectionResult

logger = logging.getLogger(__name__)

_COCO_NAMES_FALLBACK = {0: "person", 1: "bicycle", 2: "car"}


class Detector:
    """Singleton YOLOv8 detector."""

    _instance: Detector | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.settings = get_settings()
        self._model: Any | None = None
        self._cv2: Any | None = None
        self._model_load_error: str | None = None

    @classmethod
    def instance(cls) -> Detector:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # ------------------------------------------------------------------ loading
    def _load(self) -> None:
        if self._model is not None or self._model_load_error is not None:
            return
        if not self.settings.enable_detector:
            self._model_load_error = "Detector disabled via configuration"
            return
        try:
            from ultralytics import YOLO  # type: ignore

            logger.info("Loading YOLO model from %s", self.settings.yolo_model_path)
            self._model = YOLO(self.settings.yolo_model_path)
        except Exception as exc:  # pragma: no cover - environment-dependent
            self._model_load_error = f"Failed to load YOLO model: {exc}"
            logger.warning(self._model_load_error)

    def _get_cv2(self):
        if self._cv2 is None:
            try:
                import cv2  # type: ignore

                self._cv2 = cv2
            except Exception as exc:  # pragma: no cover
                logger.warning("cv2 unavailable: %s", exc)
                self._cv2 = False
        return self._cv2 if self._cv2 is not False else None

    @property
    def is_ready(self) -> bool:
        self._load()
        return self._model is not None

    @property
    def status(self) -> dict[str, Any]:
        self._load()
        return {
            "ready": self._model is not None,
            "model_path": self.settings.yolo_model_path,
            "error": self._model_load_error,
            "confidence": self.settings.yolo_confidence,
        }

    # ------------------------------------------------------------------- detect
    def detect_array(
        self,
        frame_bgr: np.ndarray,
        *,
        conf: float | None = None,
        annotate: bool = True,
    ) -> DetectionResult:
        """Run detection on an OpenCV-style BGR ndarray.

        Returns a DetectionResult with optional base64 annotated PNG.
        """
        self._load()
        conf = conf if conf is not None else self.settings.yolo_confidence
        height, width = (int(frame_bgr.shape[0]), int(frame_bgr.shape[1]))

        if self._model is None:
            return DetectionResult(
                person_count=0,
                detections=[],
                annotated_image_b64=None,
                avg_confidence=None,
                width=width,
                height=height,
            )

        results = self._model.predict(
            source=frame_bgr,
            conf=conf,
            iou=self.settings.yolo_iou,
            verbose=False,
        )
        result = results[0]
        detections: list[Detection] = []

        names = getattr(result, "names", None) or _COCO_NAMES_FALLBACK

        if result.boxes is not None:
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, cid in zip(xyxy, confs, cls_ids, strict=False):
                if cid != self.settings.person_class_id:
                    continue
                detections.append(
                    Detection(
                        class_id=int(cid),
                        class_name=str(names.get(int(cid), "unknown")),
                        confidence=float(c),
                        bbox=BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                    )
                )

        annotated_b64: str | None = None
        if annotate:
            annotated_b64 = self._annotate(frame_bgr, detections)

        avg_conf = float(np.mean([d.confidence for d in detections])) if detections else None
        return DetectionResult(
            person_count=len(detections),
            detections=detections,
            annotated_image_b64=annotated_b64,
            avg_confidence=avg_conf,
            width=width,
            height=height,
        )

    def detect_image_bytes(
        self, data: bytes, *, conf: float | None = None, annotate: bool = True
    ) -> DetectionResult:
        cv2 = self._get_cv2()
        if cv2 is None:
            image = np.array(Image.open(BytesIO(data)).convert("RGB"))[:, :, ::-1].copy()
        else:
            arr = np.frombuffer(data, dtype=np.uint8)
            image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if image is None:
                # Fallback to Pillow
                image = np.array(Image.open(BytesIO(data)).convert("RGB"))[:, :, ::-1].copy()
        return self.detect_array(image, conf=conf, annotate=annotate)

    # ---------------------------------------------------------------- annotate
    def _annotate(self, frame_bgr: np.ndarray, detections: list[Detection]) -> str:
        """Draw boxes + count label and return base64-encoded PNG."""
        cv2 = self._get_cv2()
        out = frame_bgr.copy()
        if cv2 is not None:
            for det in detections:
                x1, y1, x2, y2 = (
                    int(det.bbox.x1),
                    int(det.bbox.y1),
                    int(det.bbox.x2),
                    int(det.bbox.y2),
                )
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{det.class_name} {det.confidence:.2f}"
                cv2.putText(
                    out,
                    label,
                    (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            cv2.rectangle(out, (0, 0), (260, 40), (0, 0, 0), -1)
            cv2.putText(
                out,
                f"People: {len(detections)}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            ok, buf = cv2.imencode(".png", out)
            if ok:
                return base64.b64encode(buf.tobytes()).decode("ascii")

        # Pillow fallback (no cv2 available)
        img = Image.fromarray(out[:, :, ::-1])
        buf_io = BytesIO()
        img.save(buf_io, format="PNG")
        return base64.b64encode(buf_io.getvalue()).decode("ascii")


def get_detector() -> Detector:
    return Detector.instance()
