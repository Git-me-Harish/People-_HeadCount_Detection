"""WebSocket live-stream endpoint.

Auth: token sent as first JSON message {"token": "..."} rather than URL
query param (avoids token in server logs and browser history).

Legacy ?token= query param still accepted for backward compatibility but
a deprecation warning is logged.

Client sends base64-encoded JPEG/PNG frames; server runs detection and
returns count + bounding boxes + annotated frame.
"""

from __future__ import annotations

import base64
import json
import logging

import numpy as np
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status
from sqlalchemy.orm import Session

from ..db import SessionLocal
from ..models import DetectionRecord, NotificationChannel, User
from ..security import decode_access_token
from ..services.alerts import evaluate_alerts
from ..services.anomaly import check_anomaly
from ..services.detector import get_detector
from ..services.heatmap import update_heatmap
from ..services.notifier import dispatch as notifier_dispatch

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stream", tags=["stream"])


def _authenticate(db: Session, token: str) -> User | None:
    try:
        payload = decode_access_token(token)
        user_id = int(payload.get("sub", 0))
    except (ValueError, TypeError):
        return None
    user = db.get(User, user_id)
    if user is None or not user.is_active:
        return None
    return user


@router.websocket("/ws")
async def stream_ws(
    websocket: WebSocket,
    token: str | None = Query(None),  # legacy — prefer first-message auth
    camera_id: int | None = Query(None),
    persist_every: int = Query(15, ge=1, le=300),
) -> None:
    db = SessionLocal()
    try:
        # Accept first so we can send auth errors over the socket
        await websocket.accept()
        detector = get_detector()
        frame_idx = 0
        user: User | None = None

        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    data = json.loads(raw) if raw.startswith("{") else {"image": raw}
                except json.JSONDecodeError:
                    data = {"image": raw}

                # ── Auth resolution ──────────────────────────────────────
                if user is None:
                    # Prefer token from first message payload
                    msg_token: str | None = data.get("token")
                    if msg_token:
                        user = _authenticate(db, msg_token)
                    elif token:
                        # Fallback to legacy query-param token
                        logger.warning(
                            "WS client using deprecated ?token= query param — "
                            "send token as first JSON message instead"
                        )
                        user = _authenticate(db, token)

                    if user is None:
                        await websocket.send_json({"error": "unauthorized"})
                        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                        return

                    # If the first message was auth-only (no image), continue to next frame
                    if not data.get("image"):
                        await websocket.send_json({"authenticated": True})
                        continue
                # ─────────────────────────────────────────────────────────

                b64 = data.get("image")
                conf = data.get("confidence")
                if not b64:
                    continue
                # Strip data URL prefix if present
                if "," in b64 and b64.startswith("data:"):
                    b64 = b64.split(",", 1)[1]
                try:
                    img_bytes = base64.b64decode(b64)
                except Exception:
                    await websocket.send_json({"error": "invalid base64 frame"})
                    continue

                result = detector.detect_image_bytes(img_bytes, conf=conf, annotate=True)

                frame_idx += 1
                if frame_idx % persist_every == 0:
                    record = DetectionRecord(
                        organization_id=user.organization_id,
                        camera_id=camera_id,
                        source="stream",
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

                    # Heatmap — only when camera is known
                    if camera_id is not None and result.detections:
                        try:
                            update_heatmap(
                                db,
                                organization_id=user.organization_id,
                                camera_id=camera_id,
                                bboxes=[d.bbox for d in result.detections],
                                frame_width=result.width or 640,
                                frame_height=result.height or 480,
                            )
                        except Exception:
                            pass  # non-fatal

                    # Anomaly check
                    if camera_id is not None:
                        anomaly = check_anomaly(
                            db,
                            organization_id=user.organization_id,
                            camera_id=camera_id,
                            person_count=result.person_count,
                        )
                        if anomaly.get("is_anomaly"):
                            notifier_dispatch(
                                db,
                                organization_id=user.organization_id,
                                user_id=None,
                                title="Anomaly detected",
                                body=(
                                    f"Unusual crowd count {result.person_count} on camera {camera_id} "
                                    f"(z-score: {anomaly.get('z_score')}, baseline mean: {anomaly.get('baseline_mean')})"
                                ),
                                channels=[NotificationChannel.inbox],
                                source_type="anomaly",
                                source_id=camera_id,
                            )

                await websocket.send_json(
                    {
                        "person_count": result.person_count,
                        "avg_confidence": result.avg_confidence,
                        "detections": [
                            {
                                "class_name": d.class_name,
                                "confidence": d.confidence,
                                "bbox": d.bbox.model_dump(),
                            }
                            for d in result.detections
                        ],
                        "annotated_image_b64": result.annotated_image_b64,
                    }
                )
        except WebSocketDisconnect:
            logger.info("Stream client disconnected")
    finally:
        db.close()


# Silence unused import warning in some linters
_ = np