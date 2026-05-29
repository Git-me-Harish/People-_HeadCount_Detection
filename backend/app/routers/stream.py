"""WebSocket live-stream endpoint.

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
from ..models import DetectionRecord, User
from ..security import decode_access_token
from ..services.alerts import evaluate_alerts
from ..services.detector import get_detector

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
    token: str = Query(...),
    camera_id: int | None = Query(None),
    persist_every: int = Query(15, ge=1, le=300),
) -> None:
    db = SessionLocal()
    try:
        user = _authenticate(db, token)
        if user is None:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        await websocket.accept()
        detector = get_detector()
        frame_idx = 0

        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    data = json.loads(raw) if raw.startswith("{") else {"image": raw}
                except json.JSONDecodeError:
                    data = {"image": raw}
                b64 = data.get("image")
                conf = data.get("confidence")
                if not b64:
                    continue
                # strip data URL prefix if present
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
