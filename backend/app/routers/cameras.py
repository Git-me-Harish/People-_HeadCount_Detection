"""Camera CRUD endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import get_current_user
from ..models import Camera, Plan, UsageCounter, User
from ..schemas.camera import CameraCreate, CameraRead, CameraUpdate
from ..services.stream_manager import CameraStreamManager, StreamState, StreamStatus

router = APIRouter(prefix="/cameras", tags=["cameras"])


# Read-only schema for stream status responses:


class StreamStatusRead(BaseModel):
    camera_id: int
    state: StreamState
    last_frame_at: str | None
    last_person_count: int
    frames_processed: int
    consecutive_errors: int
    reconnect_attempts: int
    error_message: str | None
    started_at: str | None

    @classmethod
    def from_status(cls, s: StreamStatus) -> StreamStatusRead:
        return cls(
            camera_id=s.camera_id,
            state=s.state,
            last_frame_at=s.last_frame_at.isoformat() if s.last_frame_at else None,
            last_person_count=s.last_person_count,
            frames_processed=s.frames_processed,
            consecutive_errors=s.consecutive_errors,
            reconnect_attempts=s.reconnect_attempts,
            error_message=s.error_message,
            started_at=s.started_at.isoformat() if s.started_at else None,
        )


@router.get("", response_model=list[CameraRead])
def list_cameras(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[Camera]:
    return (
        db.query(Camera)
        .filter(Camera.organization_id == user.organization_id)
        .order_by(Camera.created_at.desc())
        .all()
    )


@router.post("", response_model=CameraRead, status_code=status.HTTP_201_CREATED)
def create_camera(
    payload: CameraCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Camera:
    # Plan limit enforcement
    usage = (
        db.query(UsageCounter).filter(UsageCounter.organization_id == user.organization_id).first()
    )
    if usage is not None:
        plan = db.query(Plan).filter(Plan.tier == usage.plan_tier).first()
        if plan is not None:
            current_count = (
                db.query(Camera)
                .filter(Camera.organization_id == user.organization_id, Camera.is_active.is_(True))
                .count()
            )
            if current_count >= plan.max_cameras:
                raise HTTPException(
                    status_code=status.HTTP_402_PAYMENT_REQUIRED,
                    detail=f"Camera limit reached for {plan.tier} plan ({plan.max_cameras} max). Upgrade to add more.",
                )

    cam = Camera(organization_id=user.organization_id, **payload.model_dump())
    db.add(cam)
    db.commit()
    db.refresh(cam)

    # Auto-start stream thread if a URL was provided
    if cam.stream_url and cam.is_active:
        CameraStreamManager.get().start_camera(cam.id, cam.organization_id, cam.stream_url)

    return cam


@router.get("/{camera_id}", response_model=CameraRead)
def get_camera(
    camera_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Camera:
    cam = db.get(Camera, camera_id)
    if cam is None or cam.organization_id != user.organization_id:
        raise HTTPException(status_code=404, detail="Camera not found")
    return cam


@router.patch("/{camera_id}", response_model=CameraRead)
def update_camera(
    camera_id: int,
    payload: CameraUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Camera:
    cam = db.get(Camera, camera_id)
    if cam is None or cam.organization_id != user.organization_id:
        raise HTTPException(status_code=404, detail="Camera not found")

    updated = payload.model_dump(exclude_unset=True)
    for k, v in updated.items():
        setattr(cam, k, v)
    db.commit()
    db.refresh(cam)

    manager = CameraStreamManager.get()
    stream_changed = "stream_url" in updated or "is_active" in updated

    if stream_changed:
        if cam.is_active and cam.stream_url:
            # URL or active flag changed — restart with new URL
            manager.restart_camera(cam.id, cam.organization_id, cam.stream_url)
        else:
            # Camera deactivated or URL cleared — stop thread
            manager.stop_camera(cam.id)

    return cam


@router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_camera(
    camera_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> None:
    cam = db.get(Camera, camera_id)
    if cam is None or cam.organization_id != user.organization_id:
        raise HTTPException(status_code=404, detail="Camera not found")
    # Stop stream thread before deleting the camera record
    CameraStreamManager.get().stop_camera(camera_id)
    db.delete(cam)
    db.commit()


# ── Stream management endpoints ───────────────────────────────────────────────


@router.get("/{camera_id}/stream/status", response_model=StreamStatusRead)
def get_stream_status(
    camera_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> StreamStatusRead:
    """Return the live status of the camera's RTSP puller thread.

    Useful for ops dashboards: shows state (running / reconnecting / error),
    last frame timestamp, person count, consecutive error count, etc.
    """
    cam = db.get(Camera, camera_id)
    if cam is None or cam.organization_id != user.organization_id:
        raise HTTPException(status_code=404, detail="Camera not found")

    status_obj = CameraStreamManager.get().get_status(camera_id)
    if status_obj is None:
        # No thread ever started for this camera (e.g. no stream_url)
        return StreamStatusRead(
            camera_id=camera_id,
            state=StreamState.idle,
            last_frame_at=None,
            last_person_count=0,
            frames_processed=0,
            consecutive_errors=0,
            reconnect_attempts=0,
            error_message=None if cam.stream_url else "No stream_url configured",
            started_at=None,
        )
    return StreamStatusRead.from_status(status_obj)


@router.post("/{camera_id}/stream/start", status_code=status.HTTP_200_OK)
def start_stream(
    camera_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    """Manually start (or restart) the RTSP puller thread for a camera."""
    cam = db.get(Camera, camera_id)
    if cam is None or cam.organization_id != user.organization_id:
        raise HTTPException(status_code=404, detail="Camera not found")
    if not cam.stream_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Camera has no stream_url configured"
        )
    if not cam.is_active:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Camera is not active")

    CameraStreamManager.get().restart_camera(cam.id, cam.organization_id, cam.stream_url)
    return {"camera_id": camera_id, "action": "started"}


@router.post("/{camera_id}/stream/stop", status_code=status.HTTP_200_OK)
def stop_stream(
    camera_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    """Manually stop the RTSP puller thread for a camera without deleting it."""
    cam = db.get(Camera, camera_id)
    if cam is None or cam.organization_id != user.organization_id:
        raise HTTPException(status_code=404, detail="Camera not found")

    stopped = CameraStreamManager.get().stop_camera(camera_id)
    return {"camera_id": camera_id, "action": "stopped", "was_running": stopped}
