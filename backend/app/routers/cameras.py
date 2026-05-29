"""Camera CRUD endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import get_current_user
from ..models import Camera, User
from ..schemas.camera import CameraCreate, CameraRead, CameraUpdate

router = APIRouter(prefix="/cameras", tags=["cameras"])


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
    cam = Camera(organization_id=user.organization_id, **payload.model_dump())
    db.add(cam)
    db.commit()
    db.refresh(cam)
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
    for k, v in payload.model_dump(exclude_unset=True).items():
        setattr(cam, k, v)
    db.commit()
    db.refresh(cam)
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
    db.delete(cam)
    db.commit()
