"""Public status page router — unauthenticated read-only crowd-density endpoints."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import get_current_user
from ..models import Camera, DetectionRecord, PublicPage, User

router = APIRouter(prefix="/public", tags=["public"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class PublicPageCreate(BaseModel):
    slug: str
    title: str
    description: str | None = None
    camera_ids: list[int] = []
    show_heatmap: bool = False
    brand_color: str = "#6366f1"


class PublicPageRead(BaseModel):
    slug: str
    title: str
    description: str | None
    camera_ids: list[int]
    show_heatmap: bool
    brand_color: str
    is_active: bool

    model_config = {"from_attributes": True}


class CameraLiveStatus(BaseModel):
    camera_id: int
    name: str
    location: str | None
    current_count: int
    last_updated: datetime | None
    status: str  # "live" | "idle" | "offline"


class PublicStatusResponse(BaseModel):
    slug: str
    title: str
    description: str | None
    brand_color: str
    cameras: list[CameraLiveStatus]
    generated_at: datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9\-]{2,118}[a-z0-9]$")


def _validate_slug(slug: str) -> None:
    if not _SLUG_RE.match(slug):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Slug must be 4-120 chars, lowercase alphanumeric + hyphens, no leading/trailing hyphens.",
        )


# ---------------------------------------------------------------------------
# Authenticated management endpoints
# ---------------------------------------------------------------------------


@router.post("/pages", response_model=PublicPageRead, status_code=status.HTTP_201_CREATED)
def create_public_page(
    payload: PublicPageCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> PublicPage:
    _validate_slug(payload.slug)
    existing = db.query(PublicPage).filter(PublicPage.slug == payload.slug).first()
    if existing:
        raise HTTPException(status_code=400, detail="Slug already taken")
    page = PublicPage(
        organization_id=user.organization_id,
        slug=payload.slug,
        title=payload.title,
        description=payload.description,
        camera_ids_json=json.dumps(payload.camera_ids),
        show_heatmap=payload.show_heatmap,
        brand_color=payload.brand_color,
    )
    db.add(page)
    db.commit()
    db.refresh(page)
    return _page_to_read(page)


@router.get("/pages/mine", response_model=PublicPageRead | None)
def get_my_public_page(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> PublicPageRead | None:
    page = db.query(PublicPage).filter(PublicPage.organization_id == user.organization_id).first()
    return _page_to_read(page) if page else None


@router.patch("/pages/mine", response_model=PublicPageRead)
def update_my_public_page(
    payload: PublicPageCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> PublicPageRead:
    page = db.query(PublicPage).filter(PublicPage.organization_id == user.organization_id).first()
    if page is None:
        raise HTTPException(status_code=404, detail="Public page not found")
    page.title = payload.title
    page.description = payload.description
    page.camera_ids_json = json.dumps(payload.camera_ids)
    page.show_heatmap = payload.show_heatmap
    page.brand_color = payload.brand_color
    db.commit()
    db.refresh(page)
    return _page_to_read(page)


# ---------------------------------------------------------------------------
# Public unauthenticated endpoint
# ---------------------------------------------------------------------------


@router.get("/{slug}", response_model=PublicStatusResponse)
def get_public_status(slug: str, db: Session = Depends(get_db)) -> PublicStatusResponse:
    """Unauthenticated endpoint — returns live crowd counts for a public page."""
    page = (
        db.query(PublicPage).filter(PublicPage.slug == slug, PublicPage.is_active.is_(True)).first()
    )
    if page is None:
        raise HTTPException(status_code=404, detail="Public page not found or inactive")

    camera_ids: list[int] = json.loads(page.camera_ids_json)
    cameras = db.query(Camera).filter(Camera.id.in_(camera_ids)).all()
    now = datetime.now(timezone.utc)

    camera_statuses: list[CameraLiveStatus] = []
    for cam in cameras:
        latest = (
            db.query(DetectionRecord)
            .filter(DetectionRecord.camera_id == cam.id)
            .order_by(DetectionRecord.created_at.desc())
            .first()
        )
        if latest is None:
            cam_status = "offline"
            count = 0
            last_updated = None
        else:
            age_seconds = (now - latest.created_at).total_seconds()
            cam_status = "live" if age_seconds < 120 else "idle"
            count = latest.person_count
            last_updated = latest.created_at

        camera_statuses.append(
            CameraLiveStatus(
                camera_id=cam.id,
                name=cam.name,
                location=cam.location,
                current_count=count,
                last_updated=last_updated,
                status=cam_status,
            )
        )

    return PublicStatusResponse(
        slug=page.slug,
        title=page.title,
        description=page.description,
        brand_color=page.brand_color,
        cameras=camera_statuses,
        generated_at=now,
    )


def _page_to_read(page: PublicPage) -> PublicPageRead:
    return PublicPageRead(
        slug=page.slug,
        title=page.title,
        description=page.description,
        camera_ids=json.loads(page.camera_ids_json),
        show_heatmap=page.show_heatmap,
        brand_color=page.brand_color,
        is_active=page.is_active,
    )
