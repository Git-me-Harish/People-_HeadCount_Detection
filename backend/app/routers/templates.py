"""Industry templates — list available verticals and apply one to an org."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import get_current_user
from ..models import Alert, Camera, IndustryTemplate, User, Vertical
from ..schemas.alert import AlertRead
from ..schemas.camera import CameraRead

router = APIRouter(prefix="/templates", tags=["templates"])
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Seed data — in a production Postgres setup this would live in an Alembic
# seed migration. Here we insert on-the-fly if the table is empty.
# ---------------------------------------------------------------------------

_SEED_TEMPLATES = [
    {
        "vertical": Vertical.religious,
        "name": "Temple / Religious Gathering",
        "description": "Prevent stampedes at high-density religious venues. Real-time density alerts and public darshan queue ETA.",
        "icon": "🛕",
        "default_cameras": [
            {"name": "Main Entrance", "location": "Gate 1"},
            {"name": "Queue Line", "location": "Waiting Area"},
            {"name": "Sanctum Approach", "location": "Inner Sanctum"},
        ],
        "default_alerts": [
            {"name": "Queue Overload", "threshold": 50},
            {"name": "Density Spike", "threshold": 200},
        ],
    },
    {
        "vertical": Vertical.transit,
        "name": "Public Transit / Smart City",
        "description": "Reduce platform crush, inform commuters with live occupancy data.",
        "icon": "🚉",
        "default_cameras": [
            {"name": "Platform A", "location": "Platform"},
            {"name": "Ticketing Hall", "location": "Entrance"},
        ],
        "default_alerts": [
            {"name": "Platform Overcrowd", "threshold": 100},
        ],
    },
    {
        "vertical": Vertical.retail,
        "name": "Retail / Shopping Mall",
        "description": "Footfall analytics, conversion tracking, staffing optimisation.",
        "icon": "🛍️",
        "default_cameras": [
            {"name": "Main Entrance", "location": "Entrance"},
            {"name": "Food Court", "location": "Level 2"},
        ],
        "default_alerts": [
            {"name": "Capacity Warning", "threshold": 500},
        ],
    },
    {
        "vertical": Vertical.hospital,
        "name": "Hospital / Clinic",
        "description": "Reduce OPD wait stress, manage ER surge with live waiting-room counts.",
        "icon": "🏥",
        "default_cameras": [
            {"name": "OPD Waiting Room", "location": "Ground Floor"},
            {"name": "Emergency Entrance", "location": "ER"},
        ],
        "default_alerts": [
            {"name": "ER Surge Alert", "threshold": 30},
            {"name": "OPD Overflow", "threshold": 60},
        ],
    },
    {
        "vertical": Vertical.education,
        "name": "Schools / Universities",
        "description": "Attendance tracking, library/lab utilisation, after-hours intrusion alerts.",
        "icon": "🎓",
        "default_cameras": [
            {"name": "Main Gate", "location": "Entrance"},
            {"name": "Library", "location": "Block A"},
        ],
        "default_alerts": [
            {"name": "After-Hours Intrusion", "threshold": 1},
        ],
    },
    {
        "vertical": Vertical.stadium,
        "name": "Stadiums / Events",
        "description": "Section-wise density, evacuation-trigger alerts, crowd safety.",
        "icon": "🏟️",
        "default_cameras": [
            {"name": "North Stand", "location": "Section N"},
            {"name": "South Stand", "location": "Section S"},
            {"name": "Main Exit", "location": "Exit Gate"},
        ],
        "default_alerts": [
            {"name": "Section Overcapacity", "threshold": 300},
            {"name": "Evacuation Trigger", "threshold": 1000},
        ],
    },
    {
        "vertical": Vertical.workplace,
        "name": "Workplaces / Smart Buildings",
        "description": "Energy savings, fire-code compliance, real-occupancy HVAC API.",
        "icon": "🏢",
        "default_cameras": [
            {"name": "Lobby", "location": "Ground Floor"},
            {"name": "Open Office", "location": "Floor 3"},
        ],
        "default_alerts": [
            {"name": "Fire-Code Limit", "threshold": 150},
        ],
    },
    {
        "vertical": Vertical.tourism,
        "name": "Tourism / Museums / Monuments",
        "description": "Visitor flow, queue-time transparency, capacity caps.",
        "icon": "🏛️",
        "default_cameras": [
            {"name": "Main Gate", "location": "Entrance"},
            {"name": "Exhibit Hall", "location": "Hall A"},
        ],
        "default_alerts": [
            {"name": "Capacity Cap", "threshold": 200},
        ],
    },
]


def _seed_templates(db: Session) -> None:
    """Insert seed templates if table is empty."""
    if db.query(IndustryTemplate).count() > 0:
        return
    for t in _SEED_TEMPLATES:
        tmpl = IndustryTemplate(
            vertical=t["vertical"],
            name=t["name"],
            description=t["description"],
            icon=t["icon"],
            default_cameras_json=json.dumps(t["default_cameras"]),
            default_alerts_json=json.dumps(t["default_alerts"]),
        )
        db.add(tmpl)
    db.commit()
    logger.info("Seeded %d industry templates", len(_SEED_TEMPLATES))


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class TemplateRead(BaseModel):
    id: int
    vertical: str
    name: str
    description: str
    icon: str
    default_cameras: list[dict]
    default_alerts: list[dict]

    model_config = {"from_attributes": True}


class ApplyTemplateResponse(BaseModel):
    cameras_created: list[CameraRead]
    alerts_created: list[AlertRead]
    message: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("", response_model=list[TemplateRead])
def list_templates(db: Session = Depends(get_db)) -> list[TemplateRead]:
    """List all industry vertical templates."""
    _seed_templates(db)
    templates = db.query(IndustryTemplate).order_by(IndustryTemplate.id).all()
    return [
        TemplateRead(
            id=t.id,
            vertical=t.vertical,
            name=t.name,
            description=t.description,
            icon=t.icon,
            default_cameras=json.loads(t.default_cameras_json),
            default_alerts=json.loads(t.default_alerts_json),
        )
        for t in templates
    ]


@router.post("/{vertical}/apply", response_model=ApplyTemplateResponse)
def apply_template(
    vertical: Vertical,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ApplyTemplateResponse:
    """Apply an industry template to the current user's organisation.

    Creates default cameras and alert rules. Idempotent — skips already-existing
    cameras/alerts with the same name.
    """
    _seed_templates(db)
    template = db.query(IndustryTemplate).filter(IndustryTemplate.vertical == vertical).first()
    if template is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Template not found")

    org_id = user.organization_id
    cameras_created: list[Camera] = []
    alerts_created: list[Alert] = []

    for cam_def in json.loads(template.default_cameras_json):
        existing = (
            db.query(Camera)
            .filter(Camera.organization_id == org_id, Camera.name == cam_def["name"])
            .first()
        )
        if existing is None:
            cam = Camera(
                organization_id=org_id,
                name=cam_def["name"],
                location=cam_def.get("location"),
            )
            db.add(cam)
            db.flush()
            cameras_created.append(cam)

    for alert_def in json.loads(template.default_alerts_json):
        existing = (
            db.query(Alert)
            .filter(Alert.organization_id == org_id, Alert.name == alert_def["name"])
            .first()
        )
        if existing is None:
            alert = Alert(
                organization_id=org_id,
                name=alert_def["name"],
                threshold=alert_def["threshold"],
            )
            db.add(alert)
            db.flush()
            alerts_created.append(alert)

    db.commit()
    for item in cameras_created + alerts_created:
        db.refresh(item)

    return ApplyTemplateResponse(
        cameras_created=[CameraRead.model_validate(c) for c in cameras_created],
        alerts_created=[AlertRead.model_validate(a) for a in alerts_created],
        message=f"Applied '{template.name}' template: {len(cameras_created)} cameras, {len(alerts_created)} alerts created.",
    )
