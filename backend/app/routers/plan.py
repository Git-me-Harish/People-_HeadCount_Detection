"""Plan and usage router — tier info and current usage counters."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import get_current_user
from ..models import Plan, PlanTier, UsageCounter, User

router = APIRouter(prefix="/plan", tags=["plan"])

_FREE_LIMITS = Plan(
    id=0,
    tier=PlanTier.free,
    display_name="Free",
    max_cameras=3,
    max_alerts=5,
    max_api_tokens=2,
    max_frames_per_month=10_000,
    retention_days=30,
    can_export_pdf=False,
    can_use_public_page=False,
    price_usd_monthly=0,
)


class PlanRead(BaseModel):
    tier: str
    display_name: str
    max_cameras: int
    max_alerts: int
    max_api_tokens: int
    max_frames_per_month: int
    retention_days: int
    can_export_pdf: bool
    can_use_public_page: bool
    price_usd_monthly: int


class UsageRead(BaseModel):
    plan_tier: str
    cameras_used: int
    frames_processed_month: int
    alerts_sent_month: int
    period_start: str


class PlanAndUsageResponse(BaseModel):
    plan: PlanRead
    usage: UsageRead


def _get_or_create_usage(db: Session, org_id: int) -> UsageCounter:
    usage = db.query(UsageCounter).filter(UsageCounter.organization_id == org_id).first()
    if usage is None:
        usage = UsageCounter(organization_id=org_id, plan_tier=PlanTier.free)
        db.add(usage)
        db.commit()
        db.refresh(usage)
    return usage


@router.get("", response_model=PlanAndUsageResponse)
def get_plan(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> PlanAndUsageResponse:
    usage = _get_or_create_usage(db, user.organization_id)

    # Look up plan; fall back to free limits if not seeded
    plan_row = db.query(Plan).filter(Plan.tier == usage.plan_tier).first()
    plan = plan_row or _FREE_LIMITS

    return PlanAndUsageResponse(
        plan=PlanRead(
            tier=plan.tier,
            display_name=plan.display_name,
            max_cameras=plan.max_cameras,
            max_alerts=plan.max_alerts,
            max_api_tokens=plan.max_api_tokens,
            max_frames_per_month=plan.max_frames_per_month,
            retention_days=plan.retention_days,
            can_export_pdf=plan.can_export_pdf,
            can_use_public_page=plan.can_use_public_page,
            price_usd_monthly=plan.price_usd_monthly,
        ),
        usage=UsageRead(
            plan_tier=usage.plan_tier,
            cameras_used=usage.cameras_used,
            frames_processed_month=usage.frames_processed_month,
            alerts_sent_month=usage.alerts_sent_month,
            period_start=usage.period_start.isoformat(),
        ),
    )
