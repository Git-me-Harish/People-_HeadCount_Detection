"""Plan and usage router — tier info and current usage counters."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
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


# Upgrade:

_UPGRADEABLE_TIERS: dict[str, list[str]] = {
    "free": ["pro", "enterprise"],
    "pro":  ["enterprise"],
}

# Maps tier → friendly display details used by the frontend upgrade modal.
# In a real billing integration these would come from Stripe product catalog.
_TIER_DISPLAY = {
    "pro": {
        "display_name": "Pro",
        "price_usd_monthly": 2900,  # cents
        "cameras": 25,
        "alerts": 50,
        "highlights": ["PDF & CSV exports", "Public status page", "Slack / Teams alerts", "90-day retention", "5 API tokens"],
    },
    "enterprise": {
        "display_name": "Enterprise",
        "price_usd_monthly": None,  # custom / contact sales
        "cameras": -1,
        "alerts": -1,
        "highlights": ["Unlimited cameras", "Dedicated infrastructure", "SLA support", "SSO / SAML", "Custom retention", "On-premise"],
    },
}


class UpgradeableTierInfo(BaseModel):
    tier: str
    display_name: str
    price_usd_monthly: int | None
    cameras: int
    alerts: int
    highlights: list[str]


class UpgradeOptions(BaseModel):
    current_tier: str
    available_upgrades: list[UpgradeableTierInfo]


class UpgradeRequest(BaseModel):
    target_tier: str


class UpgradeResponse(BaseModel):
    success: bool
    previous_tier: str
    new_tier: str
    message: str


@router.get("/upgrade-options", response_model=UpgradeOptions)
def get_upgrade_options(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> UpgradeOptions:
    """Return available upgrade tiers for the current org."""
    usage = _get_or_create_usage(db, user.organization_id)
    available = _UPGRADEABLE_TIERS.get(usage.plan_tier, [])
    return UpgradeOptions(
        current_tier=usage.plan_tier,
        available_upgrades=[
            UpgradeableTierInfo(tier=t, **_TIER_DISPLAY[t])  # type: ignore[arg-type]
            for t in available
            if t in _TIER_DISPLAY
        ],
    )


@router.post("/upgrade", response_model=UpgradeResponse)
def upgrade_plan(
    payload: UpgradeRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> UpgradeResponse:
    """Upgrade the organisation's plan tier.

    In production this endpoint would initiate a Stripe Checkout session
    and return a redirect URL. Here we apply the upgrade directly so the
    full flow works end-to-end without a payment provider configured.

    Only admins can upgrade the plan.
    """
    from ..models import UserRole  # local to avoid circular at module level

    if user.role != UserRole.admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only organisation admins can change the plan",
        )

    target = payload.target_tier.lower()
    if target not in _TIER_DISPLAY:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown tier '{target}'. Valid options: {', '.join(_TIER_DISPLAY)}",
        )

    usage = _get_or_create_usage(db, user.organization_id)
    allowed = _UPGRADEABLE_TIERS.get(usage.plan_tier, [])

    if target not in allowed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot upgrade from '{usage.plan_tier}' to '{target}'. "
                   f"Allowed: {allowed or 'none (already at highest tier)'}",
        )

    previous = usage.plan_tier
    usage.plan_tier = target  # type: ignore[assignment]
    db.commit()

    return UpgradeResponse(
        success=True,
        previous_tier=previous,
        new_tier=target,
        message=f"Plan upgraded from {previous} to {target} successfully.",
    )
