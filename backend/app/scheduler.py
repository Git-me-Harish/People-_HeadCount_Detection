"""APScheduler background scheduler."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from .db import SessionLocal
from .models import DetectionRecord, Plan, UsageCounter

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None


def purge_old_detection_records() -> None:
    """Delete DetectionRecords beyond each org's plan retention window.

    Queries ``usage_counters`` for plan tier, joins ``plans`` for
    ``retention_days``, then bulk-deletes rows older than the cutoff.
    Falls back to ``Settings.default_retention_days`` when no plan row
    is found for an org.
    """
    from .config import get_settings  # local import — avoids circular at module load

    settings = get_settings()
    db = SessionLocal()
    deleted_total = 0
    try:
        # Load all usage counters to get per-org plan tiers
        counters = db.query(UsageCounter).all()
        if not counters:
            logger.debug("purge_old_detection_records: no orgs found, skipping")
            return

        # Cache plan rows to avoid N+1
        plans: dict[str, Plan] = {
            p.tier: p for p in db.query(Plan).all()
        }

        now = datetime.now(timezone.utc)
        for counter in counters:
            plan = plans.get(counter.plan_tier)
            retention_days = (
                plan.retention_days
                if plan is not None
                else settings.default_retention_days
            )
            cutoff = now - timedelta(days=retention_days)

            deleted = (
                db.query(DetectionRecord)
                .filter(
                    DetectionRecord.organization_id == counter.organization_id,
                    DetectionRecord.created_at < cutoff,
                )
                .delete(synchronize_session=False)
            )
            if deleted:
                logger.info(
                    "purge_old_detection_records: org=%s plan=%s retention=%dd → deleted %d rows",
                    counter.organization_id,
                    counter.plan_tier,
                    retention_days,
                    deleted,
                )
            deleted_total += deleted

        db.commit()
        logger.info(
            "purge_old_detection_records: completed — %d rows deleted across %d orgs",
            deleted_total,
            len(counters),
        )
    except Exception:
        logger.exception("purge_old_detection_records: failed")
        db.rollback()
    finally:
        db.close()


def start_scheduler() -> None:
    """Initialise and start the background scheduler."""
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        logger.warning("Scheduler already running — skipping start")
        return

    _scheduler = BackgroundScheduler(timezone="UTC")

    # Nightly at 02:00 UTC — low-traffic window; jitter avoids thundering-herd
    # on multi-worker deployments (each process registers independently but
    # APScheduler's jobstore can be pointed at a shared DB to deduplicate).
    _scheduler.add_job(
        purge_old_detection_records,
        trigger=CronTrigger(hour=2, minute=0, timezone="UTC"),
        id="purge_detection_records",
        name="Nightly DetectionRecord purge",
        replace_existing=True,
        misfire_grace_time=3600,  # tolerate up to 1h startup delay
    )

    _scheduler.start()
    logger.info("Background scheduler started — nightly purge at 02:00 UTC")


def stop_scheduler() -> None:
    """Gracefully shut down the scheduler on app teardown."""
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Background scheduler stopped")
    _scheduler = None