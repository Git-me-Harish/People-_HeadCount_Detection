"""Reporter service — PDF report generation using ReportLab.

Generates daily/weekly summary PDFs for orgs. Falls back gracefully
if reportlab is not installed (returns None + logs warning).
"""

from __future__ import annotations

import io
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import func
from sqlalchemy.orm import Session

from ..models import Camera, DetectionRecord

logger = logging.getLogger(__name__)


def _try_import_reportlab():
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )

        return colors, A4, getSampleStyleSheet, cm, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError:
        return None


def generate_summary_pdf(
    db: Session,
    *,
    organization_id: int,
    org_name: str,
    days: int = 7,
) -> bytes | None:
    """Generate a summary PDF for the past N days.

    Returns raw PDF bytes or None if reportlab is unavailable.
    """
    rl = _try_import_reportlab()
    if rl is None:
        logger.warning("reportlab not installed — PDF generation skipped")
        return None

    colors, A4, getSampleStyleSheet, cm, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle = rl

    since = datetime.now(timezone.utc) - timedelta(days=days)

    # Aggregate per-camera stats
    rows = (
        db.query(
            Camera.name,
            func.count(DetectionRecord.id).label("samples"),
            func.max(DetectionRecord.person_count).label("peak"),
            func.avg(DetectionRecord.person_count).label("avg"),
        )
        .join(DetectionRecord, DetectionRecord.camera_id == Camera.id, isouter=True)
        .filter(
            Camera.organization_id == organization_id,
            DetectionRecord.created_at >= since,
        )
        .group_by(Camera.id, Camera.name)
        .all()
    )

    overall_peak = (
        db.query(func.max(DetectionRecord.person_count))
        .filter(
            DetectionRecord.organization_id == organization_id,
            DetectionRecord.created_at >= since,
        )
        .scalar()
        or 0
    )

    overall_avg = (
        db.query(func.avg(DetectionRecord.person_count))
        .filter(
            DetectionRecord.organization_id == organization_id,
            DetectionRecord.created_at >= since,
        )
        .scalar()
        or 0
    )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=2 * cm, leftMargin=2 * cm)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph(f"PeopleSense — {org_name}", styles["Title"]))
    story.append(Paragraph(f"Report Period: Last {days} days", styles["Normal"]))
    story.append(Spacer(1, 0.5 * cm))

    # KPI summary table
    kpi_data = [
        ["Metric", "Value"],
        ["Peak Head Count", str(overall_peak)],
        ["Average Head Count", f"{float(overall_avg):.1f}"],
        ["Report Generated", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")],
    ]
    kpi_table = Table(kpi_data, colWidths=[8 * cm, 8 * cm])
    kpi_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#6366f1")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f3ff")]),
            ]
        )
    )
    story.append(kpi_table)
    story.append(Spacer(1, 0.5 * cm))

    # Per-camera table
    if rows:
        story.append(Paragraph("Camera Breakdown", styles["Heading2"]))
        cam_data = [["Camera", "Samples", "Peak", "Avg"]] + [
            [r.name, str(r.samples or 0), str(r.peak or 0), f"{float(r.avg or 0):.1f}"]
            for r in rows
        ]
        cam_table = Table(cam_data, colWidths=[7 * cm, 3 * cm, 3 * cm, 3 * cm])
        cam_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#6366f1")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f3ff")]),
                ]
            )
        )
        story.append(cam_table)

    doc.build(story)
    return buf.getvalue()
