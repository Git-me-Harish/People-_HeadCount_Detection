"""Reports router — on-demand PDF report generation."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import get_current_user
from ..models import User
from ..services.reporter import generate_summary_pdf

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("/summary/pdf")
def download_summary_pdf(
    days: int = Query(7, ge=1, le=365),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Response:
    """Generate and stream a summary PDF for the last N days."""
    pdf_bytes = generate_summary_pdf(
        db,
        organization_id=user.organization_id,
        org_name=user.organization.name if user.organization else "Organisation",
        days=days,
    )
    if pdf_bytes is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PDF generation unavailable — reportlab not installed",
        )
    filename = f"peoplesense_report_{days}d.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
