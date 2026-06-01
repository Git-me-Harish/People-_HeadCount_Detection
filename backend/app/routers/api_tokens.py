"""API tokens router — customer-managed programmatic access tokens."""

from __future__ import annotations

import secrets
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import get_current_user
from ..models import APIToken, User
from ..security import hash_password  # bcrypt — reuse for token secret

router = APIRouter(prefix="/api-tokens", tags=["api-tokens"])

_PREFIX_LEN = 8  # ps_ + 8 chars = "ps_xxxxxxxx"
_SECRET_LEN = 32


def _generate_token() -> tuple[str, str, str]:
    """Return (prefix, secret, full_token). prefix stored plain; secret hashed."""
    prefix_raw = secrets.token_urlsafe(_PREFIX_LEN)[:_PREFIX_LEN]
    prefix = f"ps_{prefix_raw}"
    secret = secrets.token_urlsafe(_SECRET_LEN)
    full_token = f"{prefix}.{secret}"
    return prefix, secret, full_token


class APITokenCreate(BaseModel):
    name: str
    scopes: str = "read:all"
    expires_at: datetime | None = None


class APITokenRead(BaseModel):
    id: int
    name: str
    prefix: str
    scopes: str
    is_active: bool
    last_used_at: datetime | None
    expires_at: datetime | None
    created_at: datetime

    model_config = {"from_attributes": True}


class APITokenCreated(APITokenRead):
    """Returned only on creation — includes the plaintext full_token once."""
    full_token: str


@router.get("", response_model=list[APITokenRead])
def list_tokens(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[APIToken]:
    return (
        db.query(APIToken)
        .filter(APIToken.organization_id == user.organization_id)
        .order_by(APIToken.created_at.desc())
        .all()
    )


@router.post("", response_model=APITokenCreated, status_code=status.HTTP_201_CREATED)
def create_token(
    payload: APITokenCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> APITokenCreated:
    """Create a new API token. The full token is returned ONCE — store it securely."""
    prefix, secret, full_token = _generate_token()
    token = APIToken(
        organization_id=user.organization_id,
        created_by_id=user.id,
        name=payload.name,
        prefix=prefix,
        hashed_secret=hash_password(secret),
        scopes=payload.scopes,
        expires_at=payload.expires_at,
    )
    db.add(token)
    db.commit()
    db.refresh(token)
    return APITokenCreated(
        id=token.id,
        name=token.name,
        prefix=token.prefix,
        scopes=token.scopes,
        is_active=token.is_active,
        last_used_at=token.last_used_at,
        expires_at=token.expires_at,
        created_at=token.created_at,
        full_token=full_token,
    )


@router.delete("/{token_id}", status_code=status.HTTP_204_NO_CONTENT)
def revoke_token(
    token_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> None:
    token = db.query(APIToken).filter(
        APIToken.id == token_id,
        APIToken.organization_id == user.organization_id,
    ).first()
    if token is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Token not found")
    db.delete(token)
    db.commit()
