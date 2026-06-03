"""Authentication endpoints."""

from __future__ import annotations

import re
import secrets
import string

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import get_current_user
from ..models import NotificationChannel, Organization, User, UserRole
from ..schemas.auth import LoginRequest, RegisterRequest, Token
from ..schemas.user import UserRead
from ..security import create_access_token, hash_password, verify_password
from ..services.notifier import dispatch as notifier_dispatch

router = APIRouter(prefix="/auth", tags=["auth"])


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "org"
    return slug[:120]


def _ensure_unique_slug(db: Session, base: str) -> str:
    slug = base
    suffix = 2
    while db.query(Organization).filter(Organization.slug == slug).first() is not None:
        slug = f"{base}-{suffix}"
        suffix += 1
    return slug


@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, db: Session = Depends(get_db)) -> Token:
    if db.query(User).filter(User.email == payload.email.lower()).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    org = Organization(
        name=payload.organization_name,
        slug=_ensure_unique_slug(db, _slugify(payload.organization_name)),
    )
    db.add(org)
    db.flush()
    user = User(
        email=payload.email.lower(),
        full_name=payload.full_name,
        hashed_password=hash_password(payload.password),
        organization_id=org.id,
        role=UserRole.admin,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return Token(access_token=create_access_token(str(user.id)))


@router.post("/login", response_model=Token)
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> Token:
    user = db.query(User).filter(User.email == payload.email.lower()).first()
    if user is None or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Account disabled")
    return Token(access_token=create_access_token(str(user.id)))


@router.post("/token", response_model=Token, include_in_schema=False)
def login_form(
    form: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
) -> Token:
    """OAuth2-password-flow compatible token endpoint for Swagger UI."""
    return login(LoginRequest(email=form.username, password=form.password), db)


@router.get("/me", response_model=UserRead)
def me(current_user: User = Depends(get_current_user)) -> User:
    return current_user


@router.post("/refresh", response_model=Token)
def refresh_token(current_user: User = Depends(get_current_user)) -> Token:
    """Issue a fresh access token for an authenticated user.

    Call before the current token expires (client should refresh when
    less than ~5 min remain). The existing token is validated by
    get_current_user; on success a new token with a full TTL is returned.
    """
    return Token(access_token=create_access_token(str(current_user.id)))


# ── Team invite ───────────────────────────────────────────────────────────────

_TEMP_PASSWORD_CHARS = string.ascii_letters + string.digits + "!@#$%^&*"
_TEMP_PASSWORD_LENGTH = 16


class InviteRequest(BaseModel):
    email: EmailStr
    full_name: str = Field(min_length=1, max_length=120)
    role: UserRole = UserRole.member


class InviteResponse(BaseModel):
    user_id: int
    email: str
    role: UserRole
    email_sent: bool
    # Only present in non-production environments as a convenience for testing.
    # Never expose in production responses.
    temp_password: str | None = None


@router.post(
    "/invite",
    response_model=InviteResponse,
    status_code=status.HTTP_201_CREATED,
)
def invite_team_member(
    payload: InviteRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> InviteResponse:
    """Invite a new user to the caller's organisation.

    - Only **admins** can invite.
    - The invitee is created immediately with a random temporary password
      and the requested role.
    - A welcome email is dispatched via the notifier service (logs only
      when SMTP is not configured — see config.smtp_host).
    - The invitee must change their password on first login (honour system
      for now; a ``force_password_change`` column can be added if needed).
    - Returns the temp password **only** when ``environment != production``
      to ease local dev / testing. Never log or return it in prod.
    """
    if current_user.role != UserRole.admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only organisation admins can invite team members",
        )

    email_lower = payload.email.lower()
    if db.query(User).filter(User.email == email_lower).first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this email already exists",
        )

    # Generate a cryptographically random temp password
    temp_password = "".join(
        secrets.choice(_TEMP_PASSWORD_CHARS) for _ in range(_TEMP_PASSWORD_LENGTH)
    )

    new_user = User(
        email=email_lower,
        full_name=payload.full_name,
        hashed_password=hash_password(temp_password),
        organization_id=current_user.organization_id,
        role=payload.role,
        is_active=True,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # Dispatch welcome email via notifier (gracefully no-ops when SMTP is unconfigured)
    org = db.get(Organization, current_user.organization_id)
    org_name = org.name if org else "your organisation"
    email_body = (
        f"Hi {payload.full_name},\n\n"
        f"You've been invited to join {org_name} on PeopleSense "
        f"by {current_user.full_name} ({current_user.email}).\n\n"
        f"Your temporary password is: {temp_password}\n\n"
        f"Please log in and change your password immediately.\n\n"
        f"— The PeopleSense Team"
    )
    notifications = notifier_dispatch(
        db,
        organization_id=current_user.organization_id,
        user_id=new_user.id,
        title=f"You've been invited to {org_name} on PeopleSense",
        body=email_body,
        channels=[NotificationChannel.email, NotificationChannel.inbox],
        email_to=email_lower,
        source_type="invite",
        source_id=new_user.id,
    )
    email_sent = any(
        n.channel == NotificationChannel.email and n.status.value == "sent" for n in notifications
    )

    from ..config import get_settings  # local to avoid circular at module level

    settings = get_settings()
    expose_temp = settings.environment != "production"

    return InviteResponse(
        user_id=new_user.id,
        email=new_user.email,
        role=new_user.role,
        email_sent=email_sent,
        temp_password=temp_password if expose_temp else None,
    )
