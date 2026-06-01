"""Authentication endpoints."""

from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import get_current_user
from ..models import Organization, User, UserRole
from ..schemas.auth import LoginRequest, RegisterRequest, Token
from ..schemas.user import UserRead
from ..security import create_access_token, hash_password, verify_password

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