"""Pytest fixtures for PeopleSense backend."""

from __future__ import annotations

import os
import tempfile
import uuid as _uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session as _Session
from sqlalchemy.orm import sessionmaker

from app import db as app_db
from app.config import get_settings
from app.db import SessionLocal as _SessionLocal
from app.main import app
from app.models import User as _User

# Configure environment BEFORE importing the app
_TMP = Path(tempfile.mkdtemp(prefix="peoplesense-test-"))
os.environ.setdefault("DATABASE_URL", f"sqlite:///{(_TMP / 'test.db').as_posix()}")
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-jwt")
os.environ.setdefault("ENABLE_DETECTOR", "false")
os.environ.setdefault("STORAGE_DIR", str(_TMP / "storage"))
os.environ.setdefault("UPLOAD_DIR", str(_TMP / "uploads"))
os.environ.setdefault("OUTPUT_DIR", str(_TMP / "outputs"))


@pytest.fixture(scope="session", autouse=True)
def _reset_db() -> Iterator[None]:
    """Re-create the DB engine bound to the test SQLite URL."""
    settings = get_settings()
    engine = create_engine(
        settings.database_url, connect_args={"check_same_thread": False}, future=True
    )
    app_db.engine = engine
    app_db.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)
    app_db.Base.metadata.drop_all(bind=engine)
    app_db.init_db()
    yield
    app_db.Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_client(client: TestClient) -> TestClient:
    """A TestClient pre-authenticated with a fresh user."""
    import uuid

    email = f"test-{uuid.uuid4().hex[:8]}@example.com"
    resp = client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "full_name": "Test User",
            "password": "test-password-1",
            "organization_name": "TestOrg",
        },
    )
    assert resp.status_code == 201, resp.text
    token = resp.json()["access_token"]
    client.headers.update({"Authorization": f"Bearer {token}"})
    return client


@pytest.fixture
def db() -> Iterator[_Session]:
    session = _SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def auth_headers(client: TestClient) -> dict:
    """Register a fresh user and return auth headers."""
    email = f"user-{_uuid.uuid4().hex[:8]}@example.com"
    resp = client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "full_name": "Test User",
            "password": "testpass123",
            "organization_name": f"Org-{_uuid.uuid4().hex[:6]}",
        },
    )
    assert resp.status_code == 201, resp.text
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def test_user(client: TestClient, db: _Session) -> _User:
    """Return the User ORM object for the registered test user."""
    email = f"fixture-{_uuid.uuid4().hex[:8]}@example.com"
    client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "full_name": "Fixture User",
            "password": "testpass123",
            "organization_name": f"FixtureOrg-{_uuid.uuid4().hex[:6]}",
        },
    )
    user = db.query(_User).filter(_User.email == email).first()
    assert user is not None
    return user


@pytest.fixture
def authenticated_user(client: TestClient, db: _Session):
    """
    Creates a single user and returns both:
    - ORM User object
    - Authorization headers

    Ensures tests use the same authenticated user that owns the data.
    """
    email = f"user-{_uuid.uuid4().hex[:8]}@example.com"

    resp = client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "full_name": "Test User",
            "password": "testpass123",
            "organization_name": f"Org-{_uuid.uuid4().hex[:6]}",
        },
    )

    assert resp.status_code == 201, resp.text

    token = resp.json()["access_token"]

    user = db.query(_User).filter(_User.email == email).first()

    assert user is not None

    return {
        "user": user,
        "headers": {"Authorization": f"Bearer {token}"},
    }
