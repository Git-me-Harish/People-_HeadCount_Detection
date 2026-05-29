"""Pytest fixtures for PeopleSense backend."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

# Configure environment BEFORE importing the app
_TMP = Path(tempfile.mkdtemp(prefix="peoplesense-test-"))
os.environ.setdefault("DATABASE_URL", f"sqlite:///{(_TMP / 'test.db').as_posix()}")
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-jwt")
os.environ.setdefault("ENABLE_DETECTOR", "false")
os.environ.setdefault("STORAGE_DIR", str(_TMP / "storage"))
os.environ.setdefault("UPLOAD_DIR", str(_TMP / "uploads"))
os.environ.setdefault("OUTPUT_DIR", str(_TMP / "outputs"))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app import db as app_db  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.main import app  # noqa: E402


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
