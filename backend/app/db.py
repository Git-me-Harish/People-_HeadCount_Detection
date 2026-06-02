"""Database engine and session management."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import get_settings


class Base(DeclarativeBase):
    """Base ORM declarative class."""


def _make_engine():
    settings = get_settings()
    connect_args: dict = {}
    db_url = settings.database_url

    if db_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        # Extract the file path from sqlite:///... and ensure the parent dir exists.
        # as_posix() produces forward-slash paths that sqlite3 on Windows can't open
        # when the directory doesn't exist yet. We resolve via pathlib and mkdir first.
        if db_url.startswith("sqlite:///"):
            db_path_str = db_url[len("sqlite:///"):]
            db_path = Path(db_path_str)
            db_path.parent.mkdir(parents=True, exist_ok=True)

    return create_engine(db_url, connect_args=connect_args, future=True)


engine = _make_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)


def get_db() -> Iterator[Session]:
    """FastAPI dependency that yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create database tables if they don't exist.

    Importing models registers them with the Base metadata.
    """
    from . import models  # noqa: F401  (side-effect import)

    Base.metadata.create_all(bind=engine)
