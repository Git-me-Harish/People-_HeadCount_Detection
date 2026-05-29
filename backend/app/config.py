"""Application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    """Runtime configuration."""

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "PeopleSense"
    environment: str = "development"
    api_v1_prefix: str = "/api/v1"

    secret_key: str = Field(
        default="change-me-in-production-please-this-is-not-secure",
        description="JWT signing key. Override in production via SECRET_KEY env var.",
    )
    access_token_expire_minutes: int = 60 * 24  # 1 day

    database_url: str = Field(
        default=f"sqlite:///{(BACKEND_DIR / 'storage' / 'peoplesense.db').as_posix()}",
        description="SQLAlchemy DB URL. Defaults to local SQLite.",
    )

    storage_dir: Path = BACKEND_DIR / "storage"
    upload_dir: Path = BACKEND_DIR / "storage" / "uploads"
    output_dir: Path = BACKEND_DIR / "storage" / "outputs"

    yolo_model_path: str = Field(
        default=str(PROJECT_ROOT / "yolov8n.pt"),
        description="Path to YOLOv8 weights file.",
    )
    yolo_confidence: float = 0.35
    yolo_iou: float = 0.5
    person_class_id: int = 0  # COCO class id for 'person'

    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://localhost:4173",
            "http://localhost:3000",
            "http://127.0.0.1:5173",
        ]
    )

    max_upload_size_mb: int = 200
    video_frame_stride: int = 1  # process every Nth frame
    enable_detector: bool = True  # disable in tests to avoid loading weights

    def ensure_dirs(self) -> None:
        for directory in (self.storage_dir, self.upload_dir, self.output_dir):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
