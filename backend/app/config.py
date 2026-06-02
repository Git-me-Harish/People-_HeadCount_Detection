"""Application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    """Runtime configuration — all values overridable via environment."""

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
        description="JWT signing key. Override via SECRET_KEY env var.",
    )
    access_token_expire_minutes: int = 60 * 24  # 1 day

    database_url: str = Field(
        default=f"sqlite:///{BACKEND_DIR / 'storage' / 'peoplesense.db'}",
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
    video_frame_stride: int = 1
    enable_detector: bool = True

    # Phase 2: notification channels
    # Email — leave empty to disable real sends (logs only)
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    email_from: str = "noreply@peoplesense.app"

    # Anomaly detection
    anomaly_z_threshold: float = 2.5
    anomaly_baseline_days: int = 14

    # Data retention
    default_retention_days: int = 30

    # RTSP / camera stream puller
    stream_sample_interval_s: float = Field(
        default=2.0,
        description="Seconds between YOLO inference calls per camera stream. "
                    "Lower = more CPU. 2s gives 0.5fps inference which is enough for crowd counting.",
    )
    stream_persist_every_n_samples: int = Field(
        default=5,
        description="Write a DetectionRecord to DB every N successful inference calls per camera.",
    )
    stream_reconnect_max_backoff_s: float = Field(
        default=60.0,
        description="Max seconds to wait before retrying a failed RTSP connection.",
    )
    stream_max_consecutive_errors: int = Field(
        default=10,
        description="Stop the stream thread after this many consecutive frame-read errors.",
    )

    def ensure_dirs(self) -> None:
        for directory in (self.storage_dir, self.upload_dir, self.output_dir):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings