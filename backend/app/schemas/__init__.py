"""Pydantic schemas (API contracts)."""

from .alert import AlertCreate, AlertRead, AlertUpdate
from .auth import LoginRequest, RegisterRequest, Token, TokenPayload
from .camera import CameraCreate, CameraRead, CameraUpdate
from .common import BoundingBox, Detection, DetectionResult, JobSummary
from .detection import DetectionRecordRead
from .job import JobRead
from .user import UserRead

__all__ = [
    "AlertCreate",
    "AlertRead",
    "AlertUpdate",
    "BoundingBox",
    "CameraCreate",
    "CameraRead",
    "CameraUpdate",
    "Detection",
    "DetectionRecordRead",
    "DetectionResult",
    "JobRead",
    "JobSummary",
    "LoginRequest",
    "RegisterRequest",
    "Token",
    "TokenPayload",
    "UserRead",
]
