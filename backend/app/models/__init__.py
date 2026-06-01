"""Model registry — all ORM classes imported here so Base.metadata is populated."""

from .alert import Alert
from .api_token import APIToken
from .audit import AuditLog
from .camera import Camera
from .detection import DetectionRecord
from .heatmap import HeatmapSnapshot
from .job import Job, JobStatus, JobType
from .notification import Notification, NotificationChannel, NotificationStatus
from .org import Organization
from .plan import Plan, PlanTier, UsageCounter
from .public_page import PublicPage
from .template import IndustryTemplate, Vertical
from .user import User, UserRole

__all__ = [
    "Alert",
    "APIToken",
    "AuditLog",
    "Camera",
    "DetectionRecord",
    "HeatmapSnapshot",
    "Job",
    "JobStatus",
    "JobType",
    "Notification",
    "NotificationChannel",
    "NotificationStatus",
    "Organization",
    "Plan",
    "PlanTier",
    "PublicPage",
    "UsageCounter",
    "IndustryTemplate",
    "Vertical",
    "User",
    "UserRole",
]
