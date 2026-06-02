/**
 * Application-wide constants.
 * All magic values live here — never inline in components.
 */

export const APP_NAME = "PeopleSense" as const;
export const APP_TAGLINE = "Crowd Intelligence for Safer Public Spaces" as const;

export const API_BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000/api/v1";

export const ROUTES = {
  HOME: "/",
  LOGIN: "/login",
  REGISTER: "/register",
  ONBOARDING: "/onboarding",
  DASHBOARD: "/dashboard",
  CAMERAS: "/cameras",
  LIVE: "/live",
  IMAGE_DETECT: "/detect/image",
  VIDEO_DETECT: "/detect/video",
  ANALYTICS: "/analytics",
  ALERTS: "/alerts",
  NOTIFICATIONS: "/notifications",
  API_TOKENS: "/api-tokens",
  REPORTS: "/reports",
  PUBLIC_PAGE: "/public-page",
  AUDIT: "/audit",
  PLAN: "/plan",
  SETTINGS: "/settings",
} as const;

export const PLAN_TIERS = {
  FREE: "free",
  PRO: "pro",
  ENTERPRISE: "enterprise",
} as const;

export const ALERT_COOLDOWN_MINUTES = 5;

/** Density thresholds for colour coding (people per camera feed). */
export const DENSITY_THRESHOLDS = {
  LOW: 20,
  MEDIUM: 50,
  HIGH: 100,
  CRITICAL: 200,
} as const;

export const DENSITY_COLORS = {
  LOW: "#34d399",
  MEDIUM: "#f59e0b",
  HIGH: "#f97316",
  CRITICAL: "#f87171",
} as const;

export const GRID_COLS = 16;
export const GRID_ROWS = 9;

export const CHART_COLORS = {
  PRIMARY: "#00c2a8",
  SECONDARY: "#4a9eff",
  ACCENT: "#f59e0b",
  SUCCESS: "#34d399",
  DANGER: "#f87171",
} as const;

/** No emoji icons — labels only. Icon rendering handled per-component. */
export const VERTICALS = [
  { key: "religious",  label: "Temple / Religious",    sub: "Pilgrimage safety & crowd flow" },
  { key: "transit",    label: "Public Transit",         sub: "Platform density management" },
  { key: "retail",     label: "Retail / Mall",          sub: "Footfall analytics & queue time" },
  { key: "hospital",   label: "Hospital / Clinic",      sub: "Emergency area monitoring" },
  { key: "education",  label: "Schools / University",   sub: "Campus occupancy control" },
  { key: "stadium",    label: "Stadium / Events",       sub: "Gate flow & safety thresholds" },
  { key: "workplace",  label: "Workplace / Building",   sub: "Space utilization insights" },
  { key: "tourism",    label: "Tourism / Museum",       sub: "Visitor experience optimization" },
] as const;

export const MAX_NOTIFICATION_DISPLAY = 50;
export const NOTIFICATION_POLL_INTERVAL_MS = 30_000;
