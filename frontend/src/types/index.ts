/**
 * Domain types — single source of truth for all API entities.
 * Mirrors backend Pydantic schemas.
 */

// ── Auth ────────────────────────────────────────────────────────────────────

export interface User {
  id: number;
  email: string;
  full_name: string;
  role: "admin" | "member" | "viewer";
  is_active: boolean;
  organization_id: number;
  created_at: string;
}

export interface Token {
  access_token: string;
  token_type?: string;
}

// ── Organisation ────────────────────────────────────────────────────────────

export interface Organization {
  id: number;
  name: string;
  slug: string;
  created_at: string;
}

// ── Camera ──────────────────────────────────────────────────────────────────

export type StreamState =
  | "idle"
  | "starting"
  | "running"
  | "reconnecting"
  | "stopping"
  | "stopped"
  | "error";

export interface CameraStreamStatus {
  camera_id: number;
  state: StreamState;
  last_frame_at: string | null;
  last_person_count: number;
  frames_processed: number;
  consecutive_errors: number;
  reconnect_attempts: number;
  error_message: string | null;
  started_at: string | null;
}
export interface Camera {
  id: number;
  name: string;
  location: string | null;
  stream_url: string | null;
  max_capacity: number | null;
  is_active: boolean;
  organization_id: number;
  created_at: string;
}

export type CameraCreate = Pick<Camera, "name" | "location" | "stream_url" | "max_capacity">;

// ── Detection ───────────────────────────────────────────────────────────────

export interface DetectionRecord {
  id: number;
  camera_id: number | null;
  source: "image" | "video" | "stream";
  person_count: number;
  avg_confidence: number | null;
  artifact_path: string | null;
  created_at: string;
}

// ── Alert ───────────────────────────────────────────────────────────────────

export interface Alert {
  id: number;
  organization_id: number;
  camera_id: number | null;
  name: string;
  threshold: number;
  cooldown_minutes: number;
  webhook_url: string | null;
  is_enabled: boolean;
  last_triggered_at: string | null;
  created_at: string;
}

export type AlertCreate = Pick<Alert, "name" | "threshold" | "cooldown_minutes" | "camera_id" | "webhook_url">;

// ── Analytics ───────────────────────────────────────────────────────────────

export interface AnalyticsSummary {
  window_days: number;
  total_detections: number;
  peak_person_count: number;
  average_person_count: number;
  current_count: number;
  last_seen_at: string | null;
}

export interface TimeSeriesPoint {
  timestamp: string;
  count: number;
  peak: number;
  samples: number;
}

// ── Template ─────────────────────────────────────────────────────────────────

export interface IndustryTemplate {
  id: number;
  vertical: string;
  name: string;
  description: string;
  icon: string;
  default_cameras: Array<{ name: string; location?: string }>;
  default_alerts: Array<{ name: string; threshold: number }>;
}

// ── Notification ─────────────────────────────────────────────────────────────

export interface Notification {
  id: number;
  channel: string;
  status: string;
  title: string;
  body: string;
  source_type: string | null;
  source_id: number | null;
  is_read: boolean;
  created_at: string;
  read_at: string | null;
}

// ── API Token ────────────────────────────────────────────────────────────────

export interface APIToken {
  id: number;
  name: string;
  prefix: string;
  scopes: string;
  is_active: boolean;
  last_used_at: string | null;
  expires_at: string | null;
  created_at: string;
  full_token?: string; // only present on creation
}

// ── Audit ────────────────────────────────────────────────────────────────────

export interface AuditLogEntry {
  id: number;
  actor_email: string | null;
  action: string;
  resource_type: string;
  resource_id: string | null;
  ip_address: string | null;
  created_at: string;
}

// ── Plan ─────────────────────────────────────────────────────────────────────

export interface Plan {
  tier: string;
  display_name: string;
  max_cameras: number;
  max_alerts: number;
  max_api_tokens: number;
  max_frames_per_month: number;
  retention_days: number;
  can_export_pdf: boolean;
  can_use_public_page: boolean;
  price_usd_monthly: number;
}

export interface UsageCounter {
  plan_tier: string;
  cameras_used: number;
  frames_processed_month: number;
  alerts_sent_month: number;
  period_start: string;
}

export interface PlanAndUsage {
  plan: Plan;
  usage: UsageCounter;
}

// ── Heatmap ──────────────────────────────────────────────────────────────────

export interface HeatmapSnapshot {
  camera_id: number;
  camera_name: string;
  bucket_hour: string;
  grid: number[]; // 144 normalised floats
  grid_cols: number;
  grid_rows: number;
  sample_count: number;
  peak_count: number;
}

// ── Public Page ───────────────────────────────────────────────────────────────

export interface PublicPageConfig {
  slug: string;
  title: string;
  description: string | null;
  camera_ids: number[];
  show_heatmap: boolean;
  brand_color: string;
  is_active: boolean;
}

export interface PublicStatus {
  slug: string;
  title: string;
  description: string | null;
  brand_color: string;
  cameras: CameraLiveStatus[];
  generated_at: string;
}

export interface CameraLiveStatus {
  camera_id: number;
  name: string;
  location: string | null;
  current_count: number;
  last_updated: string | null;
  status: "live" | "idle" | "offline";
}

// ── Job ───────────────────────────────────────────────────────────────────────

export interface Job {
  id: number;
  job_type: string;
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  progress: number;
  summary_json: string | null;
  error_message: string | null;
  created_at: string;
  completed_at: string | null;
}

// ── Live Stream ───────────────────────────────────────────────────────────────

export interface StreamFrame {
  person_count: number;
  avg_confidence: number | null;
  detections: Array<{
    class_name: string;
    confidence: number;
    bbox: { x1: number; y1: number; x2: number; y2: number };
  }>;
  annotated_image_b64: string | null;
}

// ── API Helpers ───────────────────────────────────────────────────────────────

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
}

export type ApiError = {
  detail: string | Array<{ loc: string[]; msg: string; type: string }>;
};