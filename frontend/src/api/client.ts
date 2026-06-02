// Axios API client — single instance, typed endpoint helpers.

import axios, { type AxiosInstance } from "axios";
import { API_BASE_URL } from "../constants";
import type {
  AlertCreate,
  AnalyticsSummary,
  APIToken,
  AuditLogEntry,
  Camera,
  CameraCreate,
  CameraStreamStatus,
  DetectionRecord,
  HeatmapSnapshot,
  IndustryTemplate,
  Job,
  Notification,
  PlanAndUsage,
  PublicPageConfig,
  PublicStatus,
  TimeSeriesPoint,
  Token,
  User,
  Alert,
} from "../types";

// Re-export types for consumers
export type {
  AnalyticsSummary,
  TimeSeriesPoint,
  Camera,
  Job,
  DetectionRecord,
};

// ── Instance ───────────────────────────────────────────────────────────[...]

// Decode JWT payload without verifying signature (client-side only — verification happens server-side)
function _jwtExp(token: string): number | null {
  try {
    const payload = JSON.parse(atob(token.split(".")[1]));
    return typeof payload.exp === "number" ? payload.exp : null;
  } catch {
    return null;
  }
}

// Proactive refresh: if token expires within 5 minutes, refresh before the request goes out.
// This prevents the 24h expiry blank-screen problem on long-running ops dashboards.
let _refreshing: Promise<string | null> | null = null;

async function _maybeRefreshToken(): Promise<void> {
  const token = localStorage.getItem("ps_token");
  if (!token) return;

  const exp = _jwtExp(token);
  if (exp === null) return;

  const secsRemaining = exp - Math.floor(Date.now() / 1000);
  if (secsRemaining > 5 * 60) return; // more than 5 min left — no refresh needed

  // Deduplicate concurrent refresh calls (multiple in-flight requests hit this simultaneously)
  if (!_refreshing) {
    _refreshing = axios
      .post<Token>(`${API_BASE_URL}/auth/refresh`, null, {
        headers: { Authorization: `Bearer ${token}` },
        timeout: 10_000,
      })
      .then((res) => {
        const newToken = res.data.access_token;
        localStorage.setItem("ps_token", newToken);
        return newToken;
      })
      .catch(() => {
        // Refresh failed (token already expired) — force logout
        localStorage.removeItem("ps_token");
        window.location.href = "/login";
        return null;
      })
      .finally(() => {
        _refreshing = null;
      });
  }

  await _refreshing;
}

const createAxiosInstance = (): AxiosInstance => {
  const instance = axios.create({
    baseURL: API_BASE_URL,
    timeout: 30_000,
    headers: { "Content-Type": "application/json" },
  });

  instance.interceptors.request.use(async (config) => {
    // Skip refresh check for the refresh call itself to avoid infinite recursion
    if (!config.url?.includes("/auth/refresh")) {
      await _maybeRefreshToken();
    }
    const token = localStorage.getItem("ps_token");
    if (token) config.headers.Authorization = `Bearer ${token}`;
    return config;
  });

  instance.interceptors.response.use(
    (response) => response,
    (error) => {
      if (error.response?.status === 401) {
        localStorage.removeItem("ps_token");
        window.location.href = "/login";
      }
      return Promise.reject(error);
    }
  );

  return instance;
};

export const api = createAxiosInstance();

// ── Auth ────────────────────────────────────────────────────────────[...]

export const authApi = {
  login: (email: string, password: string) =>
    api.post<Token>("/auth/login", { email, password }),
  register: (payload: { email: string; password: string; full_name: string; organization_name: string }) =>
    api.post<Token>("/auth/register", payload),
  me: () => api.get<User>("/auth/me"),
  // Explicitly refresh token on demand (e.g. on app focus/wake)
  refresh: () => api.post<Token>("/auth/refresh"),
  invite: (payload: { email: string; full_name: string; role: "admin" | "member" | "viewer" }) =>
    api.post<{ user_id: number; email: string; role: string; email_sent: boolean; temp_password?: string }>(
      "/auth/invite",
      payload
    ),
};

// ── Cameras ───────────────────────────────────────────────────────────[...]

export const camerasApi = {
  list: () => api.get<Camera[]>("/cameras"),
  get: (id: number) => api.get<Camera>(`/cameras/${id}`),
  create: (data: CameraCreate) => api.post<Camera>("/cameras", data),
  update: (id: number, data: Partial<CameraCreate>) => api.patch<Camera>(`/cameras/${id}`, data),
  delete: (id: number) => api.delete(`/cameras/${id}`),
  // Stream management
  getStreamStatus: (id: number) => api.get<CameraStreamStatus>(`/cameras/${id}/stream/status`),
  startStream: (id: number) => api.post<{ camera_id: number; action: string }>(`/cameras/${id}/stream/start`),
  stopStream: (id: number) => api.post<{ camera_id: number; action: string; was_running: boolean }>(`/cameras/${id}/stream/stop`),
};

// ── Analytics ──────────────────────────────────────────────────────────[[...]

export const analyticsApi = {
  summary: (days = 7) => api.get<AnalyticsSummary>("/analytics/summary", { params: { days } }),
  timeseries: (days = 7, bucketMinutes = 60) =>
    api.get<TimeSeriesPoint[]>("/analytics/timeseries", { params: { days, bucket_minutes: bucketMinutes } }),
  records: (limit = 50) => api.get<DetectionRecord[]>("/analytics/records", { params: { limit } }),
};

// ── Alerts ───────────────────────────────────────────────────────────[[...]

export const alertsApi = {
  list: () => api.get<Alert[]>("/alerts"),
  create: (data: AlertCreate) => api.post<Alert>("/alerts", data),
  update: (id: number, data: Partial<AlertCreate & { is_enabled: boolean }>) =>
    api.patch<Alert>(`/alerts/${id}`, data),
  delete: (id: number) => api.delete(`/alerts/${id}`),
};

// ── Templates ──────────────────────────────────────────────────────────[[...]

export const templatesApi = {
  list: () => api.get<IndustryTemplate[]>("/templates"),
  apply: (vertical: string) =>
    api.post<{ cameras_created: Camera[]; alerts_created: Alert[]; message: string }>(
      `/templates/${vertical}/apply`
    ),
};

// ── Notifications ─────────────────────────────────────────────────────────[...]

export const notificationsApi = {
  list: (unreadOnly = false) =>
    api.get<Notification[]>("/notifications", { params: { unread_only: unreadOnly } }),
  countUnread: () => api.get<{ unread_count: number }>("/notifications/count-unread"),
  markRead: (ids: number[]) =>
    api.post<{ marked_read: number }>("/notifications/mark-read", { notification_ids: ids }),
  markAllRead: () => api.post<{ marked_read: number }>("/notifications/mark-all-read"),
  delete: (id: number) => api.delete(`/notifications/${id}`),
};

// ── API Tokens ──────────────────────────────────────────────────────────[...]

export const apiTokensApi = {
  list: () => api.get<APIToken[]>("/api-tokens"),
  create: (data: { name: string; scopes?: string; expires_at?: string }) =>
    api.post<APIToken & { full_token: string }>("/api-tokens", data),
  revoke: (id: number) => api.delete(`/api-tokens/${id}`),
};

// ── Audit ───────────────────────────────────────────────────────────[.[...]

export const auditApi = {
  list: (params?: { limit?: number; resource_type?: string }) =>
    api.get<AuditLogEntry[]>("/audit", { params }),
};

// ── Reports ───────────────────────────────────────────────────────────[...]

export class ReportUnavailableError extends Error {
  constructor(detail: string) {
    super(detail);
    this.name = "ReportUnavailableError";
  }
}

export const reportsApi = {
  downloadPdf: async (days = 7): Promise<void> => {
    const response = await api.get(`/reports/summary/pdf`, {
      params: { days },
      responseType: "blob",
    });

    // When reportlab is missing the backend returns 503 with a JSON body.
    // Axios sees responseType=blob and delivers a Blob — not an AxiosError — so
    // the caller's catch block never fires. Inspect content-type to detect this.
    const contentType: string = String(response.headers["content-type"] ?? "");
    if (!contentType.includes("application/pdf")) {
      const text = await (response.data as Blob).text();
      let detail = "PDF generation unavailable — reportlab not installed on the backend.";
      try {
        const parsed = JSON.parse(text);
        if (parsed?.detail) detail = parsed.detail;
      } catch {
        // ignore — use default message
      }
      throw new ReportUnavailableError(detail);
    }

    const url = URL.createObjectURL(response.data);
    const link = document.createElement("a");
    link.href = url;
    link.download = `peoplesense_report_${days}d.pdf`;
    link.click();
    URL.revokeObjectURL(url);
  },
};

// ── Public Page ─────────────────────────────────────────────────────────[.[...]

export const publicPageApi = {
  create: (data: Omit<PublicPageConfig, "is_active">) =>
    api.post<PublicPageConfig>("/public/pages", data),
  getMine: () => api.get<PublicPageConfig | null>("/public/pages/mine"),
  update: (data: Omit<PublicPageConfig, "is_active">) =>
    api.patch<PublicPageConfig>("/public/pages/mine", data),
  getPublic: (slug: string) => api.get<PublicStatus>(`/public/${slug}`),
};

// ── Heatmaps ──────────────────────────────────────────────────────────[.[...]

export const heatmapsApi = {
  latest: (cameraId: number) => api.get<HeatmapSnapshot>(`/heatmaps/camera/${cameraId}/latest`),
  history: (cameraId: number, hours = 24) =>
    api.get<HeatmapSnapshot[]>(`/heatmaps/camera/${cameraId}/history`, { params: { hours } }),
};

// ── Plan ────────────────────────────────────────────────────────────[...]

export const planApi = {
  get: () => api.get<PlanAndUsage>("/plan"),
  getUpgradeOptions: () => api.get<{
    current_tier: string;
    available_upgrades: Array<{
      tier: string;
      display_name: string;
      price_usd_monthly: number | null;
      cameras: number;
      alerts: number;
      highlights: string[];
    }>;
  }>("/plan/upgrade-options"),
  upgrade: (target_tier: string) => api.post<{
    success: boolean;
    previous_tier: string;
    new_tier: string;
    message: string;
  }>("/plan/upgrade", { target_tier }),
};

// ── Jobs ────────────────────────────────────────────────────────────[...]

export const jobsApi = {
  list: () => api.get<Job[]>("/jobs"),
  get: (id: number) => api.get<Job>(`/jobs/${id}`),
  cancel: (id: number) => api.delete<{ job_id: number; status: string; method: string; note?: string }>(`/jobs/${id}`),
};
