import axios from "axios";
import { useAuthStore } from "../auth/store";

export const API_PREFIX = "/api/v1";

export const api = axios.create({
  baseURL: API_PREFIX,
  timeout: 60000,
});

api.interceptors.request.use((config) => {
  const token = useAuthStore.getState().token;
  if (token) {
    config.headers = config.headers ?? {};
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

api.interceptors.response.use(
  (r) => r,
  (err) => {
    if (err?.response?.status === 401) {
      useAuthStore.getState().clear();
    }
    return Promise.reject(err);
  },
);

export interface User {
  id: number;
  email: string;
  full_name: string;
  role: string;
  is_active: boolean;
  organization_id: number;
  created_at: string;
}

export interface BBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}
export interface Detection {
  class_id: number;
  class_name: string;
  confidence: number;
  bbox: BBox;
}
export interface DetectionResult {
  person_count: number;
  detections: Detection[];
  annotated_image_b64: string | null;
  avg_confidence: number | null;
  width: number | null;
  height: number | null;
}

export interface Camera {
  id: number;
  name: string;
  location: string | null;
  stream_url: string | null;
  is_active: boolean;
  organization_id: number;
  created_at: string;
}

export interface Job {
  id: number;
  organization_id: number;
  user_id: number;
  job_type: string;
  status: "pending" | "running" | "completed" | "failed";
  progress: number;
  input_path: string;
  output_path: string | null;
  summary_json: string | null;
  error_message: string | null;
  created_at: string;
  completed_at: string | null;
}

export interface DetectionRecord {
  id: number;
  organization_id: number;
  camera_id: number | null;
  source: string;
  person_count: number;
  unique_people: number | null;
  avg_confidence: number | null;
  artifact_path: string | null;
  created_at: string;
}

export interface Alert {
  id: number;
  name: string;
  threshold: number;
  camera_id: number | null;
  webhook_url: string | null;
  is_enabled: boolean;
  organization_id: number;
  last_triggered_at: string | null;
  created_at: string;
}

export interface AnalyticsSummary {
  window_days: number;
  total_detections: number;
  peak_person_count: number;
  average_person_count: number;
  current_count: number;
  last_seen_at: string | null;
}
export interface TimeseriesPoint {
  timestamp: string;
  count: number;
  peak: number;
  samples: number;
}
