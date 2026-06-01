/**
 * Public unauthenticated crowd-status view.
 * Route: /public/:slug (no auth required)
 */
import { useParams } from "react-router-dom";
import { format } from "date-fns";
import { Activity, Clock, Users, Wifi, WifiOff } from "lucide-react";
import { publicPageApi } from "../api/client";
import { usePolling } from "../hooks";
import type { CameraLiveStatus } from "../types";

function StatusDot({ status }: { status: CameraLiveStatus["status"] }) {
  const styles: Record<string, { bg: string; pulse: boolean }> = {
    live:    { bg: "#34d399", pulse: true },
    idle:    { bg: "#f59e0b", pulse: false },
    offline: { bg: "#4a5c78", pulse: false },
  };
  const s = styles[status] ?? styles.offline;
  return (
    <span
      className={`inline-block w-2 h-2 rounded-full flex-shrink-0 ${s.pulse ? "animate-pulse" : ""}`}
      style={{ backgroundColor: s.bg }}
    />
  );
}

function CrowdBar({
  count,
  max = 300,
  brandColor,
}: {
  count: number;
  max?: number;
  brandColor: string;
}) {
  const pct = Math.min(100, (count / max) * 100);
  const barColor =
    pct > 80 ? "#f87171" : pct > 50 ? "#f59e0b" : brandColor;
  return (
    <div
      className="h-1.5 rounded-full overflow-hidden"
      style={{ backgroundColor: "rgba(255,255,255,0.08)" }}
    >
      <div
        className="h-full rounded-full transition-all duration-700"
        style={{ width: `${pct}%`, backgroundColor: barColor }}
      />
    </div>
  );
}

export default function PublicView() {
  const { slug } = useParams<{ slug: string }>();
  const { data, loading, error } = usePolling(
    () => publicPageApi.getPublic(slug!),
    30_000,
    [slug]
  );

  if (loading) {
    return (
      <div
        className="min-h-screen flex items-center justify-center"
        style={{ backgroundColor: "#0b0f14" }}
      >
        <div className="flex items-center gap-3" style={{ color: "#8a9bb5" }}>
          <Activity size={18} className="animate-pulse" style={{ color: "#00c2a8" }} />
          <span className="text-sm font-medium">Loading status...</span>
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div
        className="min-h-screen flex items-center justify-center"
        style={{ backgroundColor: "#0b0f14" }}
      >
        <div className="text-center">
          <WifiOff size={36} className="mx-auto mb-4" style={{ color: "#1e2a3d" }} />
          <p className="text-lg font-bold" style={{ color: "#e8edf5" }}>
            Page not found
          </p>
          <p className="text-sm mt-1" style={{ color: "#4a5c78" }}>
            This public status page does not exist or has been deactivated.
          </p>
        </div>
      </div>
    );
  }

  const totalCount = data.cameras.reduce((s: number, c: CameraLiveStatus) => s + c.current_count, 0);
  const liveCams = data.cameras.filter((c: CameraLiveStatus) => c.status === "live").length;
  const brand = data.brand_color || "#00c2a8";

  return (
    <div
      className="min-h-screen"
      style={{
        backgroundColor: "#0b0f14",
        backgroundImage: `radial-gradient(ellipse at 50% 0%, ${brand}18 0%, transparent 60%)`,
      }}
    >
      <div className="max-w-xl mx-auto px-5 py-12">
        {/* Header */}
        <div className="text-center mb-10">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full mb-4" style={{
            backgroundColor: `${brand}18`,
            border: `1px solid ${brand}33`,
          }}>
            <span
              className="w-1.5 h-1.5 rounded-full animate-pulse"
              style={{ backgroundColor: brand }}
            />
            <span className="text-xs font-semibold uppercase tracking-widest" style={{ color: brand }}>
              Live Status
            </span>
          </div>
          <h1 className="text-3xl font-bold mb-2" style={{ color: "#e8edf5" }}>
            {data.title}
          </h1>
          {data.description && (
            <p className="text-sm" style={{ color: "#8a9bb5" }}>
              {data.description}
            </p>
          )}
        </div>

        {/* Total count circle */}
        <div className="flex flex-col items-center mb-10">
          <div
            className="flex flex-col items-center justify-center w-36 h-36 rounded-full"
            style={{
              border: `3px solid ${brand}`,
              backgroundColor: `${brand}10`,
            }}
          >
            <span
              className="text-4xl font-mono font-medium"
              style={{ color: "#e8edf5", letterSpacing: "-0.04em" }}
            >
              {totalCount}
            </span>
            <span className="text-xs mt-1" style={{ color: "#4a5c78" }}>
              people now
            </span>
          </div>
          <p className="text-xs mt-3" style={{ color: "#4a5c78" }}>
            {liveCams} of {data.cameras.length} camera{data.cameras.length !== 1 ? "s" : ""} live
          </p>
        </div>

        {/* Camera cards */}
        <div className="space-y-3">
          {data.cameras.map((cam: CameraLiveStatus) => (
            <div
              key={cam.camera_id}
              className="rounded-xl p-5"
              style={{
                backgroundColor: "#111620",
                border: "1px solid #1e2a3d",
              }}
            >
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2 min-w-0">
                  <StatusDot status={cam.status} />
                  <span className="text-sm font-semibold truncate" style={{ color: "#e8edf5" }}>
                    {cam.name}
                  </span>
                  {cam.location && (
                    <span className="text-xs hidden sm:inline" style={{ color: "#4a5c78" }}>
                      · {cam.location}
                    </span>
                  )}
                </div>
                <div
                  className="flex items-center gap-1 text-2xs flex-shrink-0 ml-2"
                  style={{ color: cam.status === "live" ? "#34d399" : "#4a5c78" }}
                >
                  {cam.status === "live" ? <Wifi size={12} /> : <WifiOff size={12} />}
                  <span className="capitalize">{cam.status}</span>
                </div>
              </div>

              <div className="flex items-center gap-2 mb-3">
                <Users size={16} style={{ color: brand }} />
                <span
                  className="text-2xl font-mono font-medium"
                  style={{ color: "#e8edf5", letterSpacing: "-0.02em" }}
                >
                  {cam.current_count}
                </span>
                <span className="text-xs" style={{ color: "#4a5c78" }}>
                  detected
                </span>
              </div>

              <CrowdBar count={cam.current_count} brandColor={brand} />

              {cam.last_updated && (
                <p
                  className="text-2xs mt-2.5 flex items-center gap-1 font-mono"
                  style={{ color: "#4a5c78" }}
                >
                  <Clock size={10} />
                  Updated {format(new Date(cam.last_updated), "HH:mm:ss")}
                </p>
              )}
            </div>
          ))}
        </div>

        {/* Footer */}
        <div className="text-center mt-10">
          <p className="text-xs" style={{ color: "#1e2a3d" }}>
            Powered by{" "}
            <span className="font-semibold" style={{ color: brand }}>
              PeopleSense
            </span>
            {" · "}Auto-refreshes every 30s
          </p>
          <p className="text-2xs mt-1 font-mono" style={{ color: "#1e2a3d" }}>
            {format(new Date(data.generated_at), "HH:mm:ss")}
          </p>
        </div>
      </div>
    </div>
  );
}