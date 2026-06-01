import { useEffect, useRef, useState } from "react";
import { Play, Square, Radio } from "lucide-react";
import { useAuthStore } from "../auth/store";
import { Button, Card, PageHeader, AlertBanner } from "../components/ui";

interface StreamPayload {
  person_count: number;
  avg_confidence: number | null;
  annotated_image_b64: string | null;
  error?: string;
}

function MetricBlock({
  label,
  value,
  unit,
  highlight = false,
}: {
  label: string;
  value: string | number;
  unit?: string;
  highlight?: boolean;
}) {
  return (
    <div
      className="flex-1 px-5 py-4 rounded-xl"
      style={{
        backgroundColor: highlight ? "var(--accent-dim)" : "var(--bg-subtle)",
        border: `1px solid ${highlight ? "var(--accent)" : "var(--border-base)"}`,
      }}
    >
      <p className="text-2xs font-semibold uppercase tracking-widest mb-1.5" style={{ color: "var(--text-muted)" }}>
        {label}
      </p>
      <p
        className="text-3xl font-mono font-medium leading-none"
        style={{ color: highlight ? "var(--accent)" : "var(--text-primary)", letterSpacing: "-0.02em" }}
      >
        {value}
        {unit && <span className="text-sm font-normal ml-1" style={{ color: "var(--text-muted)" }}>{unit}</span>}
      </p>
    </div>
  );
}

export default function LiveStream() {
  const token = useAuthStore((s) => s.token);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<number | null>(null);
  const [running, setRunning] = useState(false);
  const [latest, setLatest] = useState<StreamPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [fps, setFps] = useState(2);
  const [frameCount, setFrameCount] = useState(0);

  const start = async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      const wsProto = window.location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(
        `${wsProto}://${window.location.host}/api/v1/stream/ws?token=${encodeURIComponent(token ?? "")}`,
      );
      wsRef.current = ws;
      ws.onmessage = (ev) => {
        try {
          const data: StreamPayload = JSON.parse(ev.data);
          setLatest(data);
          setFrameCount((c) => c + 1);
        } catch { /* ignore */ }
      };
      ws.onerror = () => setError("WebSocket connection error");
      ws.onclose = () => setRunning(false);
      ws.onopen = () => {
        setRunning(true);
        intervalRef.current = window.setInterval(() => sendFrame(), Math.max(100, 1000 / fps));
      };
    } catch (err: any) {
      setError(err?.message ?? "Unable to access webcam");
    }
  };

  const sendFrame = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ws = wsRef.current;
    if (!video || !canvas || !ws || ws.readyState !== WebSocket.OPEN) return;
    const w = video.videoWidth;
    const h = video.videoHeight;
    if (!w || !h) return;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, w, h);
    const b64 = canvas.toDataURL("image/jpeg", 0.6);
    ws.send(JSON.stringify({ image: b64 }));
  };

  const stop = () => {
    if (intervalRef.current) window.clearInterval(intervalRef.current);
    intervalRef.current = null;
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setRunning(false);
    setLatest(null);
  };

  useEffect(() => () => stop(), []);

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-5">
      <PageHeader
        title="Live Detection"
        subtitle="Real-time webcam crowd analysis via WebSocket stream"
        actions={
          <div className="flex items-center gap-3">
            {running && (
              <div className="flex items-center gap-2 text-xs px-3 py-1.5 rounded-lg"
                style={{ backgroundColor: "var(--success-dim)", border: "1px solid var(--success)", color: "var(--success)" }}>
                <span className="w-1.5 h-1.5 rounded-full bg-[var(--success)] animate-pulse" />
                {frameCount} frames
              </div>
            )}
            {running ? (
              <Button variant="danger" size="sm" icon={<Square size={13} />} onClick={stop}>
                Stop session
              </Button>
            ) : (
              <Button size="sm" icon={<Play size={13} />} onClick={start}>
                Start detection
              </Button>
            )}
          </div>
        }
      />

      {error && <AlertBanner type="error">{error}</AlertBanner>}

      {/* FPS control */}
      <Card>
        <div className="flex items-center gap-6">
          <div className="space-y-1.5">
            <label className="text-xs font-semibold uppercase tracking-wider" style={{ color: "var(--text-muted)" }}>
              Frame rate
            </label>
            <div className="flex items-center gap-3">
              <input
                type="range"
                min={1}
                max={10}
                value={fps}
                onChange={(e) => setFps(Number(e.target.value))}
                className="w-32 accent-[var(--accent)]"
              />
              <span className="text-sm font-mono font-medium w-16" style={{ color: "var(--text-primary)" }}>
                {fps} fps
              </span>
            </div>
          </div>
          <div className="h-10 w-px" style={{ backgroundColor: "var(--border-base)" }} />
          <p className="text-xs leading-relaxed max-w-xs" style={{ color: "var(--text-muted)" }}>
            Higher frame rate = more detections per second. Reduce if bandwidth is limited.
          </p>
        </div>
      </Card>

      {/* Live metrics */}
      {latest && (
        <div className="flex gap-4">
          <MetricBlock
            label="People detected"
            value={latest.person_count}
            highlight={true}
          />
          <MetricBlock
            label="Avg. confidence"
            value={
              latest.avg_confidence != null
                ? (latest.avg_confidence * 100).toFixed(1)
                : "—"
            }
            unit={latest.avg_confidence != null ? "%" : undefined}
          />
          <MetricBlock
            label="Frames processed"
            value={frameCount}
          />
        </div>
      )}

      {/* Video grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Raw feed */}
        <Card padding={false}>
          <div
            className="flex items-center gap-2.5 px-5 py-3.5"
            style={{ borderBottom: "1px solid var(--border-subtle)" }}
          >
            <Radio size={13} style={{ color: running ? "var(--success)" : "var(--text-muted)" }} />
            <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: "var(--text-muted)" }}>
              Webcam Feed
            </p>
            {running && (
              <span
                className="ml-auto text-2xs px-2 py-0.5 rounded-md font-medium"
                style={{ backgroundColor: "var(--success-dim)", color: "var(--success)" }}
              >
                Live
              </span>
            )}
          </div>
          <div className="p-4">
            <video
              ref={videoRef}
              className="w-full rounded-lg aspect-video object-cover"
              style={{ backgroundColor: "var(--bg-subtle)" }}
              muted
            />
          </div>
        </Card>

        {/* Annotated */}
        <Card padding={false}>
          <div
            className="flex items-center gap-2.5 px-5 py-3.5"
            style={{ borderBottom: "1px solid var(--border-subtle)" }}
          >
            <div
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: latest?.annotated_image_b64 ? "var(--accent)" : "var(--border-base)" }}
            />
            <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: "var(--text-muted)" }}>
              Annotated Output
            </p>
          </div>
          <div className="p-4">
            {latest?.annotated_image_b64 ? (
              <img
                src={`data:image/png;base64,${latest.annotated_image_b64}`}
                className="w-full rounded-lg aspect-video object-contain"
                style={{ backgroundColor: "var(--bg-subtle)" }}
                alt="Annotated detection frame"
              />
            ) : (
              <div
                className="w-full aspect-video rounded-lg flex items-center justify-center text-sm"
                style={{ backgroundColor: "var(--bg-subtle)", color: "var(--text-muted)" }}
              >
                {running ? "Processing first frame..." : "Start a session to see detection output"}
              </div>
            )}
          </div>
        </Card>
      </div>

      <canvas ref={canvasRef} className="hidden" />
    </div>
  );
}