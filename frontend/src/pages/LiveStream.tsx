import { useEffect, useRef, useState } from "react";
import { useAuthStore } from "../auth/store";

interface StreamPayload {
  person_count: number;
  avg_confidence: number | null;
  annotated_image_b64: string | null;
  error?: string;
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
        } catch {
          /* ignore */
        }
      };
      ws.onerror = () => setError("WebSocket error");
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
  };

  useEffect(() => {
    return () => stop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Live stream</h1>
        <p className="text-sm text-slate-500">
          Stream your webcam to the detector and see counts in real-time.
        </p>
      </div>

      <div className="card space-y-4">
        <div className="flex items-center gap-3">
          {!running ? (
            <button onClick={start} className="btn-primary">
              Start camera
            </button>
          ) : (
            <button onClick={stop} className="btn-danger">
              Stop
            </button>
          )}
          <label className="text-sm text-slate-600">
            Frames / second:
            <input
              type="number"
              min={1}
              max={10}
              value={fps}
              onChange={(e) => setFps(Math.max(1, Math.min(10, Number(e.target.value))))}
              className="ml-2 w-16 input inline-block py-1"
            />
          </label>
        </div>
        {error && <div className="text-sm text-red-600">{error}</div>}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="card">
          <h2 className="font-semibold mb-3">Webcam feed</h2>
          <video ref={videoRef} className="w-full rounded-md bg-slate-200 aspect-video" muted />
          <canvas ref={canvasRef} className="hidden" />
        </div>
        <div className="card">
          <h2 className="font-semibold mb-3">Annotated</h2>
          {latest?.annotated_image_b64 ? (
            <>
              <img
                src={`data:image/png;base64,${latest.annotated_image_b64}`}
                className="w-full rounded-md aspect-video object-contain bg-slate-100"
                alt="Annotated stream frame"
              />
              <div className="mt-3 grid grid-cols-2 gap-4 text-sm">
                <Metric label="People" value={latest.person_count} />
                <Metric
                  label="Avg. confidence"
                  value={latest.avg_confidence != null ? latest.avg_confidence.toFixed(2) : "—"}
                />
              </div>
            </>
          ) : (
            <div className="w-full aspect-video rounded-md bg-slate-100 flex items-center justify-center text-sm text-slate-400">
              Waiting for first frame…
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: number | string }) {
  return (
    <div>
      <div className="text-xs uppercase text-slate-500">{label}</div>
      <div className="text-2xl font-semibold text-brand-700">{value}</div>
    </div>
  );
}
