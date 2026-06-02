import { useEffect, useRef, useState } from "react";
import { Download, Film, Upload } from "lucide-react";
import { api, Job } from "../api/client";
import { Button, Card, PageHeader, AlertBanner } from "../components/ui";

interface JobSummary {
  frames_processed: number;
  total_frames: number;
  average_person_count: number;
  peak_person_count: number;
  unique_people: number;
  per_frame: number[];
  duration_seconds: number | null;
}

function StatBlock({ label, value }: { label: string; value: string | number }) {
  return (
    <div
      className="px-4 py-3 rounded-xl"
      style={{ backgroundColor: "var(--bg-subtle)", border: "1px solid var(--border-base)" }}
    >
      <p className="text-2xs font-semibold uppercase tracking-widest mb-1" style={{ color: "var(--text-muted)" }}>
        {label}
      </p>
      <p className="text-xl font-mono font-medium" style={{ color: "var(--text-primary)", letterSpacing: "-0.02em" }}>
        {value}
      </p>
    </div>
  );
}

const STATUS_COLORS: Record<string, string> = {
  idle:      "var(--text-muted)",
  running:   "var(--warn)",
  pending:   "var(--warn)",
  completed: "var(--success)",
  failed:    "var(--danger)",
  cancelled: "var(--text-muted)",
};

const STATUS_BG: Record<string, string> = {
  idle:      "var(--bg-subtle)",
  running:   "var(--warn-dim)",
  pending:   "var(--warn-dim)",
  completed: "var(--success-dim)",
  failed:    "var(--danger-dim)",
  cancelled: "var(--bg-subtle)",
};

export default function VideoDetect() {
  const [file, setFile] = useState<File | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const pollRef = useRef<number | null>(null);

  const startUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const { data } = await api.post<Job>("/detect/video", form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setJob(data);
      startPolling(data.id);
    } catch (err: any) {
      setError(err?.response?.data?.detail ?? "Upload failed");
    } finally {
      setLoading(false);
    }
  };

  const startPolling = (jobId: number) => {
    if (pollRef.current) window.clearInterval(pollRef.current);
    pollRef.current = window.setInterval(async () => {
      try {
        const { data } = await api.get<Job>(`/jobs/${jobId}`);
        setJob(data);
        if (data.status === "completed" || data.status === "failed") {
          if (pollRef.current) window.clearInterval(pollRef.current);
          pollRef.current = null;
        }
      } catch { /* ignore */ }
    }, 1500);
  };

  useEffect(() => () => { if (pollRef.current) window.clearInterval(pollRef.current); }, []);

  const summary: JobSummary | null = job?.summary_json ? JSON.parse(job.summary_json) : null;
  const progressPct = job ? Math.round(job.progress * 100) : 0;
  const status = job?.status ?? "idle";

  return (
    <div className="p-6 max-w-4xl mx-auto space-y-5">
      <PageHeader
        title="Video Detection"
        subtitle="Upload a video for asynchronous per-frame crowd analysis"
      />

      {/* Upload card */}
      <Card padding={false}>
        <div
          className="px-5 py-3.5"
          style={{ borderBottom: "1px solid var(--border-subtle)" }}
        >
          <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: "var(--text-muted)" }}>
            Upload Video
          </p>
        </div>
        <div className="p-5 space-y-4">
          {error && <AlertBanner type="error">{error}</AlertBanner>}

          {/* Drop zone */}
          <div
            className="rounded-xl border-2 border-dashed flex flex-col items-center justify-center py-10 cursor-pointer transition-colors"
            style={{ borderColor: file ? "var(--accent)" : "var(--border-base)" }}
            onClick={() => !file && inputRef.current?.click()}
            onDrop={(e) => {
              e.preventDefault();
              const f = e.dataTransfer.files?.[0];
              if (f && f.type.startsWith("video/")) { setFile(f); setJob(null); }
            }}
            onDragOver={(e) => e.preventDefault()}
            onMouseEnter={(e) => { if (!file) (e.currentTarget.style.borderColor = "var(--accent)"); }}
            onMouseLeave={(e) => { if (!file) (e.currentTarget.style.borderColor = "var(--border-base)"); }}
          >
            <div
              className="w-12 h-12 rounded-xl flex items-center justify-center mb-3"
              style={{ backgroundColor: file ? "var(--accent-dim)" : "var(--bg-subtle)" }}
            >
              <Film size={22} style={{ color: file ? "var(--accent)" : "var(--text-muted)" }} />
            </div>
            {file ? (
              <div className="text-center">
                <p className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
                  {file.name}
                </p>
                <p className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>
                  {(file.size / 1024 / 1024).toFixed(1)} MB
                </p>
                <button
                  className="text-xs mt-2 underline"
                  style={{ color: "var(--text-muted)" }}
                  onClick={(e) => { e.stopPropagation(); setFile(null); setJob(null); }}
                >
                  Change file
                </button>
              </div>
            ) : (
              <>
                <p className="text-sm font-medium" style={{ color: "var(--text-secondary)" }}>
                  Drop a video here
                </p>
                <p className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>
                  or click to browse — MP4, MOV, AVI
                </p>
              </>
            )}
            <input
              ref={inputRef}
              type="file"
              accept="video/*"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) { setFile(f); setJob(null); }
              }}
            />
          </div>

          <Button
            onClick={startUpload}
            disabled={!file || loading}
            loading={loading}
            icon={<Upload size={14} />}
            className="w-full justify-center"
          >
            {loading ? "Uploading..." : "Start processing"}
          </Button>
        </div>
      </Card>

      {/* Job status card */}
      {job && (
        <Card padding={false}>
          <div
            className="flex items-center justify-between px-5 py-3.5"
            style={{ borderBottom: "1px solid var(--border-subtle)" }}
          >
            <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: "var(--text-muted)" }}>
              Job #{job.id}
            </p>
            <span
              className="text-2xs font-semibold px-2.5 py-1 rounded-md capitalize"
              style={{
                color: STATUS_COLORS[status] ?? "var(--text-muted)",
                backgroundColor: STATUS_BG[status] ?? "var(--bg-subtle)",
              }}
            >
              {status}
            </span>
          </div>

          <div className="p-5 space-y-5">
            {/* Progress bar */}
            {(status === "running" || status === "pending") && (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <p className="text-xs" style={{ color: "var(--text-muted)" }}>Processing frames...</p>
                  <span className="text-xs font-mono font-medium" style={{ color: "var(--text-primary)" }}>
                    {progressPct}%
                  </span>
                </div>
                <div
                  className="h-1.5 rounded-full overflow-hidden"
                  style={{ backgroundColor: "var(--bg-muted)" }}
                >
                  <div
                    className="h-full rounded-full transition-all duration-500"
                    style={{
                      width: `${progressPct}%`,
                      backgroundColor: status === "running" ? "var(--accent)" : "var(--text-muted)",
                    }}
                  />
                </div>
              </div>
            )}

            {job.error_message && <AlertBanner type="error">{job.error_message}</AlertBanner>}

            {/* Summary stats */}
            {summary && (
              <div className="space-y-3">
                <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: "var(--text-muted)" }}>
                  Analysis Summary
                </p>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <StatBlock
                    label="Frames"
                    value={`${summary.frames_processed} / ${summary.total_frames}`}
                  />
                  <StatBlock label="Peak count" value={summary.peak_person_count} />
                  <StatBlock label="Average" value={Number(summary.average_person_count).toFixed(1)} />
                  <StatBlock
                    label="Duration"
                    value={summary.duration_seconds != null ? `${summary.duration_seconds.toFixed(1)}s` : "—"}
                  />
                </div>
              </div>
            )}

            {status === "completed" && (
              <a
                href={`/api/v1/jobs/${job.id}/artifact`}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all"
                style={{ backgroundColor: "var(--accent)", color: "#0b0f14" }}
              >
                <Download size={14} />
                Download annotated video
              </a>
            )}
          </div>
        </Card>
      )}
    </div>
  );
}