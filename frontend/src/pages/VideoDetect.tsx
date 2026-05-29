import { useEffect, useRef, useState } from "react";
import { api, Job } from "../api/client";

interface JobSummary {
  frames_processed: number;
  total_frames: number;
  average_person_count: number;
  peak_person_count: number;
  unique_people: number;
  per_frame: number[];
  duration_seconds: number | null;
}

export default function VideoDetect() {
  const [file, setFile] = useState<File | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
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
      } catch {
        /* ignore */
      }
    }, 1500);
  };

  useEffect(() => {
    return () => {
      if (pollRef.current) window.clearInterval(pollRef.current);
    };
  }, []);

  const summary: JobSummary | null = job?.summary_json ? JSON.parse(job.summary_json) : null;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Video detection</h1>
        <p className="text-sm text-slate-500">
          Upload a video; we'll process it asynchronously and produce an annotated MP4 with
          per-frame counts.
        </p>
      </div>

      <div className="card space-y-4">
        <input
          type="file"
          accept="video/*"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          className="block w-full text-sm file:mr-4 file:rounded-md file:border-0 file:bg-brand-50 file:px-4 file:py-2 file:text-sm file:font-medium file:text-brand-700 hover:file:bg-brand-100"
        />
        <button onClick={startUpload} disabled={!file || loading} className="btn-primary">
          {loading ? "Uploading…" : "Start processing"}
        </button>
        {error && <div className="text-sm text-red-600">{error}</div>}
      </div>

      {job && (
        <div className="card space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <div className="text-xs uppercase text-slate-500">Job #{job.id}</div>
              <div className="text-lg font-semibold capitalize">{job.status}</div>
            </div>
            <div className="flex-1 max-w-md">
              <div className="h-2 w-full rounded-full bg-slate-200 overflow-hidden">
                <div
                  className="h-full bg-brand-600 transition-all"
                  style={{ width: `${Math.round(job.progress * 100)}%` }}
                />
              </div>
              <div className="text-xs text-slate-500 mt-1">
                {Math.round(job.progress * 100)}%
              </div>
            </div>
          </div>

          {job.error_message && (
            <div className="text-sm text-red-600 bg-red-50 rounded-md px-3 py-2">
              {job.error_message}
            </div>
          )}

          {summary && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <Stat label="Frames" value={`${summary.frames_processed} / ${summary.total_frames}`} />
              <Stat label="Peak count" value={summary.peak_person_count} />
              <Stat label="Average" value={summary.average_person_count.toFixed(2)} />
              <Stat
                label="Duration"
                value={
                  summary.duration_seconds != null
                    ? `${summary.duration_seconds.toFixed(1)}s`
                    : "—"
                }
              />
            </div>
          )}

          {job.status === "completed" && (
            <a
              className="btn-primary inline-flex w-fit"
              href={`/api/v1/jobs/${job.id}/artifact`}
              target="_blank"
              rel="noreferrer"
            >
              Download annotated video
            </a>
          )}
        </div>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="bg-slate-50 rounded-md px-4 py-3">
      <div className="text-xs uppercase text-slate-500">{label}</div>
      <div className="text-lg font-semibold">{value}</div>
    </div>
  );
}
