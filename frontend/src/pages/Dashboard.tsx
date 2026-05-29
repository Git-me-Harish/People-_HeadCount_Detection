import { useEffect, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, AnalyticsSummary, DetectionRecord, TimeseriesPoint } from "../api/client";

export default function Dashboard() {
  const [summary, setSummary] = useState<AnalyticsSummary | null>(null);
  const [series, setSeries] = useState<TimeseriesPoint[]>([]);
  const [records, setRecords] = useState<DetectionRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [detectorStatus, setDetectorStatus] = useState<{ ready: boolean; error?: string | null } | null>(
    null,
  );

  useEffect(() => {
    let cancelled = false;
    const fetchAll = async () => {
      try {
        const [s, t, r, d] = await Promise.all([
          api.get("/analytics/summary"),
          api.get("/analytics/timeseries?days=7&bucket_minutes=60"),
          api.get("/analytics/records?limit=8"),
          api.get("/detect/status"),
        ]);
        if (cancelled) return;
        setSummary(s.data);
        setSeries(t.data);
        setRecords(r.data);
        setDetectorStatus(d.data);
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    fetchAll();
    const id = setInterval(fetchAll, 15000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Dashboard</h1>
          <p className="text-sm text-slate-500">Operational view of crowd analytics across your sites.</p>
        </div>
        {detectorStatus && (
          <div
            className={`text-xs px-3 py-1.5 rounded-full font-medium ${
              detectorStatus.ready
                ? "bg-emerald-50 text-emerald-700 border border-emerald-200"
                : "bg-amber-50 text-amber-700 border border-amber-200"
            }`}
          >
            Detector: {detectorStatus.ready ? "online" : detectorStatus.error ?? "offline"}
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Kpi label="Current count" value={summary?.current_count ?? 0} loading={loading} accent="brand" />
        <Kpi
          label="7-day peak"
          value={summary?.peak_person_count ?? 0}
          loading={loading}
          accent="amber"
        />
        <Kpi
          label="7-day average"
          value={summary?.average_person_count ?? 0}
          loading={loading}
          accent="emerald"
        />
        <Kpi
          label="Total detections"
          value={summary?.total_detections ?? 0}
          loading={loading}
          accent="violet"
        />
      </div>

      <div className="card">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h2 className="font-semibold">Last 7 days — people detected</h2>
            <p className="text-xs text-slate-500">Hourly averages with peak markers</p>
          </div>
        </div>
        <div className="h-64">
          {series.length === 0 ? (
            <EmptyChart />
          ) : (
            <ResponsiveContainer>
              <AreaChart data={series}>
                <defs>
                  <linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#2563eb" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#2563eb" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis
                  dataKey="timestamp"
                  tickFormatter={(t) => new Date(t).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit" })}
                  fontSize={11}
                />
                <YAxis fontSize={11} />
                <Tooltip
                  labelFormatter={(t) => new Date(t as string).toLocaleString()}
                  formatter={(value: number, name: string) => [value, name === "count" ? "avg" : name]}
                />
                <Area type="monotone" dataKey="count" stroke="#2563eb" strokeWidth={2} fill="url(#g)" />
                <Area type="monotone" dataKey="peak" stroke="#f59e0b" strokeWidth={1} fillOpacity={0} />
              </AreaChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      <div className="card">
        <h2 className="font-semibold mb-3">Recent detections</h2>
        <table className="w-full text-sm">
          <thead className="text-xs text-slate-500 uppercase">
            <tr>
              <th className="text-left py-2">When</th>
              <th className="text-left py-2">Source</th>
              <th className="text-left py-2">Camera</th>
              <th className="text-right py-2">People</th>
              <th className="text-right py-2">Confidence</th>
            </tr>
          </thead>
          <tbody>
            {records.length === 0 && (
              <tr>
                <td colSpan={5} className="text-center text-slate-400 py-6">
                  No detections yet — upload an image or start a stream.
                </td>
              </tr>
            )}
            {records.map((r) => (
              <tr key={r.id} className="border-t border-slate-100">
                <td className="py-2">{new Date(r.created_at).toLocaleString()}</td>
                <td>{r.source}</td>
                <td>{r.camera_id ?? "—"}</td>
                <td className="text-right font-medium">{r.person_count}</td>
                <td className="text-right">
                  {r.avg_confidence != null ? r.avg_confidence.toFixed(2) : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Kpi({
  label,
  value,
  loading,
  accent,
}: {
  label: string;
  value: number;
  loading: boolean;
  accent: "brand" | "amber" | "emerald" | "violet";
}) {
  const accents: Record<string, string> = {
    brand: "from-brand-50 to-brand-100 text-brand-700",
    amber: "from-amber-50 to-amber-100 text-amber-700",
    emerald: "from-emerald-50 to-emerald-100 text-emerald-700",
    violet: "from-violet-50 to-violet-100 text-violet-700",
  };
  return (
    <div className={`card bg-gradient-to-br ${accents[accent]}`}>
      <div className="text-xs uppercase tracking-wide opacity-80">{label}</div>
      <div className="text-3xl font-semibold mt-1">{loading ? "…" : value}</div>
    </div>
  );
}

function EmptyChart() {
  return (
    <div className="h-full flex items-center justify-center text-sm text-slate-400">
      No data yet
    </div>
  );
}
