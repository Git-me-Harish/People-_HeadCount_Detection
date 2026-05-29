import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, AnalyticsSummary, TimeseriesPoint } from "../api/client";

export default function Analytics() {
  const [days, setDays] = useState(7);
  const [bucket, setBucket] = useState(60);
  const [series, setSeries] = useState<TimeseriesPoint[]>([]);
  const [summary, setSummary] = useState<AnalyticsSummary | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      try {
        const [s, t] = await Promise.all([
          api.get(`/analytics/summary?days=${days}`),
          api.get(`/analytics/timeseries?days=${days}&bucket_minutes=${bucket}`),
        ]);
        setSummary(s.data);
        setSeries(t.data);
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [days, bucket]);

  const sourceDist: { name: string; count: number }[] = (() => {
    if (series.length === 0) return [];
    const max = series.reduce((m, p) => Math.max(m, p.peak), 0);
    const buckets = [0, 0, 0, 0];
    series.forEach((p) => {
      const r = Math.min(3, Math.floor((p.count / (max || 1)) * 4));
      buckets[r] += 1;
    });
    return [
      { name: "Low", count: buckets[0] },
      { name: "Med-low", count: buckets[1] },
      { name: "Med-high", count: buckets[2] },
      { name: "High", count: buckets[3] },
    ];
  })();

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Analytics</h1>
        <p className="text-sm text-slate-500">Trends, peaks, and density distribution.</p>
      </div>
      <div className="card flex flex-wrap gap-6 items-center">
        <div>
          <label className="label">Window (days)</label>
          <select
            className="input"
            value={days}
            onChange={(e) => setDays(parseInt(e.target.value))}
          >
            <option value={1}>Last 24 hours</option>
            <option value={7}>Last 7 days</option>
            <option value={30}>Last 30 days</option>
            <option value={90}>Last 90 days</option>
          </select>
        </div>
        <div>
          <label className="label">Bucket size</label>
          <select
            className="input"
            value={bucket}
            onChange={(e) => setBucket(parseInt(e.target.value))}
          >
            <option value={5}>5 min</option>
            <option value={15}>15 min</option>
            <option value={60}>1 hour</option>
            <option value={240}>4 hours</option>
            <option value={1440}>1 day</option>
          </select>
        </div>
        {summary && (
          <div className="ml-auto text-sm grid grid-cols-3 gap-x-8 gap-y-1">
            <span className="text-slate-500">Detections</span>
            <span className="text-slate-500">Peak</span>
            <span className="text-slate-500">Average</span>
            <span className="font-semibold">{summary.total_detections}</span>
            <span className="font-semibold">{summary.peak_person_count}</span>
            <span className="font-semibold">{summary.average_person_count}</span>
          </div>
        )}
      </div>

      <div className="card">
        <h2 className="font-semibold mb-3">People over time</h2>
        <div className="h-72">
          {series.length === 0 ? (
            <div className="h-full flex items-center justify-center text-slate-400">
              {loading ? "Loading…" : "No data in this window"}
            </div>
          ) : (
            <ResponsiveContainer>
              <LineChart data={series}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis
                  dataKey="timestamp"
                  tickFormatter={(t) =>
                    new Date(t).toLocaleString(undefined, {
                      month: "short",
                      day: "numeric",
                      hour: "2-digit",
                    })
                  }
                  fontSize={11}
                />
                <YAxis fontSize={11} />
                <Tooltip
                  labelFormatter={(t) => new Date(t as string).toLocaleString()}
                />
                <Legend />
                <Line type="monotone" dataKey="count" stroke="#2563eb" name="avg" strokeWidth={2} />
                <Line type="monotone" dataKey="peak" stroke="#f59e0b" name="peak" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      <div className="card">
        <h2 className="font-semibold mb-3">Density distribution</h2>
        <div className="h-56">
          {sourceDist.length === 0 ? (
            <div className="h-full flex items-center justify-center text-slate-400">
              No data
            </div>
          ) : (
            <ResponsiveContainer>
              <BarChart data={sourceDist}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#2563eb" />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>
    </div>
  );
}
