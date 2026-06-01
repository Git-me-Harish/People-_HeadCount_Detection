import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  Area,
  AreaChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, AnalyticsSummary, TimeseriesPoint } from "../api/client";
import { Card, PageHeader, Spinner, KpiCard, Select } from "../components/ui";
import { Activity, TrendingUp, Zap, BarChart2 } from "lucide-react";

const ChartTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div
      className="rounded-lg px-3 py-2.5 text-xs shadow-lg"
      style={{
        backgroundColor: "var(--bg-elevated)",
        border: "1px solid var(--border-base)",
        color: "var(--text-primary)",
      }}
    >
      <p className="font-mono mb-1.5" style={{ color: "var(--text-muted)", fontSize: "10px" }}>{label}</p>
      {payload.map((p: any) => (
        <p key={p.name} className="flex items-center gap-2 font-mono">
          <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: p.color }} />
          <span style={{ color: "var(--text-secondary)" }}>{p.name}:</span>
          <span className="font-medium" style={{ color: "var(--text-primary)" }}>{p.value}</span>
        </p>
      ))}
    </div>
  );
};

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
      { name: "Low",      count: buckets[0] },
      { name: "Med-Low",  count: buckets[1] },
      { name: "Med-High", count: buckets[2] },
      { name: "High",     count: buckets[3] },
    ];
  })();

  const chartData = series.map((p) => ({
    time: new Date(p.timestamp).toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
    }),
    count: p.count,
    peak: p.peak,
  }));

  const densityBarColors = ["var(--success)", "var(--warn)", "#f97316", "var(--danger)"];

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-5">
      <PageHeader
        title="Analytics"
        subtitle="Crowd trends, peak analysis, and density distribution"
      />

      {/* Controls */}
      <Card>
        <div className="flex flex-wrap items-end gap-5">
          <Select
            label="Time window"
            value={String(days)}
            onChange={(e) => setDays(parseInt(e.target.value))}
            options={[
              { value: "1",  label: "Last 24 hours" },
              { value: "7",  label: "Last 7 days" },
              { value: "30", label: "Last 30 days" },
              { value: "90", label: "Last 90 days" },
            ]}
          />
          <Select
            label="Bucket size"
            value={String(bucket)}
            onChange={(e) => setBucket(parseInt(e.target.value))}
            options={[
              { value: "5",    label: "5 minutes" },
              { value: "15",   label: "15 minutes" },
              { value: "60",   label: "1 hour" },
              { value: "240",  label: "4 hours" },
              { value: "1440", label: "1 day" },
            ]}
          />
          {summary && (
            <div
              className="ml-auto flex items-center gap-6 py-2 px-4 rounded-lg text-sm"
              style={{ backgroundColor: "var(--bg-subtle)", border: "1px solid var(--border-base)" }}
            >
              {[
                { label: "Total", value: summary.total_detections.toLocaleString() },
                { label: "Peak",  value: summary.peak_person_count },
                { label: "Avg",   value: Number(summary.average_person_count).toFixed(1) },
              ].map((s) => (
                <div key={s.label} className="text-center">
                  <p
                    className="text-base font-mono font-medium"
                    style={{ color: "var(--text-primary)", letterSpacing: "-0.02em" }}
                  >
                    {s.value}
                  </p>
                  <p className="text-2xs" style={{ color: "var(--text-muted)" }}>{s.label}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      </Card>

      {/* Summary KPIs */}
      {summary && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <KpiCard
            label="Total Detections"
            value={summary.total_detections.toLocaleString()}
            sub={`last ${days} days`}
            icon={<Zap size={15} />}
            color="accent"
          />
          <KpiCard
            label="Peak Count"
            value={summary.peak_person_count}
            sub="highest recorded"
            icon={<TrendingUp size={15} />}
            color="amber"
          />
          <KpiCard
            label="Average Count"
            value={Number(summary.average_person_count).toFixed(1)}
            sub="per detection"
            icon={<Activity size={15} />}
            color="green"
          />
          <KpiCard
            label="Current Count"
            value={summary.current_count ?? 0}
            sub="detected now"
            icon={<BarChart2 size={15} />}
            color="accent"
          />
        </div>
      )}

      {/* Time series chart */}
      <Card padding={false}>
        <div
          className="flex items-center justify-between px-5 py-3.5"
          style={{ borderBottom: "1px solid var(--border-subtle)" }}
        >
          <div>
            <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: "var(--text-muted)" }}>
              People Over Time
            </p>
          </div>
          <div className="flex items-center gap-4 text-2xs" style={{ color: "var(--text-muted)" }}>
            <span className="flex items-center gap-1.5">
              <span className="w-3 h-0.5" style={{ backgroundColor: "var(--accent)" }} />
              Average
            </span>
            <span className="flex items-center gap-1.5">
              <span
                className="w-3 h-0.5"
                style={{ backgroundImage: "repeating-linear-gradient(90deg,var(--warn) 0 4px,transparent 4px 6px)" }}
              />
              Peak
            </span>
          </div>
        </div>
        <div className="p-5">
          <div style={{ height: 280 }}>
            {loading ? (
              <div className="h-full flex items-center justify-center">
                <Spinner />
              </div>
            ) : series.length === 0 ? (
              <div className="h-full flex items-center justify-center text-sm" style={{ color: "var(--text-muted)" }}>
                No data in this window
              </div>
            ) : (
              <ResponsiveContainer>
                <AreaChart data={chartData}>
                  <defs>
                    <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="var(--accent)" stopOpacity={0.18} />
                      <stop offset="95%" stopColor="var(--accent)" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" />
                  <XAxis
                    dataKey="time"
                    tick={{ fontSize: 10, fill: "var(--text-muted)", fontFamily: "DM Mono" }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis
                    tick={{ fontSize: 10, fill: "var(--text-muted)", fontFamily: "DM Mono" }}
                    axisLine={false}
                    tickLine={false}
                    width={36}
                  />
                  <Tooltip content={<ChartTooltip />} />
                  <Area
                    type="monotone"
                    dataKey="count"
                    stroke="var(--accent)"
                    strokeWidth={2}
                    fill="url(#areaGrad)"
                    dot={false}
                    name="avg"
                  />
                  <Line
                    type="monotone"
                    dataKey="peak"
                    stroke="var(--warn)"
                    strokeWidth={1.5}
                    dot={false}
                    name="peak"
                    strokeDasharray="4 2"
                  />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      </Card>

      {/* Density distribution */}
      <Card padding={false}>
        <div
          className="px-5 py-3.5"
          style={{ borderBottom: "1px solid var(--border-subtle)" }}
        >
          <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: "var(--text-muted)" }}>
            Density Distribution
          </p>
          <p className="text-sm font-medium mt-0.5" style={{ color: "var(--text-primary)" }}>
            Time buckets by crowd level
          </p>
        </div>
        <div className="p-5">
          <div style={{ height: 220 }}>
            {sourceDist.length === 0 ? (
              <div className="h-full flex items-center justify-center text-sm" style={{ color: "var(--text-muted)" }}>
                No data
              </div>
            ) : (
              <ResponsiveContainer>
                <BarChart data={sourceDist} barSize={40}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" vertical={false} />
                  <XAxis
                    dataKey="name"
                    tick={{ fontSize: 11, fill: "var(--text-muted)", fontFamily: "DM Sans" }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis
                    tick={{ fontSize: 10, fill: "var(--text-muted)", fontFamily: "DM Mono" }}
                    axisLine={false}
                    tickLine={false}
                    width={32}
                  />
                  <Tooltip content={<ChartTooltip />} />
                  <Bar
                    dataKey="count"
                    name="Buckets"
                    radius={[4, 4, 0, 0]}
                    fill="var(--accent)"
                  />
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      </Card>
    </div>
  );
}