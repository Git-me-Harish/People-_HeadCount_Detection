import { Activity, AlertTriangle, TrendingUp, Users, Zap } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";
import {
  CartesianGrid,
  Tooltip,
  XAxis,
  YAxis,
  Area,
  AreaChart,
  ResponsiveContainer,
} from "recharts";
import { analyticsApi, alertsApi } from "../api/client";
import { KpiCard, Card, PageHeader, Spinner, Badge, DensityDot, StatusPill } from "../components/ui";
import { ROUTES } from "../constants";
import { useAsync } from "../hooks";
import { format } from "date-fns";

// Themed chart tooltip
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
      <p className="font-mono font-medium mb-1.5" style={{ color: "var(--text-muted)" }}>{label}</p>
      {payload.map((p: any) => (
        <p key={p.name} className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full" style={{ backgroundColor: p.color }} />
          <span style={{ color: "var(--text-secondary)" }}>{p.name}:</span>
          <span className="font-mono font-medium" style={{ color: "var(--text-primary)" }}>{p.value}</span>
        </p>
      ))}
    </div>
  );
};

export default function Dashboard() {
  const { t: _t } = useTranslation();

  const { data: summary, loading: sumLoading } = useAsync(() => analyticsApi.summary(7));
  const { data: timeseries, loading: tsLoading } = useAsync(() => analyticsApi.timeseries(7, 60));
  const { data: alerts } = useAsync(() => alertsApi.list());
  const { data: records } = useAsync(() => analyticsApi.records(10));

  const chartData =
    timeseries?.slice(-24).map((p) => ({
      time: format(new Date(p.timestamp), "HH:mm"),
      avg: p.count,
      peak: p.peak,
    })) ?? [];

  const recentAlerts = alerts?.filter((a) => a.last_triggered_at).slice(0, 5) ?? [];

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-5">
      <PageHeader
        title="Operations Overview"
        subtitle="Real-time crowd intelligence dashboard"
        actions={
          <Link
            to={ROUTES.LIVE}
            className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all"
            style={{
              backgroundColor: "var(--success-dim)",
              border: "1px solid var(--success)",
              color: "var(--success)",
            }}
          >
            <span className="w-1.5 h-1.5 rounded-full bg-[var(--success)] animate-pulse" />
            Live Feed
          </Link>
        }
      />

      {/* KPI row */}
      {sumLoading ? (
        <div className="flex justify-center py-12">
          <Spinner />
        </div>
      ) : (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <KpiCard
            label="Current Count"
            value={summary?.current_count ?? 0}
            sub="people detected now"
            icon={<Users size={16} />}
            color="accent"
          />
          <KpiCard
            label="Peak Count"
            value={summary?.peak_person_count ?? 0}
            sub={`last ${summary?.window_days ?? 7} days`}
            icon={<TrendingUp size={16} />}
            color="amber"
          />
          <KpiCard
            label="Avg Count"
            value={summary?.average_person_count?.toFixed(1) ?? "0"}
            sub="per detection"
            icon={<Activity size={16} />}
            color="green"
          />
          <KpiCard
            label="Total Detections"
            value={summary?.total_detections?.toLocaleString() ?? 0}
            sub={`last ${summary?.window_days ?? 7} days`}
            icon={<Zap size={16} />}
            color="red"
          />
        </div>
      )}

      {/* Chart + Alerts row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Trend chart */}
        <Card className="lg:col-span-2" padding={false}>
          <div
            className="flex items-center justify-between px-5 py-3.5"
            style={{ borderBottom: "1px solid var(--border-subtle)" }}
          >
            <div>
              <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: "var(--text-muted)" }}>
                People Count
              </p>
              <p className="text-sm font-medium mt-0.5" style={{ color: "var(--text-primary)" }}>
                24-Hour Trend
              </p>
            </div>
            <div className="flex items-center gap-4 text-2xs" style={{ color: "var(--text-muted)" }}>
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-0.5 rounded-full" style={{ backgroundColor: "var(--accent)" }} />
                Average
              </span>
              <span className="flex items-center gap-1.5">
                <span
                  className="w-3 h-0.5 rounded-full"
                  style={{
                    backgroundImage: `repeating-linear-gradient(90deg, var(--warn) 0 4px, transparent 4px 6px)`,
                  }}
                />
                Peak
              </span>
            </div>
          </div>
          <div className="p-5">
            {tsLoading ? (
              <div className="flex justify-center py-14">
                <Spinner />
              </div>
            ) : chartData.length === 0 ? (
              <div
                className="flex flex-col items-center justify-center py-14 text-sm"
                style={{ color: "var(--text-muted)" }}
              >
                <Activity size={28} className="mb-3 opacity-20" />
                No data yet — start a detection session
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={220}>
                <AreaChart data={chartData}>
                  <defs>
                    <linearGradient id="avgGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="var(--accent)" stopOpacity={0.15} />
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
                    width={32}
                  />
                  <Tooltip content={<ChartTooltip />} />
                  <Area
                    type="monotone"
                    dataKey="avg"
                    stroke="var(--accent)"
                    strokeWidth={2}
                    fill="url(#avgGrad)"
                    dot={false}
                    name="Average"
                  />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </div>
        </Card>

        {/* Recent alerts */}
        <Card padding={false}>
          <div
            className="flex items-center justify-between px-5 py-3.5"
            style={{ borderBottom: "1px solid var(--border-subtle)" }}
          >
            <div>
              <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: "var(--text-muted)" }}>
                Recent Alerts
              </p>
            </div>
            <Link
              to={ROUTES.ALERTS}
              className="text-2xs font-medium transition-colors"
              style={{ color: "var(--accent)" }}
            >
              View all
            </Link>
          </div>
          <div>
            {recentAlerts.length === 0 ? (
              <div
                className="px-5 py-10 text-center text-sm"
                style={{ color: "var(--text-muted)" }}
              >
                No alerts triggered
              </div>
            ) : (
              recentAlerts.map((alert) => (
                <div
                  key={alert.id}
                  className="px-5 py-3 flex items-center gap-3 transition-colors"
                  style={{ borderBottom: "1px solid var(--border-subtle)" }}
                >
                  <div
                    className="w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0"
                    style={{ backgroundColor: "var(--warn-dim)" }}
                  >
                    <AlertTriangle size={13} style={{ color: "var(--warn)" }} />
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="text-xs font-medium truncate" style={{ color: "var(--text-primary)" }}>
                      {alert.name}
                    </p>
                    <p className="text-2xs mt-0.5 font-mono" style={{ color: "var(--text-muted)" }}>
                      threshold: {alert.threshold}
                    </p>
                  </div>
                  <StatusPill active={alert.is_enabled} labelOn="On" labelOff="Off" />
                </div>
              ))
            )}
          </div>
        </Card>
      </div>

      {/* Recent detections table */}
      <Card padding={false}>
        <div
          className="flex items-center justify-between px-5 py-3.5"
          style={{ borderBottom: "1px solid var(--border-subtle)" }}
        >
          <div>
            <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: "var(--text-muted)" }}>
              Recent Detections
            </p>
          </div>
          <Link
            to={ROUTES.ANALYTICS}
            className="text-2xs font-medium transition-colors"
            style={{ color: "var(--accent)" }}
          >
            Full analytics
          </Link>
        </div>

        {/* Table header */}
        <div
          className="grid grid-cols-4 gap-4 px-5 py-2 text-2xs font-semibold uppercase tracking-wider"
          style={{ color: "var(--text-muted)", borderBottom: "1px solid var(--border-subtle)" }}
        >
          <span>Density</span>
          <span>Count</span>
          <span>Source</span>
          <span className="text-right">Time</span>
        </div>

        {!records || records.length === 0 ? (
          <div className="px-5 py-10 text-center text-sm" style={{ color: "var(--text-muted)" }}>
            No detections recorded yet
          </div>
        ) : (
          records.slice(0, 8).map((r) => (
            <div
              key={r.id}
              className="grid grid-cols-4 gap-4 px-5 py-3 items-center transition-colors"
              style={{ borderBottom: "1px solid var(--border-subtle)" }}
              onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "var(--bg-subtle)")}
              onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
            >
              <div className="flex items-center gap-2">
                <DensityDot count={r.person_count} />
                <span
                  className="text-2xs font-medium"
                  style={{
                    color:
                      r.person_count >= 200
                        ? "var(--danger)"
                        : r.person_count >= 100
                        ? "#f97316"
                        : r.person_count >= 50
                        ? "var(--warn)"
                        : "var(--success)",
                  }}
                >
                  {r.person_count >= 200
                    ? "Critical"
                    : r.person_count >= 100
                    ? "High"
                    : r.person_count >= 50
                    ? "Medium"
                    : "Low"}
                </span>
              </div>
              <span className="text-sm font-mono font-medium" style={{ color: "var(--text-primary)" }}>
                {r.person_count}
              </span>
              <Badge color="gray">{r.source}</Badge>
              <span className="text-2xs text-right font-mono" style={{ color: "var(--text-muted)" }}>
                {format(new Date(r.created_at), "MMM d, HH:mm")}
              </span>
            </div>
          ))
        )}
      </Card>
    </div>
  );
}