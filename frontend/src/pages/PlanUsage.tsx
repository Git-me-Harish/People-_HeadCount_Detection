import { Zap, CheckCircle2, XCircle } from "lucide-react";
import { planApi } from "../api/client";
import { Card, PageHeader, Spinner, Badge, Button } from "../components/ui";
import { useAsync } from "../hooks";

function UsageBar({
  used,
  max,
  label,
}: {
  used: number;
  max: number;
  label: string;
}) {
  const isUnlimited = max === -1;
  const pct = isUnlimited ? 0 : Math.min(100, (used / max) * 100);

  const barColor =
    pct > 90
      ? "var(--danger)"
      : pct > 70
      ? "var(--warn)"
      : "var(--accent)";

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-sm" style={{ color: "var(--text-secondary)" }}>
          {label}
        </span>
        <span className="text-sm font-mono font-medium" style={{ color: "var(--text-primary)" }}>
          {used.toLocaleString()}
          <span style={{ color: "var(--text-muted)" }}>
            {" / "}
            {isUnlimited ? "Unlimited" : max.toLocaleString()}
          </span>
        </span>
      </div>
      <div
        className="h-1.5 rounded-full overflow-hidden"
        style={{ backgroundColor: "var(--bg-muted)" }}
      >
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{
            width: isUnlimited ? "0%" : `${pct}%`,
            backgroundColor: barColor,
          }}
        />
      </div>
      {pct > 80 && !isUnlimited && (
        <p className="text-2xs" style={{ color: pct > 90 ? "var(--danger)" : "var(--warn)" }}>
          {Math.round(pct)}% of limit used
        </p>
      )}
    </div>
  );
}

const TIER_BADGE: Record<string, string> = {
  free:       "gray",
  pro:        "accent",
  enterprise: "blue",
};

export default function PlanUsage() {
  const { data, loading } = useAsync(() => planApi.get());

  if (loading || !data) {
    return (
      <div className="flex justify-center py-20">
        <Spinner />
      </div>
    );
  }

  const { plan, usage } = data;
  const isFree = plan.tier === "free";

  return (
    <div className="p-6 max-w-3xl mx-auto space-y-5">
      <PageHeader title="Plan & Usage" subtitle="Current subscription and monthly consumption" />

      {/* Plan summary */}
      <Card>
        <div className="flex items-center justify-between mb-5">
          <div className="flex items-center gap-3">
            <div
              className="w-10 h-10 rounded-xl flex items-center justify-center"
              style={{ backgroundColor: "var(--accent-dim)" }}
            >
              <Zap size={18} style={{ color: "var(--accent)" }} />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-semibold text-base" style={{ color: "var(--text-primary)" }}>
                  {plan.display_name}
                </span>
                <Badge color={TIER_BADGE[plan.tier] ?? "gray"}>
                  {plan.tier.toUpperCase()}
                </Badge>
              </div>
              <p className="text-xs mt-0.5" style={{ color: "var(--text-muted)" }}>
                {plan.price_usd_monthly === 0
                  ? "Free forever"
                  : `$${(plan.price_usd_monthly / 100).toFixed(2)} / month`}
              </p>
            </div>
          </div>
          {isFree && (
            <Button size="sm" icon={<Zap size={13} />}>
              Upgrade to Pro
            </Button>
          )}
        </div>

        <div
          className="rounded-xl p-4 space-y-4"
          style={{ backgroundColor: "var(--bg-subtle)", border: "1px solid var(--border-base)" }}
        >
          <UsageBar
            used={usage.cameras_used}
            max={plan.max_cameras}
            label="Cameras"
          />
          <UsageBar
            used={usage.frames_processed_month}
            max={plan.max_frames_per_month}
            label="Frames this month"
          />
          <UsageBar
            used={usage.alerts_sent_month}
            max={plan.max_alerts}
            label="Alert rules"
          />
        </div>
      </Card>

      {/* Feature matrix */}
      <Card padding={false}>
        <div
          className="px-5 py-3.5"
          style={{ borderBottom: "1px solid var(--border-subtle)" }}
        >
          <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: "var(--text-muted)" }}>
            Plan Features
          </p>
        </div>
        <div className="p-5">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {[
              { label: "PDF report exports",          enabled: plan.can_export_pdf },
              { label: "Public status page",          enabled: plan.can_use_public_page },
              { label: `${plan.retention_days}-day data retention`, enabled: true },
              { label: `${plan.max_api_tokens} API token${plan.max_api_tokens !== 1 ? "s" : ""}`, enabled: true },
              { label: "Slack & Teams alerts",        enabled: plan.tier !== "free" },
              { label: "Priority support",            enabled: plan.tier === "enterprise" },
              { label: "SSO / SAML",                  enabled: plan.tier === "enterprise" },
              { label: "On-premise deployment",       enabled: plan.tier === "enterprise" },
            ].map((f) => (
              <div key={f.label} className="flex items-center gap-2.5">
                {f.enabled ? (
                  <CheckCircle2 size={14} style={{ color: "var(--success)", flexShrink: 0 }} />
                ) : (
                  <XCircle size={14} style={{ color: "var(--border-base)", flexShrink: 0 }} />
                )}
                <span
                  className="text-sm"
                  style={{
                    color: f.enabled ? "var(--text-secondary)" : "var(--text-muted)",
                    textDecoration: f.enabled ? "none" : "line-through",
                  }}
                >
                  {f.label}
                </span>
              </div>
            ))}
          </div>
        </div>
      </Card>

      {/* Upgrade CTA for free tier */}
      {isFree && (
        <div
          className="rounded-xl p-5 flex items-center justify-between gap-4"
          style={{
            backgroundColor: "var(--accent-dim)",
            border: "1px solid rgba(0,194,168,0.3)",
          }}
        >
          <div>
            <p className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
              Scale with Pro
            </p>
            <p className="text-xs mt-0.5" style={{ color: "var(--text-muted)" }}>
              25 cameras, 50 alerts, PDF reports, Slack/Teams — from $29/month.
            </p>
          </div>
          <Button size="sm" className="flex-shrink-0">
            Upgrade now
          </Button>
        </div>
      )}
    </div>
  );
}