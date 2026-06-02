import { useState } from "react";
import { Zap, CheckCircle2, XCircle, ArrowRight, Sparkles } from "lucide-react";
import { planApi } from "../api/client";
import { Card, PageHeader, Spinner, Badge, Button, Modal, AlertBanner } from "../components/ui";
import { useAsync } from "../hooks";

// ── Usage bar ─────────────────────────────────────────────────────────────────

function UsageBar({ used, max, label }: { used: number; max: number; label: string }) {
  const isUnlimited = max === -1;
  const pct = isUnlimited ? 0 : Math.min(100, (used / max) * 100);
  const barColor = pct > 90 ? "var(--danger)" : pct > 70 ? "var(--warn)" : "var(--accent)";
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-sm" style={{ color: "var(--text-secondary)" }}>{label}</span>
        <span className="text-sm font-mono font-medium" style={{ color: "var(--text-primary)" }}>
          {used.toLocaleString()}
          <span style={{ color: "var(--text-muted)" }}>
            {" / "}{isUnlimited ? "Unlimited" : max.toLocaleString()}
          </span>
        </span>
      </div>
      <div className="h-1.5 rounded-full overflow-hidden" style={{ backgroundColor: "var(--bg-muted)" }}>
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: isUnlimited ? "0%" : `${pct}%`, backgroundColor: barColor }}
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

// ── Upgrade modal ─────────────────────────────────────────────────────────────

interface UpgradeTier {
  tier: string;
  display_name: string;
  price_usd_monthly: number | null;
  cameras: number;
  alerts: number;
  highlights: string[];
}

function UpgradeModal({
  open,
  onClose,
  onSuccess,
}: {
  open: boolean;
  onClose: () => void;
  onSuccess: () => void;
}) {
  const { data: opts, loading: loadingOpts } = useAsync(() => planApi.getUpgradeOptions(), [], { enabled: open });
  const [selected, setSelected] = useState<string | null>(null);
  const [upgrading, setUpgrading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);

  const handleUpgrade = async () => {
    if (!selected) return;
    setUpgrading(true);
    setError(null);
    try {
      await planApi.upgrade(selected);
      setDone(true);
      setTimeout(() => {
        onSuccess();
        onClose();
        setDone(false);
        setSelected(null);
      }, 1600);
    } catch (err: any) {
      setError(err?.response?.data?.detail ?? "Upgrade failed — please try again.");
    } finally {
      setUpgrading(false);
    }
  };

  const upgrades: UpgradeTier[] = opts?.available_upgrades ?? [];

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="Upgrade your plan"
      footer={
        <>
          <Button variant="secondary" size="sm" onClick={onClose} disabled={upgrading}>Cancel</Button>
          <Button
            size="sm"
            icon={done ? <CheckCircle2 size={13} /> : <Zap size={13} />}
            onClick={handleUpgrade}
            disabled={!selected || upgrading || done}
          >
            {done ? "Upgraded!" : upgrading ? "Upgrading…" : `Upgrade to ${selected ?? "…"}`}
          </Button>
        </>
      }
    >
      <div className="space-y-4">
        {error && <AlertBanner type="error">{error}</AlertBanner>}

        {loadingOpts ? (
          <div className="flex justify-center py-8"><Spinner /></div>
        ) : upgrades.length === 0 ? (
          <p className="text-sm text-center py-6" style={{ color: "var(--text-muted)" }}>
            You're already on the highest available plan.
          </p>
        ) : (
          <div className="space-y-3">
            {upgrades.map((tier) => {
              const isSelected = selected === tier.tier;
              const isEnterprise = tier.price_usd_monthly === null;
              return (
                <div
                  key={tier.tier}
                  onClick={() => !isEnterprise && setSelected(tier.tier)}
                  className="rounded-xl p-4 transition-all duration-150"
                  style={{
                    border: isSelected
                      ? "1.5px solid var(--accent)"
                      : "1px solid var(--border-base)",
                    backgroundColor: isSelected ? "var(--accent-dim)" : "var(--bg-subtle)",
                    cursor: isEnterprise ? "default" : "pointer",
                    opacity: isEnterprise ? 0.7 : 1,
                  }}
                >
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="font-semibold text-sm" style={{ color: "var(--text-primary)" }}>
                          {tier.display_name}
                        </span>
                        {tier.tier === "pro" && (
                          <span
                            className="text-2xs px-1.5 py-0.5 rounded font-semibold uppercase tracking-wide"
                            style={{ backgroundColor: "var(--accent)", color: "var(--text-inverse)" }}
                          >
                            Popular
                          </span>
                        )}
                      </div>
                      <p className="text-xs mt-0.5" style={{ color: "var(--text-muted)" }}>
                        {tier.cameras === -1 ? "Unlimited" : tier.cameras} cameras
                        {" · "}
                        {tier.alerts === -1 ? "Unlimited" : tier.alerts} alerts
                      </p>
                    </div>
                    <div className="text-right">
                      {isEnterprise ? (
                        <p className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
                          Custom
                        </p>
                      ) : (
                        <>
                          <p className="font-mono font-bold text-lg" style={{ color: "var(--text-primary)", letterSpacing: "-0.03em" }}>
                            ${(tier.price_usd_monthly! / 100).toFixed(0)}
                          </p>
                          <p className="text-2xs" style={{ color: "var(--text-muted)" }}>/month</p>
                        </>
                      )}
                    </div>
                  </div>
                  <ul className="space-y-1.5">
                    {tier.highlights.map((h) => (
                      <li key={h} className="flex items-center gap-2 text-xs" style={{ color: "var(--text-secondary)" }}>
                        <CheckCircle2 size={11} style={{ color: "var(--accent)", flexShrink: 0 }} />
                        {h}
                      </li>
                    ))}
                  </ul>
                  {isEnterprise && (
                    <a
                      href="mailto:sales@peoplesense.app"
                      className="mt-3 flex items-center gap-1.5 text-xs font-medium"
                      style={{ color: "var(--accent)" }}
                      onClick={(e) => e.stopPropagation()}
                    >
                      Contact sales <ArrowRight size={11} />
                    </a>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </Modal>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

const TIER_BADGE: Record<string, string> = {
  free: "gray", pro: "accent", enterprise: "blue",
};

export default function PlanUsage() {
  const { data, loading, refetch } = useAsync(() => planApi.get());
  const [showUpgrade, setShowUpgrade] = useState(false);

  if (loading || !data) {
    return <div className="flex justify-center py-20"><Spinner /></div>;
  }

  const { plan, usage } = data;
  const isMaxTier = plan.tier === "enterprise";

  return (
    <div className="p-6 max-w-3xl mx-auto space-y-5">
      <PageHeader title="Plan & Usage" subtitle="Current subscription and monthly consumption" />

      {/* Plan summary */}
      <Card>
        <div className="flex items-center justify-between mb-5">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{ backgroundColor: "var(--accent-dim)" }}>
              <Zap size={18} style={{ color: "var(--accent)" }} />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-semibold text-base" style={{ color: "var(--text-primary)" }}>
                  {plan.display_name}
                </span>
                <Badge color={TIER_BADGE[plan.tier] ?? "gray"}>{plan.tier.toUpperCase()}</Badge>
              </div>
              <p className="text-xs mt-0.5" style={{ color: "var(--text-muted)" }}>
                {plan.price_usd_monthly === 0 ? "Free forever" : `$${(plan.price_usd_monthly / 100).toFixed(2)} / month`}
              </p>
            </div>
          </div>
          {!isMaxTier && (
            <Button size="sm" icon={<Zap size={13} />} onClick={() => setShowUpgrade(true)}>
              Upgrade
            </Button>
          )}
        </div>

        <div className="rounded-xl p-4 space-y-4" style={{ backgroundColor: "var(--bg-subtle)", border: "1px solid var(--border-base)" }}>
          <UsageBar used={usage.cameras_used} max={plan.max_cameras} label="Cameras" />
          <UsageBar used={usage.frames_processed_month} max={plan.max_frames_per_month} label="Frames this month" />
          <UsageBar used={usage.alerts_sent_month} max={plan.max_alerts} label="Alert rules" />
        </div>
      </Card>

      {/* Feature matrix */}
      <Card padding={false}>
        <div className="px-5 py-3.5" style={{ borderBottom: "1px solid var(--border-subtle)" }}>
          <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: "var(--text-muted)" }}>
            Plan Features
          </p>
        </div>
        <div className="p-5">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {[
              { label: "PDF report exports",           enabled: plan.can_export_pdf },
              { label: "Public status page",           enabled: plan.can_use_public_page },
              { label: `${plan.retention_days}-day data retention`, enabled: true },
              { label: `${plan.max_api_tokens} API token${plan.max_api_tokens !== 1 ? "s" : ""}`, enabled: true },
              { label: "Slack & Teams alerts",         enabled: plan.tier !== "free" },
              { label: "Priority support",             enabled: plan.tier === "enterprise" },
              { label: "SSO / SAML",                   enabled: plan.tier === "enterprise" },
              { label: "On-premise deployment",        enabled: plan.tier === "enterprise" },
            ].map((f) => (
              <div key={f.label} className="flex items-center gap-2.5">
                {f.enabled
                  ? <CheckCircle2 size={14} style={{ color: "var(--success)", flexShrink: 0 }} />
                  : <XCircle     size={14} style={{ color: "var(--border-base)", flexShrink: 0 }} />}
                <span
                  className="text-sm"
                  style={{ color: f.enabled ? "var(--text-secondary)" : "var(--text-muted)", textDecoration: f.enabled ? "none" : "line-through" }}
                >
                  {f.label}
                </span>
              </div>
            ))}
          </div>
        </div>
      </Card>

      {/* Upgrade CTA banner — shown when not max tier */}
      {!isMaxTier && (
        <div
          className="rounded-xl p-5 flex items-center justify-between gap-4"
          style={{ backgroundColor: "var(--accent-dim)", border: "1px solid rgba(0,194,168,0.3)" }}
        >
          <div className="flex items-center gap-3">
            <Sparkles size={18} style={{ color: "var(--accent)", flexShrink: 0 }} />
            <div>
              <p className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
                {plan.tier === "free" ? "Scale with Pro" : "Go Enterprise"}
              </p>
              <p className="text-xs mt-0.5" style={{ color: "var(--text-muted)" }}>
                {plan.tier === "free"
                  ? "25 cameras, 50 alerts, PDF reports, Slack/Teams — from $29/month."
                  : "Unlimited cameras, dedicated infra, SLA, SSO, and on-premise options."}
              </p>
            </div>
          </div>
          <Button size="sm" className="flex-shrink-0" onClick={() => setShowUpgrade(true)}>
            Upgrade now
          </Button>
        </div>
      )}

      <UpgradeModal
        open={showUpgrade}
        onClose={() => setShowUpgrade(false)}
        onSuccess={refetch}
      />
    </div>
  );
}