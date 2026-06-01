import { useEffect, useState } from "react";
import { AlertTriangle, Plus, Trash2, Webhook, Clock } from "lucide-react";
import { alertsApi, camerasApi } from "../api/client";
import type { Alert as AlertModel, Camera } from "../types";
import {
  Button,
  Card,
  PageHeader,
  Modal,
  Input,
  Select,
  StatusPill,
  EmptyState,
  AlertBanner,
} from "../components/ui";
import { format } from "date-fns";

interface AlertForm {
  name: string;
  threshold: number;
  cooldown_minutes: number;
  camera_id: string; // "all" | stringified number — converts on submit
  webhook_url: string;
}

const DEFAULT_FORM: AlertForm = {
  name: "",
  threshold: 10,
  cooldown_minutes: 10,
  camera_id: "all",
  webhook_url: "",
};

export default function Alerts() {
  const [alerts, setAlerts]     = useState<AlertModel[]>([]);
  const [cameras, setCameras]   = useState<Camera[]>([]);
  const [form, setForm]         = useState<AlertForm>(DEFAULT_FORM);
  const [showForm, setShowForm] = useState(false);
  const [error, setError]       = useState<string | null>(null);
  const [deleting, setDeleting] = useState<number | null>(null);

  const load = async () => {
    const [alertsRes, camerasRes] = await Promise.all([
      alertsApi.list(),
      camerasApi.list(),
    ]);
    setAlerts(alertsRes.data);
    setCameras(camerasRes.data);
  };

  useEffect(() => { load(); }, []);

  const cameraOptions = [
    { value: "all", label: "All cameras (global)" },
    ...cameras.map((c) => ({ value: String(c.id), label: c.name + (c.location ? ` — ${c.location}` : "") })),
  ];

  const closeForm = () => {
    setShowForm(false);
    setError(null);
    setForm(DEFAULT_FORM);
  };

  const submit = async () => {
    setError(null);
    try {
      await alertsApi.create({
        name: form.name,
        threshold: form.threshold,
        cooldown_minutes: form.cooldown_minutes,
        camera_id: form.camera_id === "all" ? null : Number(form.camera_id),
        webhook_url: form.webhook_url.trim() || null,
      });
      closeForm();
      load();
    } catch (err: any) {
      setError(err?.response?.data?.detail ?? "Failed to create alert.");
    }
  };

  const toggle = async (a: AlertModel) => {
    await alertsApi.update(a.id, { is_enabled: !a.is_enabled });
    load();
  };

  const remove = async (id: number) => {
    if (!confirm("Permanently delete this alert?")) return;
    setDeleting(id);
    try {
      await alertsApi.delete(id);
      load();
    } finally {
      setDeleting(null);
    }
  };

  const cameraName = (id: number | null) => {
    if (id === null) return "All cameras";
    return cameras.find((c) => c.id === id)?.name ?? `Camera ${id}`;
  };

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-5">
      <PageHeader
        title="Alerts"
        subtitle="Trigger notifications when people count exceeds thresholds"
        actions={
          <Button icon={<Plus size={14} />} onClick={() => setShowForm(true)} size="sm">
            New alert
          </Button>
        }
      />

      <Card padding={false}>
        {/* Table header */}
        <div
          className="grid items-center px-5 py-3 text-2xs font-semibold uppercase tracking-wider"
          style={{
            gridTemplateColumns: "2fr 100px 110px 140px 1fr 80px 80px",
            borderBottom: "1px solid var(--border-subtle)",
            color: "var(--text-muted)",
          }}
        >
          <span>Alert name</span>
          <span>Threshold</span>
          <span>Cooldown</span>
          <span>Camera scope</span>
          <span>Last triggered</span>
          <span>Status</span>
          <span />
        </div>

        {alerts.length === 0 ? (
          <EmptyState
            icon={<AlertTriangle size={28} />}
            title="No alerts configured"
            description="Create an alert to receive notifications when crowd density exceeds your threshold."
            action={
              <Button size="sm" icon={<Plus size={13} />} onClick={() => setShowForm(true)}>
                New alert
              </Button>
            }
          />
        ) : (
          alerts.map((a) => (
            <div
              key={a.id}
              className="grid items-center px-5 py-3.5 transition-colors"
              style={{
                gridTemplateColumns: "2fr 100px 110px 140px 1fr 80px 80px",
                borderBottom: "1px solid var(--border-subtle)",
              }}
              onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "var(--bg-subtle)")}
              onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
            >
              {/* Name */}
              <div className="flex items-center gap-2.5 min-w-0">
                <div
                  className="w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0"
                  style={{ backgroundColor: a.is_enabled ? "var(--warn-dim)" : "var(--bg-subtle)" }}
                >
                  <AlertTriangle
                    size={13}
                    style={{ color: a.is_enabled ? "var(--warn)" : "var(--text-muted)" }}
                  />
                </div>
                <span className="text-sm font-medium truncate" style={{ color: "var(--text-primary)" }}>
                  {a.name}
                </span>
              </div>

              {/* Threshold */}
              <span className="text-sm font-mono font-medium" style={{ color: "var(--text-primary)" }}>
                {a.threshold}{" "}
                <span className="text-2xs font-normal" style={{ color: "var(--text-muted)" }}>
                  people
                </span>
              </span>

              {/* Cooldown */}
              <div className="flex items-center gap-1" style={{ color: "var(--text-muted)" }}>
                <Clock size={11} />
                <span className="text-2xs font-mono">{a.cooldown_minutes}m</span>
              </div>

              {/* Camera scope */}
              <span
                className="text-2xs truncate"
                style={{ color: "var(--text-secondary)" }}
                title={cameraName(a.camera_id)}
              >
                {cameraName(a.camera_id)}
              </span>

              {/* Last triggered */}
              <span className="text-2xs font-mono" style={{ color: "var(--text-muted)" }}>
                {a.last_triggered_at
                  ? format(new Date(a.last_triggered_at), "MMM d, HH:mm")
                  : "—"}
              </span>

              {/* Toggle */}
              <div onClick={() => toggle(a)} className="cursor-pointer">
                <StatusPill active={a.is_enabled} />
              </div>

              {/* Delete */}
              <div className="flex justify-end">
                <button
                  onClick={() => remove(a.id)}
                  disabled={deleting === a.id}
                  className="p-1.5 rounded-md transition-colors disabled:opacity-40"
                  style={{ color: "var(--text-muted)" }}
                  onMouseEnter={(e) => {
                    (e.currentTarget as HTMLElement).style.color = "var(--danger)";
                    (e.currentTarget as HTMLElement).style.backgroundColor = "var(--danger-dim)";
                  }}
                  onMouseLeave={(e) => {
                    (e.currentTarget as HTMLElement).style.color = "var(--text-muted)";
                    (e.currentTarget as HTMLElement).style.backgroundColor = "transparent";
                  }}
                  title="Delete alert"
                >
                  <Trash2 size={13} />
                </button>
              </div>
            </div>
          ))
        )}
      </Card>

      {/* Create alert modal */}
      <Modal
        open={showForm}
        onClose={closeForm}
        title="New Alert"
        footer={
          <>
            <Button variant="secondary" size="sm" onClick={closeForm}>
              Cancel
            </Button>
            <Button size="sm" onClick={submit} disabled={!form.name || form.threshold < 1}>
              Create alert
            </Button>
          </>
        }
      >
        <div className="space-y-4">
          {error && <AlertBanner type="error">{error}</AlertBanner>}

          <Input
            label="Alert name"
            value={form.name}
            onChange={(e) => setForm({ ...form, name: e.target.value })}
            placeholder="e.g. Gate A overcrowding"
          />

          <div className="grid grid-cols-2 gap-3">
            <Input
              label="Threshold (people)"
              type="number"
              min={1}
              value={String(form.threshold)}
              onChange={(e) => setForm({ ...form, threshold: Math.max(1, parseInt(e.target.value) || 1) })}
              helper="Fires when count exceeds this"
            />
            <Input
              label="Cooldown (minutes)"
              type="number"
              min={1}
              max={1440}
              value={String(form.cooldown_minutes)}
              onChange={(e) =>
                setForm({ ...form, cooldown_minutes: Math.max(1, parseInt(e.target.value) || 1) })
              }
              helper="Min gap between firings"
            />
          </div>

          <Select
            label="Camera scope"
            value={form.camera_id}
            options={cameraOptions}
            onChange={(e) => setForm({ ...form, camera_id: e.target.value })}
          />

          <Input
            label="Webhook URL"
            value={form.webhook_url}
            onChange={(e) => setForm({ ...form, webhook_url: e.target.value })}
            placeholder="https://hooks.slack.com/..."
            helper="Optional — receives a POST with detection payload"
          />
        </div>
      </Modal>
    </div>
  );
}