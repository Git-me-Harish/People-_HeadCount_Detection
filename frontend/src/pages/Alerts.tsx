import { useEffect, useState } from "react";
import { api, Alert as AlertModel } from "../api/client";

export default function Alerts() {
  const [alerts, setAlerts] = useState<AlertModel[]>([]);
  const [form, setForm] = useState({ name: "", threshold: 10, webhook_url: "" });
  const [showForm, setShowForm] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    const { data } = await api.get<AlertModel[]>("/alerts");
    setAlerts(data);
  };

  useEffect(() => {
    load();
  }, []);

  const submit = async () => {
    setError(null);
    try {
      await api.post("/alerts", {
        name: form.name,
        threshold: form.threshold,
        webhook_url: form.webhook_url || null,
        is_enabled: true,
      });
      setForm({ name: "", threshold: 10, webhook_url: "" });
      setShowForm(false);
      load();
    } catch (err: any) {
      setError(err?.response?.data?.detail ?? "Failed to create alert");
    }
  };

  const toggle = async (a: AlertModel) => {
    await api.patch(`/alerts/${a.id}`, { is_enabled: !a.is_enabled });
    load();
  };

  const remove = async (id: number) => {
    if (!confirm("Delete this alert?")) return;
    await api.delete(`/alerts/${id}`);
    load();
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Alerts</h1>
          <p className="text-sm text-slate-500">
            Trigger webhooks when the people count crosses a threshold.
          </p>
        </div>
        <button className="btn-primary" onClick={() => setShowForm((s) => !s)}>
          {showForm ? "Cancel" : "New alert"}
        </button>
      </div>

      {showForm && (
        <div className="card space-y-3">
          <div>
            <label className="label">Name</label>
            <input
              className="input"
              value={form.name}
              onChange={(e) => setForm({ ...form, name: e.target.value })}
            />
          </div>
          <div>
            <label className="label">Threshold (people)</label>
            <input
              type="number"
              className="input"
              min={1}
              value={form.threshold}
              onChange={(e) => setForm({ ...form, threshold: parseInt(e.target.value) })}
            />
          </div>
          <div>
            <label className="label">Webhook URL (optional)</label>
            <input
              className="input"
              placeholder="https://hooks.example.com/peoplesense"
              value={form.webhook_url}
              onChange={(e) => setForm({ ...form, webhook_url: e.target.value })}
            />
          </div>
          {error && <div className="text-sm text-red-600">{error}</div>}
          <button className="btn-primary" onClick={submit} disabled={!form.name}>
            Create
          </button>
        </div>
      )}

      <div className="card">
        <table className="w-full text-sm">
          <thead className="text-xs text-slate-500 uppercase">
            <tr>
              <th className="text-left py-2">Name</th>
              <th className="text-left py-2">Threshold</th>
              <th className="text-left py-2">Webhook</th>
              <th className="text-left py-2">Last triggered</th>
              <th className="text-left py-2">Status</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {alerts.length === 0 && (
              <tr>
                <td colSpan={6} className="text-center text-slate-400 py-6">
                  No alerts configured.
                </td>
              </tr>
            )}
            {alerts.map((a) => (
              <tr key={a.id} className="border-t border-slate-100">
                <td className="py-2 font-medium">{a.name}</td>
                <td>{a.threshold}</td>
                <td className="truncate max-w-xs">{a.webhook_url ?? "—"}</td>
                <td>
                  {a.last_triggered_at ? new Date(a.last_triggered_at).toLocaleString() : "—"}
                </td>
                <td>
                  <button
                    onClick={() => toggle(a)}
                    className={`inline-block px-2 py-0.5 text-xs rounded-full ${
                      a.is_enabled
                        ? "bg-emerald-50 text-emerald-700"
                        : "bg-slate-100 text-slate-600"
                    }`}
                  >
                    {a.is_enabled ? "enabled" : "disabled"}
                  </button>
                </td>
                <td className="text-right">
                  <button className="text-red-600 text-xs hover:underline" onClick={() => remove(a.id)}>
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
