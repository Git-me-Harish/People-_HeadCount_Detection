import { useEffect, useState } from "react";
import { api, Camera } from "../api/client";

export default function Cameras() {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [showForm, setShowForm] = useState(false);
  const [form, setForm] = useState({ name: "", location: "", stream_url: "" });
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    const { data } = await api.get<Camera[]>("/cameras");
    setCameras(data);
  };

  useEffect(() => {
    load();
  }, []);

  const submit = async () => {
    setError(null);
    try {
      await api.post("/cameras", {
        name: form.name,
        location: form.location || null,
        stream_url: form.stream_url || null,
      });
      setForm({ name: "", location: "", stream_url: "" });
      setShowForm(false);
      load();
    } catch (err: any) {
      setError(err?.response?.data?.detail ?? "Failed to create camera");
    }
  };

  const remove = async (id: number) => {
    if (!confirm("Delete this camera?")) return;
    await api.delete(`/cameras/${id}`);
    load();
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Cameras</h1>
          <p className="text-sm text-slate-500">
            Configure RTSP/HTTP streams and physical locations being monitored.
          </p>
        </div>
        <button className="btn-primary" onClick={() => setShowForm((s) => !s)}>
          {showForm ? "Cancel" : "Add camera"}
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
            <label className="label">Location</label>
            <input
              className="input"
              value={form.location}
              onChange={(e) => setForm({ ...form, location: e.target.value })}
            />
          </div>
          <div>
            <label className="label">Stream URL (RTSP/HTTP)</label>
            <input
              className="input"
              placeholder="rtsp://camera.example.com/stream"
              value={form.stream_url}
              onChange={(e) => setForm({ ...form, stream_url: e.target.value })}
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
              <th className="text-left py-2">Location</th>
              <th className="text-left py-2">Stream URL</th>
              <th className="text-left py-2">Status</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {cameras.length === 0 && (
              <tr>
                <td colSpan={5} className="text-center text-slate-400 py-6">
                  No cameras configured yet.
                </td>
              </tr>
            )}
            {cameras.map((c) => (
              <tr key={c.id} className="border-t border-slate-100">
                <td className="py-2 font-medium">{c.name}</td>
                <td>{c.location ?? "—"}</td>
                <td className="truncate max-w-xs">{c.stream_url ?? "—"}</td>
                <td>
                  <span
                    className={`inline-block px-2 py-0.5 text-xs rounded-full ${
                      c.is_active
                        ? "bg-emerald-50 text-emerald-700"
                        : "bg-slate-100 text-slate-600"
                    }`}
                  >
                    {c.is_active ? "active" : "disabled"}
                  </span>
                </td>
                <td className="text-right">
                  <button className="text-red-600 text-xs hover:underline" onClick={() => remove(c.id)}>
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
