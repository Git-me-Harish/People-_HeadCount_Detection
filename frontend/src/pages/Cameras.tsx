import { useEffect, useState } from "react";
import { Camera as CameraIcon, MapPin, Plus, Radio, Trash2 } from "lucide-react";
import { api, Camera } from "../api/client";
import { Button, Card, PageHeader, Modal, Input, EmptyState, AlertBanner, Badge, StatusPill } from "../components/ui";

export default function Cameras() {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [showForm, setShowForm] = useState(false);
  const [form, setForm] = useState({ name: "", location: "", stream_url: "" });
  const [error, setError] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<number | null>(null);

  const load = async () => {
    const { data } = await api.get<Camera[]>("/cameras");
    setCameras(data);
  };

  useEffect(() => { load(); }, []);

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
      setError(err?.response?.data?.detail ?? "Failed to add camera");
    }
  };

  const remove = async (id: number) => {
    if (!confirm("Permanently remove this camera?")) return;
    setDeleting(id);
    try {
      await api.delete(`/cameras/${id}`);
      load();
    } finally {
      setDeleting(null);
    }
  };

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-5">
      <PageHeader
        title="Cameras"
        subtitle="Manage RTSP / HTTP stream sources and physical locations"
        actions={
          <Button icon={<Plus size={14} />} size="sm" onClick={() => setShowForm(true)}>
            Add camera
          </Button>
        }
      />

      <Card padding={false}>
        {/* Column headers */}
        <div
          className="grid items-center px-5 py-3 text-2xs font-semibold uppercase tracking-wider"
          style={{
            gridTemplateColumns: "2fr 1.5fr 2fr 80px 56px",
            borderBottom: "1px solid var(--border-subtle)",
            color: "var(--text-muted)",
          }}
        >
          <span>Camera</span>
          <span>Location</span>
          <span>Stream URL</span>
          <span>Status</span>
          <span />
        </div>

        {cameras.length === 0 ? (
          <EmptyState
            icon={<CameraIcon size={28} />}
            title="No cameras configured"
            description="Add an IP camera stream URL to start monitoring crowd density in real time."
            action={
              <Button size="sm" icon={<Plus size={13} />} onClick={() => setShowForm(true)}>
                Add first camera
              </Button>
            }
          />
        ) : (
          cameras.map((c) => (
            <div
              key={c.id}
              className="grid items-center px-5 py-3.5 transition-colors"
              style={{
                gridTemplateColumns: "2fr 1.5fr 2fr 80px 56px",
                borderBottom: "1px solid var(--border-subtle)",
              }}
              onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "var(--bg-subtle)")}
              onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
            >
              {/* Name */}
              <div className="flex items-center gap-2.5 min-w-0">
                <div
                  className="w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0"
                  style={{ backgroundColor: "var(--accent-dim)" }}
                >
                  <CameraIcon size={13} style={{ color: "var(--accent)" }} />
                </div>
                <div className="min-w-0">
                  <p className="text-sm font-medium truncate" style={{ color: "var(--text-primary)" }}>
                    {c.name}
                  </p>
                  <p className="text-2xs font-mono" style={{ color: "var(--text-muted)" }}>
                    ID #{c.id}
                  </p>
                </div>
              </div>

              {/* Location */}
              <div className="flex items-center gap-1.5 min-w-0">
                {c.location ? (
                  <>
                    <MapPin size={11} style={{ color: "var(--text-muted)", flexShrink: 0 }} />
                    <span className="text-xs truncate" style={{ color: "var(--text-secondary)" }}>
                      {c.location}
                    </span>
                  </>
                ) : (
                  <span className="text-xs" style={{ color: "var(--text-muted)" }}>—</span>
                )}
              </div>

              {/* Stream URL */}
              <div className="flex items-center gap-1.5 min-w-0">
                {c.stream_url ? (
                  <>
                    <Radio size={11} style={{ color: "var(--text-muted)", flexShrink: 0 }} />
                    <span
                      className="text-2xs font-mono truncate"
                      style={{ color: "var(--text-muted)" }}
                      title={c.stream_url}
                    >
                      {c.stream_url}
                    </span>
                  </>
                ) : (
                  <Badge color="gray">Manual / API</Badge>
                )}
              </div>

              {/* Status */}
              <StatusPill active={true} labelOn="Active" />

              {/* Delete */}
              <div className="flex justify-end">
                <button
                  onClick={() => remove(c.id)}
                  disabled={deleting === c.id}
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
                >
                  <Trash2 size={13} />
                </button>
              </div>
            </div>
          ))
        )}
      </Card>

      {/* Add camera modal */}
      <Modal
        open={showForm}
        onClose={() => { setShowForm(false); setError(null); }}
        title="Add Camera"
        footer={
          <>
            <Button variant="secondary" size="sm" onClick={() => { setShowForm(false); setError(null); }}>
              Cancel
            </Button>
            <Button size="sm" onClick={submit} disabled={!form.name}>
              Add camera
            </Button>
          </>
        }
      >
        <div className="space-y-4">
          {error && <AlertBanner type="error">{error}</AlertBanner>}

          <Input
            label="Camera name"
            value={form.name}
            onChange={(e) => setForm({ ...form, name: e.target.value })}
            placeholder="e.g. Main Entrance East"
          />

          <Input
            label="Location"
            value={form.location}
            onChange={(e) => setForm({ ...form, location: e.target.value })}
            placeholder="e.g. Ground Floor, Gate 3"
            helper="Physical location for reference"
          />

          <Input
            label="Stream URL"
            value={form.stream_url}
            onChange={(e) => setForm({ ...form, stream_url: e.target.value })}
            placeholder="rtsp://192.168.1.100/stream"
            helper="RTSP or HTTP stream. Leave blank to use the upload API."
          />
        </div>
      </Modal>
    </div>
  );
}