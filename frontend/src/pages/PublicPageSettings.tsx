import { useEffect, useState } from "react";
import { ExternalLink, Globe } from "lucide-react";
import { publicPageApi, camerasApi } from "../api/client";
import { Button, Card, Input, PageHeader, Spinner, Toggle, AlertBanner } from "../components/ui";
import { useAsync } from "../hooks";
import type { PublicPageConfig } from "../types";

export default function PublicPageSettings() {
  const {
    data: existingPage,
    loading,
    refetch,
  } = useAsync(() => publicPageApi.getMine().catch(() => ({ data: null })));
  const { data: cameras } = useAsync(() => camerasApi.list());

  const [form, setForm] = useState<Omit<PublicPageConfig, "is_active">>({
    slug: "",
    title: "",
    description: "",
    camera_ids: [],
    show_heatmap: false,
    brand_color: "#00c2a8",
  });
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (existingPage) {
      setForm({
        slug: existingPage.slug,
        title: existingPage.title,
        description: existingPage.description ?? "",
        camera_ids: existingPage.camera_ids,
        show_heatmap: existingPage.show_heatmap,
        brand_color: existingPage.brand_color,
      });
    }
  }, [existingPage]);

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    try {
      if (existingPage) {
        await publicPageApi.update(form);
      } else {
        await publicPageApi.create(form);
      }
      setSaved(true);
      refetch();
      setTimeout(() => setSaved(false), 2500);
    } catch (e: unknown) {
      setError(
        (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? "Save failed"
      );
    } finally {
      setSaving(false);
    }
  };

  const toggleCamera = (id: number) => {
    setForm((f) => ({
      ...f,
      camera_ids: f.camera_ids.includes(id)
        ? f.camera_ids.filter((c) => c !== id)
        : [...f.camera_ids, id],
    }));
  };

  const publicUrl =
    typeof window !== "undefined" && form.slug
      ? `${window.location.origin}/public/${form.slug}`
      : null;

  if (loading) {
    return (
      <div className="flex justify-center py-20">
        <Spinner />
      </div>
    );
  }

  return (
    <div className="p-6 max-w-2xl mx-auto space-y-5">
      <PageHeader
        title="Public Status Page"
        subtitle="Share a live crowd-status feed with visitors — no login required"
      />

      {/* Live URL preview */}
      {existingPage && publicUrl && (
        <Card>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3 min-w-0">
              <div
                className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
                style={{ backgroundColor: "var(--success-dim)" }}
              >
                <Globe size={15} style={{ color: "var(--success)" }} />
              </div>
              <div className="min-w-0">
                <p className="text-xs font-semibold uppercase tracking-wider mb-0.5" style={{ color: "var(--text-muted)" }}>
                  Live public URL
                </p>
                <a
                  href={publicUrl}
                  target="_blank"
                  rel="noreferrer"
                  className="text-sm font-mono flex items-center gap-1 hover:underline"
                  style={{ color: "var(--accent)" }}
                >
                  {publicUrl}
                  <ExternalLink size={11} />
                </a>
              </div>
            </div>
            <span
              className="text-2xs px-2 py-0.5 rounded-md font-medium flex-shrink-0"
              style={{ backgroundColor: "var(--success-dim)", color: "var(--success)" }}
            >
              Published
            </span>
          </div>
        </Card>
      )}

      {/* Settings form */}
      <Card padding={false}>
        <div
          className="px-5 py-3.5"
          style={{ borderBottom: "1px solid var(--border-subtle)" }}
        >
          <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: "var(--text-muted)" }}>
            Page Configuration
          </p>
        </div>
        <div className="p-5 space-y-4">
          {error && <AlertBanner type="error">{error}</AlertBanner>}
          {saved && <AlertBanner type="success">Changes saved successfully.</AlertBanner>}

          <Input
            label="URL slug"
            placeholder="my-temple-queue"
            value={form.slug}
            onChange={(e) => setForm((f) => ({ ...f, slug: e.target.value }))}
            helper={`Public URL: /public/${form.slug || "your-slug"}`}
            disabled={!!existingPage}
          />

          <Input
            label="Page title"
            placeholder="Temple Queue Status"
            value={form.title}
            onChange={(e) => setForm((f) => ({ ...f, title: e.target.value }))}
          />

          <Input
            label="Description"
            placeholder="Live crowd status for visitors"
            value={form.description ?? ""}
            onChange={(e) => setForm((f) => ({ ...f, description: e.target.value }))}
            helper="Shown below the page title. Keep it brief."
          />

          {/* Brand color */}
          <div className="space-y-1.5">
            <label className="block text-xs font-semibold uppercase tracking-wider" style={{ color: "var(--text-muted)" }}>
              Brand color
            </label>
            <div className="flex items-center gap-3">
              <input
                type="color"
                value={form.brand_color}
                onChange={(e) => setForm((f) => ({ ...f, brand_color: e.target.value }))}
                className="w-10 h-10 rounded-lg cursor-pointer p-0.5"
                style={{ border: "1px solid var(--border-base)", backgroundColor: "var(--bg-subtle)" }}
              />
              <code className="text-xs font-mono" style={{ color: "var(--text-secondary)" }}>
                {form.brand_color}
              </code>
            </div>
          </div>

          <Toggle
            checked={form.show_heatmap}
            onChange={(v) => setForm((f) => ({ ...f, show_heatmap: v }))}
            label="Show density heatmap on public page"
          />
        </div>
      </Card>

      {/* Camera selection */}
      {cameras && cameras.length > 0 && (
        <Card padding={false}>
          <div
            className="px-5 py-3.5"
            style={{ borderBottom: "1px solid var(--border-subtle)" }}
          >
            <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: "var(--text-muted)" }}>
              Cameras to expose
            </p>
            <p className="text-xs mt-0.5" style={{ color: "var(--text-muted)" }}>
              Only selected cameras will be visible on the public page
            </p>
          </div>
          <div className="p-5 space-y-3">
            {cameras.map((cam) => (
              <Toggle
                key={cam.id}
                checked={form.camera_ids.includes(cam.id)}
                onChange={() => toggleCamera(cam.id)}
                label={`${cam.name}${cam.location ? ` — ${cam.location}` : ""}`}
              />
            ))}
          </div>
        </Card>
      )}

      {/* Save */}
      <div className="flex justify-end">
        <Button loading={saving} onClick={handleSave}>
          {saved ? "Saved" : existingPage ? "Save changes" : "Create page"}
        </Button>
      </div>
    </div>
  );
}