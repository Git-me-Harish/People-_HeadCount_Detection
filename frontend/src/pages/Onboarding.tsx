/**
 * Onboarding wizard — vertical picker + template application.
 * Shown once after registration if no cameras exist.
 */
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Activity, ArrowRight, CheckCircle2 } from "lucide-react";
import { useTranslation } from "react-i18next";
import { templatesApi } from "../api/client";
import { Button, Card, Spinner, AlertBanner } from "../components/ui";
import { ROUTES, VERTICALS } from "../constants";
import type { IndustryTemplate } from "../types";
import { useAsync } from "../hooks";

export default function Onboarding() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [selected, setSelected] = useState<string | null>(null);
  const [applying, setApplying] = useState(false);
  const [done, setDone] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { data: templates, loading } = useAsync(() => templatesApi.list());

  const handleApply = async () => {
    if (!selected) return;
    setApplying(true);
    setError(null);
    try {
      await templatesApi.apply(selected);
      setDone(true);
      setTimeout(() => navigate(ROUTES.DASHBOARD), 1800);
    } catch (e: unknown) {
      setError(
        (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
          "Failed to apply template"
      );
    } finally {
      setApplying(false);
    }
  };

  if (done) {
    return (
      <div
        className="min-h-screen flex flex-col items-center justify-center"
        style={{ backgroundColor: "var(--bg-base)" }}
      >
        <div
          className="w-14 h-14 rounded-2xl flex items-center justify-center mb-5"
          style={{ backgroundColor: "var(--success-dim)" }}
        >
          <CheckCircle2 size={28} style={{ color: "var(--success)" }} />
        </div>
        <h2 className="text-xl font-bold mb-1" style={{ color: "var(--text-primary)" }}>
          Workspace configured
        </h2>
        <p className="text-sm" style={{ color: "var(--text-muted)" }}>
          Redirecting to your dashboard...
        </p>
      </div>
    );
  }

  const selectedTemplate = templates?.find((t: IndustryTemplate) => t.vertical === selected);

  return (
    <div
      className="min-h-screen flex flex-col items-center justify-center p-6"
      style={{ backgroundColor: "var(--bg-base)" }}
    >
      <div className="w-full max-w-3xl">
        {/* Header */}
        <div className="text-center mb-10">
          <div className="flex items-center justify-center gap-2.5 mb-4">
            <div
              className="w-9 h-9 rounded-xl flex items-center justify-center"
              style={{ backgroundColor: "var(--accent)" }}
            >
              <Activity size={18} style={{ color: "#0b0f14" }} />
            </div>
            <span className="font-bold text-xl" style={{ color: "var(--text-primary)" }}>
              PeopleSense
            </span>
          </div>
          <h1 className="text-2xl font-bold mb-2" style={{ color: "var(--text-primary)" }}>
            Set up your workspace
          </h1>
          <p className="text-sm max-w-sm mx-auto" style={{ color: "var(--text-muted)" }}>
            Select the vertical that best describes your environment. We'll pre-configure
            cameras and alert rules so you can start monitoring immediately.
          </p>
        </div>

        {/* Vertical grid */}
        {loading ? (
          <div className="flex justify-center py-12">
            <Spinner />
          </div>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
            {VERTICALS.map((v) => {
              const tmpl = templates?.find((t: IndustryTemplate) => t.vertical === v.key);
              const isSelected = selected === v.key;
              return (
                <button
                  key={v.key}
                  onClick={() => setSelected(v.key)}
                  className="relative p-4 rounded-xl text-left transition-all duration-150"
                  style={{
                    backgroundColor: isSelected ? "var(--accent-dim)" : "var(--bg-surface)",
                    border: `1px solid ${isSelected ? "var(--accent)" : "var(--border-base)"}`,
                  }}
                  onMouseEnter={(e) => {
                    if (!isSelected) (e.currentTarget.style.borderColor = "rgba(0,194,168,0.4)");
                  }}
                  onMouseLeave={(e) => {
                    if (!isSelected) (e.currentTarget.style.borderColor = "var(--border-base)");
                  }}
                >
                  <div
                    className="w-8 h-8 rounded-lg flex items-center justify-center mb-3"
                    style={{ backgroundColor: isSelected ? "var(--accent)" : "var(--bg-subtle)" }}
                  >
                    {isSelected ? (
                      <CheckCircle2 size={16} style={{ color: "#0b0f14" }} />
                    ) : (
                      <div style={{ color: "var(--text-muted)", fontFamily: "DM Mono", fontSize: "11px", fontWeight: 600 }}>
                        {v.key.slice(0, 2).toUpperCase()}
                      </div>
                    )}
                  </div>
                  <p
                    className="text-xs font-semibold leading-tight mb-1"
                    style={{ color: isSelected ? "var(--accent)" : "var(--text-primary)" }}
                  >
                    {v.label}
                  </p>
                  {tmpl && (
                    <p className="text-2xs leading-tight hidden sm:block" style={{ color: "var(--text-muted)" }}>
                      {tmpl.default_cameras.length} cameras · {tmpl.default_alerts.length} alerts
                    </p>
                  )}
                </button>
              );
            })}
          </div>
        )}

        {/* Template preview */}
        {selectedTemplate && (
          <Card className="mb-6" padding={false}>
            <div
              className="px-5 py-3.5"
              style={{ borderBottom: "1px solid var(--border-subtle)" }}
            >
              <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: "var(--text-muted)" }}>
                Template Preview — {selectedTemplate.name}
              </p>
            </div>
            <div className="p-5">
              <p className="text-xs mb-4" style={{ color: "var(--text-secondary)" }}>
                {selectedTemplate.description}
              </p>
              <div className="grid grid-cols-2 gap-5">
                <div>
                  <p className="text-2xs font-semibold uppercase tracking-wider mb-2" style={{ color: "var(--text-muted)" }}>
                    Cameras to create
                  </p>
                  <ul className="space-y-1">
                    {selectedTemplate.default_cameras.map(
                      (c: { name: string; location?: string }, i: number) => (
                        <li key={i} className="flex items-center gap-2 text-xs" style={{ color: "var(--text-secondary)" }}>
                          <span className="w-1 h-1 rounded-full flex-shrink-0" style={{ backgroundColor: "var(--accent)" }} />
                          {c.name}
                        </li>
                      )
                    )}
                  </ul>
                </div>
                <div>
                  <p className="text-2xs font-semibold uppercase tracking-wider mb-2" style={{ color: "var(--text-muted)" }}>
                    Alert rules
                  </p>
                  <ul className="space-y-1">
                    {selectedTemplate.default_alerts.map(
                      (a: { name: string; threshold: number }, i: number) => (
                        <li key={i} className="flex items-center gap-2 text-xs" style={{ color: "var(--text-secondary)" }}>
                          <span className="w-1 h-1 rounded-full flex-shrink-0" style={{ backgroundColor: "var(--warn)" }} />
                          {a.name}{" "}
                          <span className="font-mono" style={{ color: "var(--text-muted)" }}>
                            ({a.threshold}+)
                          </span>
                        </li>
                      )
                    )}
                  </ul>
                </div>
              </div>
            </div>
          </Card>
        )}

        {error && <AlertBanner type="error">{error}</AlertBanner>}

        {/* Actions */}
        <div className="flex items-center gap-3 justify-center mt-6">
          <button
            onClick={() => navigate(ROUTES.DASHBOARD)}
            className="text-sm transition-colors"
            style={{ color: "var(--text-muted)" }}
          >
            {t("skipOnboarding")}
          </button>
          <Button
            onClick={handleApply}
            disabled={!selected}
            loading={applying}
            icon={<ArrowRight size={15} />}
          >
            {t("applyTemplate")}
          </Button>
        </div>
      </div>
    </div>
  );
}