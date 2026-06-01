import { useState } from "react";
import { Download, FileText, CheckCircle2 } from "lucide-react";
import { reportsApi, ReportUnavailableError } from "../api/client";
import { Button, Card, PageHeader, AlertBanner } from "../components/ui";

const WINDOWS = [
  { label: "Last 7 days",  days: 7,  desc: "Weekly crowd summary" },
  { label: "Last 14 days", days: 14, desc: "Bi-weekly trend report" },
  { label: "Last 30 days", days: 30, desc: "Monthly operations report" },
  { label: "Last 90 days", days: 90, desc: "Quarterly analytics export" },
];

const INCLUDES = [
  "Peak and average head counts per window",
  "Per-camera breakdown (samples, peak, average)",
  "Total detection events with timeline",
  "Report generation timestamp and metadata",
];

export default function Reports() {
  const [downloading, setDownloading] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleDownload = async (days: number) => {
    setDownloading(days);
    setError(null);
    try {
      await reportsApi.downloadPdf(days);
    } catch (err: unknown) {
      if (err instanceof ReportUnavailableError) {
        setError(err.message);
      } else {
        setError("Download failed — please try again or contact support.");
      }
    } finally {
      setDownloading(null);
    }
  };

  return (
    <div className="p-6 max-w-3xl mx-auto space-y-5">
      <PageHeader
        title="Reports"
        subtitle="Download PDF summaries of crowd activity and analytics"
      />

      {error && <AlertBanner type="error">{error}</AlertBanner>}

      {/* Report windows */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {WINDOWS.map(({ label, days, desc }) => (
          <Card key={days} className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-3 min-w-0">
              <div
                className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0"
                style={{ backgroundColor: "var(--accent-dim)" }}
              >
                <FileText size={16} style={{ color: "var(--accent)" }} />
              </div>
              <div className="min-w-0">
                <p className="text-sm font-medium truncate" style={{ color: "var(--text-primary)" }}>
                  {label}
                </p>
                <p className="text-2xs" style={{ color: "var(--text-muted)" }}>
                  {desc}
                </p>
              </div>
            </div>
            <Button
              variant="secondary"
              size="sm"
              icon={<Download size={13} />}
              loading={downloading === days}
              onClick={() => handleDownload(days)}
              className="flex-shrink-0"
            >
              PDF
            </Button>
          </Card>
        ))}
      </div>

      {/* What's included */}
      <Card padding={false}>
        <div
          className="px-5 py-3.5"
          style={{ borderBottom: "1px solid var(--border-subtle)" }}
        >
          <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: "var(--text-muted)" }}>
            Report Contents
          </p>
        </div>
        <div className="p-5">
          <ul className="space-y-2.5">
            {INCLUDES.map((item) => (
              <li key={item} className="flex items-start gap-2.5 text-sm" style={{ color: "var(--text-secondary)" }}>
                <CheckCircle2 size={14} className="flex-shrink-0 mt-0.5" style={{ color: "var(--accent)" }} />
                {item}
              </li>
            ))}
          </ul>
        </div>
      </Card>
    </div>
  );
}