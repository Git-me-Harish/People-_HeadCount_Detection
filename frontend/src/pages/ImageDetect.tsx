import { useRef, useState } from "react";
import { Upload, ImagePlus, X } from "lucide-react";
import { api, DetectionResult } from "../api/client";
import { Button, Card, PageHeader, AlertBanner } from "../components/ui";

function ResultMetric({
  label,
  value,
  highlight = false,
}: {
  label: string;
  value: string | number;
  highlight?: boolean;
}) {
  return (
    <div
      className="flex-1 px-5 py-4 rounded-xl"
      style={{
        backgroundColor: highlight ? "var(--accent-dim)" : "var(--bg-subtle)",
        border: `1px solid ${highlight ? "rgba(0,194,168,0.35)" : "var(--border-base)"}`,
      }}
    >
      <p className="text-2xs font-semibold uppercase tracking-widest mb-1.5" style={{ color: "var(--text-muted)" }}>
        {label}
      </p>
      <p
        className="text-3xl font-mono font-medium leading-none"
        style={{ color: highlight ? "var(--accent)" : "var(--text-primary)", letterSpacing: "-0.02em" }}
      >
        {value}
      </p>
    </div>
  );
}

export default function ImageDetect() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [confidence, setConfidence] = useState(0.35);
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = (f: File) => {
    setFile(f);
    setResult(null);
    setError(null);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target?.result as string);
    reader.readAsDataURL(f);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const f = e.dataTransfer.files?.[0];
    if (f && f.type.startsWith("image/")) handleFile(f);
  };

  const clear = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  const onSubmit = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("confidence", String(confidence));
      form.append("annotate", "true");
      const { data } = await api.post("/detect/image", form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(data);
    } catch (err: any) {
      setError(err?.response?.data?.detail ?? "Detection failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 max-w-4xl mx-auto space-y-5">
      <PageHeader
        title="Image Detection"
        subtitle="Upload an image to count people and receive annotated output"
      />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* Upload card */}
        <Card padding={false}>
          <div
            className="px-5 py-3.5"
            style={{ borderBottom: "1px solid var(--border-subtle)" }}
          >
            <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: "var(--text-muted)" }}>
              Upload Image
            </p>
          </div>
          <div className="p-5 space-y-4">
            {error && <AlertBanner type="error">{error}</AlertBanner>}

            {/* Drop zone */}
            {!preview ? (
              <div
                className="rounded-xl border-2 border-dashed flex flex-col items-center justify-center py-12 cursor-pointer transition-colors"
                style={{ borderColor: "var(--border-base)" }}
                onClick={() => inputRef.current?.click()}
                onDrop={handleDrop}
                onDragOver={(e) => e.preventDefault()}
                onMouseEnter={(e) => (e.currentTarget.style.borderColor = "var(--accent)")}
                onMouseLeave={(e) => (e.currentTarget.style.borderColor = "var(--border-base)")}
              >
                <div
                  className="w-12 h-12 rounded-xl flex items-center justify-center mb-3"
                  style={{ backgroundColor: "var(--accent-dim)" }}
                >
                  <ImagePlus size={22} style={{ color: "var(--accent)" }} />
                </div>
                <p className="text-sm font-medium" style={{ color: "var(--text-secondary)" }}>
                  Drop an image here
                </p>
                <p className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>
                  or click to browse — JPEG, PNG, WebP
                </p>
                <input
                  ref={inputRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) handleFile(f);
                  }}
                />
              </div>
            ) : (
              <div className="relative">
                <img
                  src={preview}
                  alt="Preview"
                  className="w-full rounded-xl object-contain max-h-52"
                  style={{ backgroundColor: "var(--bg-subtle)" }}
                />
                <button
                  onClick={clear}
                  className="absolute top-2 right-2 p-1.5 rounded-lg"
                  style={{ backgroundColor: "var(--bg-elevated)", color: "var(--text-muted)" }}
                >
                  <X size={13} />
                </button>
                <p className="text-xs mt-2 font-mono truncate" style={{ color: "var(--text-muted)" }}>
                  {file?.name}
                </p>
              </div>
            )}

            {/* Confidence slider */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-xs font-semibold uppercase tracking-wider" style={{ color: "var(--text-muted)" }}>
                  Confidence threshold
                </label>
                <span className="text-sm font-mono font-medium" style={{ color: "var(--text-primary)" }}>
                  {confidence.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min={0.05}
                max={0.95}
                step={0.05}
                value={confidence}
                onChange={(e) => setConfidence(parseFloat(e.target.value))}
                className="w-full accent-[var(--accent)]"
              />
              <div className="flex justify-between text-2xs" style={{ color: "var(--text-muted)" }}>
                <span>0.05 — more detections</span>
                <span>0.95 — fewer, precise</span>
              </div>
            </div>

            <Button
              onClick={onSubmit}
              disabled={!file || loading}
              loading={loading}
              icon={<Upload size={14} />}
              className="w-full justify-center"
            >
              {loading ? "Running detection..." : "Run detection"}
            </Button>
          </div>
        </Card>

        {/* Results card */}
        <Card padding={false}>
          <div
            className="px-5 py-3.5"
            style={{ borderBottom: "1px solid var(--border-subtle)" }}
          >
            <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: "var(--text-muted)" }}>
              Detection Output
            </p>
          </div>
          <div className="p-5">
            {!result ? (
              <div
                className="flex flex-col items-center justify-center py-16 text-sm"
                style={{ color: "var(--text-muted)" }}
              >
                <div
                  className="w-12 h-12 rounded-xl flex items-center justify-center mb-3"
                  style={{ backgroundColor: "var(--bg-subtle)" }}
                >
                  <Upload size={20} style={{ color: "var(--text-muted)" }} />
                </div>
                Results will appear here
              </div>
            ) : (
              <div className="space-y-4">
                {/* Metrics */}
                <div className="flex gap-3">
                  <ResultMetric label="People detected" value={result.person_count} highlight={true} />
                  <ResultMetric
                    label="Avg. confidence"
                    value={result.avg_confidence != null ? result.avg_confidence.toFixed(2) : "—"}
                  />
                </div>
                <div
                  className="flex items-center justify-between px-3 py-2 rounded-lg text-xs font-mono"
                  style={{ backgroundColor: "var(--bg-subtle)", color: "var(--text-muted)" }}
                >
                  <span>Resolution</span>
                  <span style={{ color: "var(--text-primary)" }}>{result.width} × {result.height}</span>
                </div>

                {/* Annotated image */}
                {result.annotated_image_b64 && (
                  <div>
                    <p className="text-2xs font-semibold uppercase tracking-wider mb-2" style={{ color: "var(--text-muted)" }}>
                      Annotated result
                    </p>
                    <img
                      src={`data:image/png;base64,${result.annotated_image_b64}`}
                      alt="Annotated detection result"
                      className="w-full rounded-xl object-contain"
                      style={{
                        backgroundColor: "var(--bg-subtle)",
                        border: "1px solid var(--border-base)",
                        maxHeight: "280px",
                      }}
                    />
                  </div>
                )}
              </div>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}