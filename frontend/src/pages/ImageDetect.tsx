import { useState } from "react";
import { api, DetectionResult } from "../api/client";

export default function ImageDetect() {
  const [file, setFile] = useState<File | null>(null);
  const [confidence, setConfidence] = useState(0.35);
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Image detection</h1>
        <p className="text-sm text-slate-500">Upload an image to count people instantly.</p>
      </div>
      <div className="card space-y-4">
        <div>
          <label className="label">Image file</label>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            className="block w-full text-sm file:mr-4 file:rounded-md file:border-0 file:bg-brand-50 file:px-4 file:py-2 file:text-sm file:font-medium file:text-brand-700 hover:file:bg-brand-100"
          />
        </div>
        <div>
          <label className="label">Confidence threshold: {confidence.toFixed(2)}</label>
          <input
            type="range"
            min={0.05}
            max={0.95}
            step={0.05}
            value={confidence}
            onChange={(e) => setConfidence(parseFloat(e.target.value))}
            className="w-full"
          />
        </div>
        <button onClick={onSubmit} disabled={!file || loading} className="btn-primary">
          {loading ? "Running detection…" : "Run detection"}
        </button>
        {error && <div className="text-sm text-red-600">{error}</div>}
      </div>

      {result && (
        <div className="card">
          <div className="flex flex-wrap items-center gap-4 mb-4">
            <div>
              <div className="text-xs uppercase text-slate-500">People detected</div>
              <div className="text-3xl font-bold text-brand-700">{result.person_count}</div>
            </div>
            <div>
              <div className="text-xs uppercase text-slate-500">Avg. confidence</div>
              <div className="text-lg font-semibold">
                {result.avg_confidence != null ? result.avg_confidence.toFixed(2) : "—"}
              </div>
            </div>
            <div>
              <div className="text-xs uppercase text-slate-500">Resolution</div>
              <div className="text-lg font-semibold">
                {result.width}×{result.height}
              </div>
            </div>
          </div>
          {result.annotated_image_b64 && (
            <img
              src={`data:image/png;base64,${result.annotated_image_b64}`}
              alt="Annotated detection result"
              className="w-full max-h-[60vh] object-contain rounded-md border border-slate-200"
            />
          )}
        </div>
      )}
    </div>
  );
}
