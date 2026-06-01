import { FormEvent, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Activity, ArrowRight, CheckCircle2 } from "lucide-react";
import { api } from "../api/client";
import { useAuthStore } from "../auth/store";

const PERKS = [
  "No credit card required",
  "Up to 3 cameras on free tier",
  "GDPR-compliant by default",
  "Deploy in under 5 minutes",
];

export default function Register() {
  const [form, setForm] = useState({
    full_name: "",
    organization_name: "",
    email: "",
    password: "",
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const setToken = useAuthStore((s) => s.setToken);
  const setUser = useAuthStore((s) => s.setUser);
  const navigate = useNavigate();

  const update =
    (key: keyof typeof form) =>
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setForm((prev) => ({ ...prev, [key]: e.target.value }));
    };

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const { data } = await api.post("/auth/register", form);
      setToken(data.access_token);
      const me = await api.get("/auth/me");
      setUser(me.data);
      navigate("/dashboard");
    } catch (err: any) {
      setError(
        err?.response?.data?.detail || err?.message || "Registration failed. Please try again."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex" style={{ backgroundColor: "var(--bg-base)" }}>
      {/* Left branding panel */}
      <div
        className="ops hidden lg:flex flex-col justify-between w-[420px] flex-shrink-0 p-12"
        style={{ backgroundColor: "var(--bg-surface)", borderRight: "1px solid var(--border-base)" }}
      >
        <div className="flex items-center gap-2.5">
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center"
            style={{ backgroundColor: "var(--accent)" }}
          >
            <Activity size={16} style={{ color: "#0b0f14" }} />
          </div>
          <span className="font-semibold" style={{ color: "var(--text-primary)" }}>
            PeopleSense
          </span>
        </div>

        <div>
          <h2
            className="text-2xl font-bold mb-4 leading-tight"
            style={{ color: "var(--text-primary)" }}
          >
            Your operations command center starts here.
          </h2>
          <p className="text-sm mb-8 leading-relaxed" style={{ color: "var(--text-muted)" }}>
            Join organizations that use PeopleSense to monitor crowd density,
            prevent incidents, and optimize space utilization in real time.
          </p>
          <ul className="space-y-3">
            {PERKS.map((p) => (
              <li key={p} className="flex items-center gap-2.5 text-sm" style={{ color: "var(--text-secondary)" }}>
                <CheckCircle2 size={15} style={{ color: "var(--accent)", flexShrink: 0 }} />
                {p}
              </li>
            ))}
          </ul>

          {/* Deploy preview GIF — sits directly below perks */}
          <div
            className="mt-8 rounded-xl overflow-hidden"
            style={{ border: "1px solid rgba(255,255,255,0.08)" }}
          >
            <img
              src="/images/SignUp.gif"
              alt="Deploy in under 5 minutes"
              className="w-full object-cover"
              style={{ maxHeight: "160px", objectPosition: "center" }}
            />
          </div>
        </div>

        <div
          className="rounded-xl p-4"
          style={{ backgroundColor: "var(--bg-elevated)", border: "1px solid var(--border-base)" }}
        >
          <p className="text-xs" style={{ color: "var(--text-muted)" }}>
            Trusted by operations teams at transport hubs, pilgrimage sites, hospitals,
            and large public venues across South Asia and beyond.
          </p>
        </div>
      </div>

      {/* Form panel */}
      <div className="flex-1 flex items-center justify-center px-6 py-12">
        <div className="w-full max-w-sm">
          {/* Mobile logo */}
          <div className="flex items-center gap-2.5 mb-10 lg:hidden">
            <div
              className="w-8 h-8 rounded-lg flex items-center justify-center"
              style={{ backgroundColor: "var(--accent)" }}
            >
              <Activity size={16} style={{ color: "#0b0f14" }} />
            </div>
            <span className="font-semibold" style={{ color: "var(--text-primary)" }}>
              PeopleSense
            </span>
          </div>

          <h1
            className="text-2xl font-bold mb-1 tracking-tight"
            style={{ color: "var(--text-primary)" }}
          >
            Create your workspace
          </h1>
          <p className="text-sm mb-8" style={{ color: "var(--text-muted)" }}>
            Free forever. No credit card required.
          </p>

          <form onSubmit={onSubmit} className="space-y-4">
            {error && (
              <div
                className="px-4 py-3 rounded-lg text-sm"
                style={{
                  backgroundColor: "var(--danger-dim)",
                  border: "1px solid rgba(239,68,68,0.25)",
                  color: "var(--danger)",
                }}
              >
                {error}
              </div>
            )}

            {[
              { key: "organization_name" as const, label: "Organization name", type: "text", placeholder: "Acme Transit Authority" },
              { key: "full_name" as const, label: "Your full name", type: "text", placeholder: "Ravi Kumar" },
              { key: "email" as const, label: "Work email", type: "email", placeholder: "ravi@acme.in" },
              { key: "password" as const, label: "Password", type: "password", placeholder: "Min. 8 characters" },
            ].map((field) => (
              <div key={field.key} className="space-y-1.5">
                <label
                  className="block text-xs font-semibold uppercase tracking-wider"
                  style={{ color: "var(--text-muted)" }}
                >
                  {field.label}
                </label>
                <input
                  type={field.type}
                  required
                  minLength={field.key === "password" ? 8 : undefined}
                  value={form[field.key]}
                  onChange={update(field.key)}
                  placeholder={field.placeholder}
                  className="block w-full rounded-lg text-sm px-3 py-2.5 transition-all focus:outline-none focus:ring-2"
                  style={{
                    backgroundColor: "var(--bg-subtle)",
                    border: "1px solid var(--border-base)",
                    color: "var(--text-primary)",
                  }}
                />
              </div>
            ))}

            <button
              type="submit"
              disabled={loading}
              className="w-full flex items-center justify-center gap-2 py-2.5 rounded-lg text-sm font-semibold transition-all disabled:opacity-50 mt-2"
              style={{ backgroundColor: "var(--accent)", color: "#0b0f14" }}
            >
              {loading ? (
                <span className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" />
              ) : (
                <>
                  Create account
                  <ArrowRight size={14} />
                </>
              )}
            </button>

            <p className="text-2xs text-center" style={{ color: "var(--text-muted)" }}>
              By creating an account you agree to our Terms of Service and Privacy Policy.
            </p>
          </form>

          <p className="text-sm text-center mt-6" style={{ color: "var(--text-muted)" }}>
            Already have an account?{" "}
            <Link to="/login" className="font-medium" style={{ color: "var(--accent)" }}>
              Sign in
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}