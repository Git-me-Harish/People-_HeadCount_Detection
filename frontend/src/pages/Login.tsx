import { FormEvent, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Activity, ArrowRight, Eye, EyeOff } from "lucide-react";
import { api } from "../api/client";
import { useAuthStore } from "../auth/store";

export default function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPw, setShowPw] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const setToken = useAuthStore((s) => s.setToken);
  const setUser = useAuthStore((s) => s.setUser);
  const navigate = useNavigate();

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const { data } = await api.post("/auth/login", { email, password });
      setToken(data.access_token);
      const me = await api.get("/auth/me");
      setUser(me.data);
      navigate("/dashboard");
    } catch (err: any) {
      setError(err?.response?.data?.detail ?? "Invalid credentials. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      className="min-h-screen flex"
      style={{ backgroundColor: "var(--bg-base)" }}
    >
      {/* Left panel — branding */}
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
          <blockquote
            className="text-xl font-medium leading-relaxed mb-6"
            style={{ color: "var(--text-primary)" }}
          >
            Crowd intelligence that helps us manage temple flow during peak festivals. 
            Incidents down 60% since deployment.
          </blockquote>
          <div>
            <p className="text-sm font-medium" style={{ color: "var(--text-secondary)" }}>
              Operations Manager
            </p>
            <p className="text-xs" style={{ color: "var(--text-muted)" }}>
              Tirumala Tirupati Devasthanams
            </p>
          </div>
           <div
            className="mt-8 rounded-xl overflow-hidden"
            style={{ border: "1px solid var(--border-base)" }}
          >
            <img
              src="/images/Login.gif"
              alt="PeopleSense in action"
              className="w-full object-cover"
              style={{ maxHeight: "160px", objectPosition: "center" }}
            />
          </div>
        </div>

        <div
          className="grid grid-cols-2 gap-4 text-center rounded-xl p-4"
          style={{ backgroundColor: "var(--bg-elevated)", border: "1px solid var(--border-base)" }}
        >
          {[
            { v: "500ms", l: "Avg latency" },
            { v: "99.9%", l: "Uptime SLA" },
            { v: "8+",    l: "Verticals" },
            { v: "GDPR",  l: "Compliant" },
          ].map((s) => (
            <div key={s.l} className="py-2">
              <div
                className="text-lg font-mono font-medium"
                style={{ color: "var(--accent)", letterSpacing: "-0.02em" }}
              >
                {s.v}
              </div>
              <div className="text-2xs mt-0.5" style={{ color: "var(--text-muted)" }}>
                {s.l}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Right panel — form */}
      <div className="flex-1 flex items-center justify-center px-6">
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
            Welcome back
          </h1>
          <p className="text-sm mb-8" style={{ color: "var(--text-muted)" }}>
            Sign in to your operations dashboard
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

            <div className="space-y-1.5">
              <label
                className="block text-xs font-semibold uppercase tracking-wider"
                style={{ color: "var(--text-muted)" }}
              >
                Email address
              </label>
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@company.com"
                className="block w-full rounded-lg text-sm px-3 py-2.5 transition-all focus:outline-none focus:ring-2"
                style={{
                  backgroundColor: "var(--bg-subtle)",
                  border: "1px solid var(--border-base)",
                  color: "var(--text-primary)",
                }}
              />
            </div>

            <div className="space-y-1.5">
              <label
                className="block text-xs font-semibold uppercase tracking-wider"
                style={{ color: "var(--text-muted)" }}
              >
                Password
              </label>
              <div className="relative">
                <input
                  type={showPw ? "text" : "password"}
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  className="block w-full rounded-lg text-sm px-3 py-2.5 pr-10 transition-all focus:outline-none focus:ring-2"
                  style={{
                    backgroundColor: "var(--bg-subtle)",
                    border: "1px solid var(--border-base)",
                    color: "var(--text-primary)",
                  }}
                />
                <button
                  type="button"
                  onClick={() => setShowPw((s) => !s)}
                  className="absolute right-3 top-1/2 -translate-y-1/2"
                  style={{ color: "var(--text-muted)" }}
                >
                  {showPw ? <EyeOff size={14} /> : <Eye size={14} />}
                </button>
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full flex items-center justify-center gap-2 py-2.5 rounded-lg text-sm font-semibold transition-all disabled:opacity-50 mt-6"
              style={{ backgroundColor: "var(--accent)", color: "#0b0f14" }}
            >
              {loading ? (
                <span className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" />
              ) : (
                <>
                  Sign in
                  <ArrowRight size={14} />
                </>
              )}
            </button>
          </form>

          <p className="text-sm text-center mt-6" style={{ color: "var(--text-muted)" }}>
            No account?{" "}
            <Link
              to="/register"
              className="font-medium transition-colors"
              style={{ color: "var(--accent)" }}
            >
              Create workspace
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}