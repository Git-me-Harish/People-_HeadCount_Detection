import { FormEvent, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api/client";
import { useAuthStore } from "../auth/store";

export default function Register() {
  const [form, setForm] = useState({
    full_name: "",
    organization_name: "",
    email: "",
    password: "",
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const setSession = useAuthStore((s) => s.setSession);
  const navigate = useNavigate();

  const update = (key: keyof typeof form) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setForm({ ...form, [key]: e.target.value });

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const { data } = await api.post("/auth/register", form);
      setSession(data.access_token);
      const me = await api.get("/auth/me");
      setSession(data.access_token, me.data);
      navigate("/");
    } catch (err: any) {
      setError(err?.response?.data?.detail ?? "Registration failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-brand-50 via-white to-brand-50 px-4 py-8">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <div className="text-3xl font-bold text-brand-900">PeopleSense</div>
          <div className="text-sm text-slate-500">Create your workspace</div>
        </div>
        <form onSubmit={onSubmit} className="card space-y-4">
          <h1 className="text-xl font-semibold">Create account</h1>
          {error && (
            <div className="bg-red-50 text-red-700 text-sm rounded-md px-3 py-2">{error}</div>
          )}
          <div>
            <label className="label">Organization name</label>
            <input required value={form.organization_name} onChange={update("organization_name")} className="input" />
          </div>
          <div>
            <label className="label">Your full name</label>
            <input required value={form.full_name} onChange={update("full_name")} className="input" />
          </div>
          <div>
            <label className="label">Email</label>
            <input type="email" required value={form.email} onChange={update("email")} className="input" />
          </div>
          <div>
            <label className="label">Password</label>
            <input
              type="password"
              required
              minLength={8}
              value={form.password}
              onChange={update("password")}
              className="input"
            />
            <p className="text-xs text-slate-500 mt-1">At least 8 characters.</p>
          </div>
          <button type="submit" disabled={loading} className="btn-primary w-full">
            {loading ? "Creating…" : "Create account"}
          </button>
          <div className="text-center text-sm text-slate-500">
            Already have an account?{" "}
            <Link to="/login" className="text-brand-600 hover:underline">
              Sign in
            </Link>
          </div>
        </form>
      </div>
    </div>
  );
}
