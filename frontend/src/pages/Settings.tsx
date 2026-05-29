import { useEffect, useState } from "react";
import { api } from "../api/client";
import { useAuthStore } from "../auth/store";

export default function Settings() {
  const user = useAuthStore((s) => s.user);
  const [status, setStatus] = useState<any>(null);

  useEffect(() => {
    api.get("/detect/status").then((r) => setStatus(r.data));
  }, []);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Settings</h1>
        <p className="text-sm text-slate-500">Account & detector configuration.</p>
      </div>

      <div className="card">
        <h2 className="font-semibold mb-3">Profile</h2>
        <dl className="grid grid-cols-1 sm:grid-cols-2 gap-y-2 text-sm">
          <dt className="text-slate-500">Name</dt>
          <dd>{user?.full_name}</dd>
          <dt className="text-slate-500">Email</dt>
          <dd>{user?.email}</dd>
          <dt className="text-slate-500">Role</dt>
          <dd className="capitalize">{user?.role}</dd>
          <dt className="text-slate-500">Organization</dt>
          <dd>#{user?.organization_id}</dd>
        </dl>
      </div>

      <div className="card">
        <h2 className="font-semibold mb-3">Detector</h2>
        {status ? (
          <pre className="bg-slate-50 rounded-md p-3 text-xs overflow-x-auto">
            {JSON.stringify(status, null, 2)}
          </pre>
        ) : (
          <div className="text-sm text-slate-400">Loading…</div>
        )}
      </div>

      <div className="card">
        <h2 className="font-semibold mb-2">API documentation</h2>
        <p className="text-sm text-slate-600">
          Explore and try the REST API via auto-generated docs.
        </p>
        <div className="mt-3 flex gap-3">
          <a className="btn-secondary" href="/docs" target="_blank" rel="noreferrer">
            Swagger UI
          </a>
          <a className="btn-secondary" href="/redoc" target="_blank" rel="noreferrer">
            ReDoc
          </a>
        </div>
      </div>
    </div>
  );
}
