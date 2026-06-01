import { useEffect, useState } from "react";
import { ExternalLink, User, Cpu, Code2 } from "lucide-react";
import { api } from "../api/client";
import { useAuthStore } from "../auth/store";
import { Card, PageHeader, Button, Badge } from "../components/ui";

function ProfileRow({ label, value }: { label: string; value?: string | number }) {
  return (
    <div
      className="flex items-center justify-between py-3"
      style={{ borderBottom: "1px solid var(--border-subtle)" }}
    >
      <span className="text-xs font-medium uppercase tracking-wider" style={{ color: "var(--text-muted)" }}>
        {label}
      </span>
      <span className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
        {value ?? "—"}
      </span>
    </div>
  );
}

export default function Settings() {
  const user = useAuthStore((s) => s.user);
  const [status, setStatus] = useState<any>(null);

  useEffect(() => {
    api.get("/detect/status").then((r) => setStatus(r.data)).catch(() => {});
  }, []);

  return (
    <div className="p-6 max-w-3xl mx-auto space-y-5">
      <PageHeader
        title="Settings"
        subtitle="Account profile and detector configuration"
      />

      {/* Profile card */}
      <Card padding={false}>
        <div
          className="flex items-center gap-3 px-5 py-4"
          style={{ borderBottom: "1px solid var(--border-subtle)" }}
        >
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center"
            style={{ backgroundColor: "var(--accent-dim)" }}
          >
            <User size={16} style={{ color: "var(--accent)" }} />
          </div>
          <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: "var(--text-muted)" }}>
            Profile
          </p>
        </div>
        <div className="px-5">
          <ProfileRow label="Full name"        value={user?.full_name} />
          <ProfileRow label="Email"            value={user?.email} />
          <ProfileRow label="Role"             value={user?.role} />
          <ProfileRow label="Organization ID"  value={user?.organization_id ? `#${user.organization_id}` : undefined} />
        </div>
      </Card>

      {/* Detector status */}
      <Card padding={false}>
        <div
          className="flex items-center gap-3 px-5 py-4"
          style={{ borderBottom: "1px solid var(--border-subtle)" }}
        >
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center"
            style={{ backgroundColor: "var(--bg-subtle)" }}
          >
            <Cpu size={16} style={{ color: "var(--text-muted)" }} />
          </div>
          <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: "var(--text-muted)" }}>
            Detector Status
          </p>
          {status && (
            <Badge color="green">Online</Badge>
          )}
        </div>
        <div className="p-5">
          {status ? (
            <pre
              className="rounded-lg p-4 text-xs overflow-x-auto font-mono leading-relaxed"
              style={{
                backgroundColor: "var(--bg-subtle)",
                border: "1px solid var(--border-base)",
                color: "var(--text-secondary)",
              }}
            >
              {JSON.stringify(status, null, 2)}
            </pre>
          ) : (
            <div className="text-sm" style={{ color: "var(--text-muted)" }}>
              Loading detector status...
            </div>
          )}
        </div>
      </Card>

      {/* API docs */}
      <Card padding={false}>
        <div
          className="flex items-center gap-3 px-5 py-4"
          style={{ borderBottom: "1px solid var(--border-subtle)" }}
        >
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center"
            style={{ backgroundColor: "var(--bg-subtle)" }}
          >
            <Code2 size={16} style={{ color: "var(--text-muted)" }} />
          </div>
          <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: "var(--text-muted)" }}>
            API Documentation
          </p>
        </div>
        <div className="px-5 py-4">
          <p className="text-sm mb-4" style={{ color: "var(--text-secondary)" }}>
            Explore and test the REST API directly via auto-generated documentation.
          </p>
          <div className="flex gap-3">
            <Button
              variant="secondary"
              size="sm"
              icon={<ExternalLink size={12} />}
              onClick={() => window.open("/docs", "_blank")}
            >
              Swagger UI
            </Button>
            <Button
              variant="secondary"
              size="sm"
              icon={<ExternalLink size={12} />}
              onClick={() => window.open("/redoc", "_blank")}
            >
              ReDoc
            </Button>
          </div>
        </div>
      </Card>
    </div>
  );
}