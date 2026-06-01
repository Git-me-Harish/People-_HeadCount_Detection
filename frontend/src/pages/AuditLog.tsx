import { format } from "date-fns";
import { Shield } from "lucide-react";
import { auditApi } from "../api/client";
import { Badge, Card, EmptyState, PageHeader, Spinner } from "../components/ui";
import { useAsync } from "../hooks";

const ACTION_COLOR: Record<string, string> = {
  create: "green",
  delete: "red",
  update: "accent",
  login:  "blue",
  revoke: "yellow",
};

export default function AuditLog() {
  const { data, loading } = useAsync(() => auditApi.list({ limit: 100 }));

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-4">
      <PageHeader
        title="Audit Log"
        subtitle="Immutable record of every state-changing action in your workspace"
      />

      <Card padding={false}>
        {loading ? (
          <div className="flex justify-center py-12">
            <Spinner />
          </div>
        ) : !data || data.length === 0 ? (
          <EmptyState
            icon={<Shield size={28} />}
            title="No audit events yet"
            description="All create, update, and delete actions will appear here for compliance review."
          />
        ) : (
          <>
            {/* Column headers */}
            <div
              className="grid px-5 py-3 text-2xs font-semibold uppercase tracking-wider"
              style={{
                gridTemplateColumns: "160px 1fr 1fr 1fr 100px",
                borderBottom: "1px solid var(--border-subtle)",
                color: "var(--text-muted)",
              }}
            >
              <span>Time</span>
              <span>Actor</span>
              <span>Action</span>
              <span>Resource</span>
              <span>IP address</span>
            </div>

            {data.map((entry) => {
              const verb = entry.action.split(".").pop() ?? entry.action;
              return (
                <div
                  key={entry.id}
                  className="grid items-center px-5 py-3 transition-colors"
                  style={{
                    gridTemplateColumns: "160px 1fr 1fr 1fr 100px",
                    borderBottom: "1px solid var(--border-subtle)",
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "var(--bg-subtle)")}
                  onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
                >
                  <span className="text-2xs font-mono whitespace-nowrap" style={{ color: "var(--text-muted)" }}>
                    {format(new Date(entry.created_at), "MMM d, HH:mm:ss")}
                  </span>

                  <span className="text-xs truncate pr-4" style={{ color: "var(--text-secondary)" }}>
                    {entry.actor_email ?? "system"}
                  </span>

                  <div>
                    <Badge color={ACTION_COLOR[verb] ?? "gray"}>
                      {entry.action}
                    </Badge>
                  </div>

                  <span className="text-xs font-mono" style={{ color: "var(--text-secondary)" }}>
                    {entry.resource_type}
                    {entry.resource_id ? (
                      <span style={{ color: "var(--text-muted)" }}> #{entry.resource_id}</span>
                    ) : null}
                  </span>

                  <span className="text-2xs font-mono" style={{ color: "var(--text-muted)" }}>
                    {entry.ip_address ?? "—"}
                  </span>
                </div>
              );
            })}
          </>
        )}
      </Card>
    </div>
  );
}