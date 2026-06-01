import { useState } from "react";
import { Copy, Eye, EyeOff, Key, Plus, Trash2, X } from "lucide-react";
import { format } from "date-fns";
import { apiTokensApi } from "../api/client";
import { Badge, Button, Card, EmptyState, Input, Modal, PageHeader, Spinner, AlertBanner } from "../components/ui";
import { useAsync } from "../hooks";
import type { APIToken } from "../types";

export default function ApiTokens() {
  const { data, loading, refetch } = useAsync(() => apiTokensApi.list());
  const [showCreate, setShowCreate] = useState(false);
  const [newToken, setNewToken] = useState<string | null>(null);
  const [form, setForm] = useState({ name: "", scopes: "read:all" });
  const [creating, setCreating] = useState(false);
  const [revealed, setRevealed] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleCreate = async () => {
    if (!form.name.trim()) return;
    setCreating(true);
    try {
      const res = await apiTokensApi.create({ name: form.name, scopes: form.scopes });
      setNewToken(res.data.full_token ?? null);
      setShowCreate(false);
      setForm({ name: "", scopes: "read:all" });
      refetch();
    } finally {
      setCreating(false);
    }
  };

  const handleRevoke = async (id: number) => {
    if (!confirm("Revoke this token? Apps using it will lose access immediately.")) return;
    await apiTokensApi.revoke(id);
    refetch();
  };

  const copy = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const tokens: APIToken[] = data ?? [];

  return (
    <div className="p-6 max-w-4xl mx-auto space-y-4">
      <PageHeader
        title="API Tokens"
        subtitle="Programmatic access to your PeopleSense workspace"
        actions={
          <Button size="sm" icon={<Plus size={14} />} onClick={() => setShowCreate(true)}>
            New token
          </Button>
        }
      />

      {/* Newly created token — show once */}
      {newToken && (
        <Card padding={false}>
          <div
            className="flex items-center justify-between px-5 py-3.5"
            style={{ borderBottom: "1px solid var(--border-subtle)", backgroundColor: "var(--success-dim)" }}
          >
            <div className="flex items-center gap-2">
              <Key size={14} style={{ color: "var(--success)" }} />
              <p className="text-xs font-semibold" style={{ color: "var(--success)" }}>
                Token created — copy it now. This is the only time it will be shown.
              </p>
            </div>
            <button onClick={() => setNewToken(null)} style={{ color: "var(--text-muted)" }}>
              <X size={14} />
            </button>
          </div>
          <div className="px-5 py-4">
            <div
              className="flex items-center gap-2 rounded-lg px-3 py-2.5"
              style={{
                backgroundColor: "var(--bg-subtle)",
                border: "1px solid var(--border-base)",
              }}
            >
              <code className="text-xs flex-1 break-all font-mono" style={{ color: "var(--text-primary)" }}>
                {revealed ? newToken : `${newToken.slice(0, 14)}${"•".repeat(36)}`}
              </code>
              <button
                onClick={() => setRevealed((r) => !r)}
                className="p-1.5 rounded-md transition-colors flex-shrink-0"
                style={{ color: "var(--text-muted)" }}
                onMouseEnter={(e) => (e.currentTarget.style.color = "var(--text-primary)")}
                onMouseLeave={(e) => (e.currentTarget.style.color = "var(--text-muted)")}
              >
                {revealed ? <EyeOff size={13} /> : <Eye size={13} />}
              </button>
              <button
                onClick={() => copy(newToken)}
                className="p-1.5 rounded-md transition-colors flex-shrink-0"
                style={{ color: copied ? "var(--success)" : "var(--text-muted)" }}
                onMouseEnter={(e) => { if (!copied) (e.currentTarget.style.color = "var(--accent)"); }}
                onMouseLeave={(e) => { if (!copied) (e.currentTarget.style.color = "var(--text-muted)"); }}
              >
                <Copy size={13} />
              </button>
            </div>
            {copied && (
              <p className="text-2xs mt-1.5" style={{ color: "var(--success)" }}>
                Copied to clipboard
              </p>
            )}
          </div>
        </Card>
      )}

      {/* Token list */}
      <Card padding={false}>
        {loading ? (
          <div className="flex justify-center py-12">
            <Spinner />
          </div>
        ) : tokens.length === 0 ? (
          <EmptyState
            icon={<Key size={28} />}
            title="No API tokens"
            description="Create a token to access the PeopleSense API programmatically from your applications."
            action={
              <Button size="sm" icon={<Plus size={13} />} onClick={() => setShowCreate(true)}>
                Create token
              </Button>
            }
          />
        ) : (
          <>
            {/* Column headers */}
            <div
              className="grid px-5 py-3 text-2xs font-semibold uppercase tracking-wider"
              style={{
                gridTemplateColumns: "2fr 120px 1fr 120px 80px 48px",
                borderBottom: "1px solid var(--border-subtle)",
                color: "var(--text-muted)",
              }}
            >
              <span>Name</span>
              <span>Prefix</span>
              <span>Scopes</span>
              <span>Last used</span>
              <span>Status</span>
              <span />
            </div>

            {tokens.map((t) => (
              <div
                key={t.id}
                className="grid items-center px-5 py-3.5 transition-colors"
                style={{
                  gridTemplateColumns: "2fr 120px 1fr 120px 80px 48px",
                  borderBottom: "1px solid var(--border-subtle)",
                }}
                onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "var(--bg-subtle)")}
                onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
              >
                <span className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
                  {t.name}
                </span>

                <code
                  className="text-2xs font-mono px-2 py-0.5 rounded-md"
                  style={{
                    backgroundColor: "var(--bg-subtle)",
                    color: "var(--text-secondary)",
                    border: "1px solid var(--border-base)",
                  }}
                >
                  {t.prefix}
                </code>

                <span className="text-xs" style={{ color: "var(--text-muted)" }}>
                  {t.scopes}
                </span>

                <span className="text-2xs font-mono" style={{ color: "var(--text-muted)" }}>
                  {t.last_used_at ? format(new Date(t.last_used_at), "MMM d, yyyy") : "Never"}
                </span>

                <Badge color={t.is_active ? "green" : "gray"}>
                  {t.is_active ? "Active" : "Revoked"}
                </Badge>

                <div className="flex justify-end">
                  <button
                    onClick={() => handleRevoke(t.id)}
                    className="p-1.5 rounded-md transition-colors"
                    style={{ color: "var(--text-muted)" }}
                    onMouseEnter={(e) => {
                      (e.currentTarget as HTMLElement).style.color = "var(--danger)";
                      (e.currentTarget as HTMLElement).style.backgroundColor = "var(--danger-dim)";
                    }}
                    onMouseLeave={(e) => {
                      (e.currentTarget as HTMLElement).style.color = "var(--text-muted)";
                      (e.currentTarget as HTMLElement).style.backgroundColor = "transparent";
                    }}
                    title="Revoke token"
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
              </div>
            ))}
          </>
        )}
      </Card>

      {/* Create modal */}
      <Modal
        open={showCreate}
        onClose={() => setShowCreate(false)}
        title="Create API Token"
        footer={
          <>
            <Button variant="secondary" size="sm" onClick={() => setShowCreate(false)}>
              Cancel
            </Button>
            <Button size="sm" loading={creating} onClick={handleCreate} disabled={!form.name.trim()}>
              Create token
            </Button>
          </>
        }
      >
        <div className="space-y-4">
          <AlertBanner type="warn">
            The full token value is shown once upon creation. Store it in a secrets manager.
          </AlertBanner>

          <Input
            label="Token name"
            placeholder="e.g. Dashboard integration"
            value={form.name}
            onChange={(e) => setForm((f) => ({ ...f, name: e.target.value }))}
          />
          <Input
            label="Scopes"
            placeholder="read:all"
            value={form.scopes}
            onChange={(e) => setForm((f) => ({ ...f, scopes: e.target.value }))}
            helper="Comma-separated — e.g. read:detections,read:analytics"
          />
        </div>
      </Modal>
    </div>
  );
}