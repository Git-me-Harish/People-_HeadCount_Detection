import { useState } from "react";
import { Bell, Check, CheckCheck, Trash2 } from "lucide-react";
import { format } from "date-fns";
import { notificationsApi } from "../api/client";
import { Button, Card, EmptyState, PageHeader, Spinner, Badge } from "../components/ui";
import { useAsync } from "../hooks";
import type { Notification } from "../types";

const TYPE_COLORS: Record<string, string> = {
  alert:   "red",
  info:    "accent",
  system:  "gray",
  warning: "yellow",
};

export default function Notifications() {
  const [unreadOnly, setUnreadOnly] = useState(false);
  const { data, loading, refetch } = useAsync(
    () => notificationsApi.list(unreadOnly),
    [unreadOnly]
  );

  const markRead = async (id: number) => {
    await notificationsApi.markRead([id]);
    refetch();
  };

  const markAll = async () => {
    await notificationsApi.markAllRead();
    refetch();
  };

  const remove = async (id: number) => {
    await notificationsApi.delete(id);
    refetch();
  };

  const notifications: Notification[] = data ?? [];
  const unread = notifications.filter((n) => !n.is_read).length;

  return (
    <div className="p-6 max-w-3xl mx-auto space-y-4">
      <PageHeader
        title="Notifications"
        subtitle={unread > 0 ? `${unread} unread message${unread > 1 ? "s" : ""}` : "All caught up"}
        actions={
          <div className="flex items-center gap-2">
            <button
              onClick={() => setUnreadOnly((u) => !u)}
              className="text-xs px-3 py-1.5 rounded-lg border transition-all"
              style={{
                borderColor: unreadOnly ? "var(--accent)" : "var(--border-base)",
                backgroundColor: unreadOnly ? "var(--accent-dim)" : "transparent",
                color: unreadOnly ? "var(--accent)" : "var(--text-muted)",
              }}
            >
              Unread only
            </button>
            {unread > 0 && (
              <Button variant="secondary" size="sm" icon={<CheckCheck size={13} />} onClick={markAll}>
                Mark all read
              </Button>
            )}
          </div>
        }
      />

      <Card padding={false}>
        {loading ? (
          <div className="flex justify-center py-12">
            <Spinner />
          </div>
        ) : notifications.length === 0 ? (
          <EmptyState
            icon={<Bell size={28} />}
            title="No notifications"
            description={
              unreadOnly
                ? "No unread notifications."
                : "Alerts and system messages will appear here."
            }
          />
        ) : (
          <div>
            {notifications.map((n) => (
              <div
                key={n.id}
                className="flex items-start gap-4 px-5 py-4 transition-colors"
                style={{
                  borderBottom: "1px solid var(--border-subtle)",
                  backgroundColor: !n.is_read ? "var(--accent-dim)" : "transparent",
                }}
                onMouseEnter={(e) => {
                  if (n.is_read) (e.currentTarget.style.backgroundColor = "var(--bg-subtle)");
                }}
                onMouseLeave={(e) => {
                  (e.currentTarget.style.backgroundColor = !n.is_read ? "var(--accent-dim)" : "transparent");
                }}
              >
                {/* Unread indicator */}
                <div className="flex-shrink-0 mt-1.5">
                  <span
                    className="inline-block w-1.5 h-1.5 rounded-full"
                    style={{
                      backgroundColor: n.is_read ? "var(--border-base)" : "var(--accent)",
                    }}
                  />
                </div>

                {/* Content */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-0.5">
                    <p className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
                      {n.title}
                    </p>
                    {n.type && (
                      <Badge color={TYPE_COLORS[n.type] ?? "gray"}>{n.type}</Badge>
                    )}
                  </div>
                  <p className="text-xs leading-relaxed" style={{ color: "var(--text-secondary)" }}>
                    {n.body}
                  </p>
                  <p className="text-2xs mt-1.5 font-mono" style={{ color: "var(--text-muted)" }}>
                    {format(new Date(n.created_at), "MMM d, yyyy · HH:mm")}
                  </p>
                </div>

                {/* Actions */}
                <div className="flex items-center gap-0.5 flex-shrink-0">
                  {!n.is_read && (
                    <button
                      onClick={() => markRead(n.id)}
                      className="p-1.5 rounded-md transition-colors"
                      style={{ color: "var(--text-muted)" }}
                      onMouseEnter={(e) => {
                        (e.currentTarget as HTMLElement).style.color = "var(--accent)";
                        (e.currentTarget as HTMLElement).style.backgroundColor = "var(--accent-dim)";
                      }}
                      onMouseLeave={(e) => {
                        (e.currentTarget as HTMLElement).style.color = "var(--text-muted)";
                        (e.currentTarget as HTMLElement).style.backgroundColor = "transparent";
                      }}
                      title="Mark as read"
                    >
                      <Check size={13} />
                    </button>
                  )}
                  <button
                    onClick={() => remove(n.id)}
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
                    title="Delete"
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}