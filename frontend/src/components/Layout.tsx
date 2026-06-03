import { useState } from "react";
import { Link, Outlet, useLocation, useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import {
  Activity,
  AlertTriangle,
  BarChart2,
  Bell,
  Camera,
  ChevronLeft,
  ChevronRight,
  FileText,
  Globe,
  Key,
  LogOut,
  Radio,
  Settings,
  Shield,
  Sun,
  Moon,
  Zap,
  ImagePlus,
  Film,
} from "lucide-react";
import { useAuthStore } from "../auth/store";
import { useThemeMode, useUnreadCount } from "../hooks";
import { ROUTES } from "../constants";

interface NavItem {
  to: string;
  icon: React.ReactNode;
  labelKey: string;
  adminOnly?: boolean;
  badge?: number;
  group?: string;
}

function useNavItems(unreadCount: number): NavItem[] {
  const { t: _t } = useTranslation();
  return [
    { to: ROUTES.DASHBOARD,     icon: <Activity size={16} />,     labelKey: "dashboard",     group: "monitor" },
    { to: ROUTES.CAMERAS,       icon: <Camera size={16} />,       labelKey: "cameras",       group: "monitor" },
    { to: ROUTES.LIVE,          icon: <Radio size={16} />,        labelKey: "liveStream",    group: "monitor" },
    { to: ROUTES.IMAGE_DETECT, icon: <ImagePlus size={16} />, labelKey: "imageDetect", group: "monitor" },
    { to: ROUTES.VIDEO_DETECT, icon: <Film size={16} />,       labelKey: "videoDetect", group: "monitor" },
    { to: ROUTES.ANALYTICS,     icon: <BarChart2 size={16} />,    labelKey: "analytics",     group: "intel" },
    { to: ROUTES.ALERTS,        icon: <AlertTriangle size={16} />,labelKey: "alerts",        group: "intel" },
    {
      to: ROUTES.NOTIFICATIONS,
      icon: <Bell size={16} />,
      labelKey: "notifications",
      badge: unreadCount || undefined,
      group: "intel",
    },
    { to: ROUTES.REPORTS,       icon: <FileText size={16} />,     labelKey: "reports",       group: "intel" },
    { to: ROUTES.PUBLIC_PAGE,   icon: <Globe size={16} />,        labelKey: "publicPage",    group: "config" },
    { to: ROUTES.API_TOKENS,    icon: <Key size={16} />,          labelKey: "apiTokens",     adminOnly: true, group: "config" },
    { to: ROUTES.AUDIT,         icon: <Shield size={16} />,       labelKey: "auditLog",      adminOnly: true, group: "config" },
    { to: ROUTES.PLAN,          icon: <Zap size={16} />,          labelKey: "plan",          group: "config" },
    { to: ROUTES.SETTINGS,      icon: <Settings size={16} />,     labelKey: "settings",      group: "config" },
  ];
}

const GROUP_LABELS: Record<string, string> = {
  monitor: "Monitoring",
  intel:   "Intelligence",
  config:  "Configuration",
};

export default function Layout() {
  const { t } = useTranslation();
  const { user, clear } = useAuthStore();
  const navigate = useNavigate();
  const location = useLocation();
  const [collapsed, setCollapsed] = useState(false);
  const [mode, toggleMode] = useThemeMode();
  const { count: unreadCount } = useUnreadCount();
  const navItems = useNavItems(unreadCount);
  const isAdmin = user?.role === "admin";

  const handleSignOut = () => {
    clear();
    navigate(ROUTES.LOGIN);
  };

  // Group nav items
  const groups = ["monitor", "intel", "config"] as const;
  const grouped = groups.reduce(
    (acc, g) => ({
      ...acc,
      [g]: navItems.filter((i) => i.group === g && (!i.adminOnly || isAdmin)),
    }),
    {} as Record<string, NavItem[]>
  );

  const isOps = mode === "ops";

  return (
    <div className="flex h-screen overflow-hidden" style={{ backgroundColor: "var(--bg-base)" }}>
      {/* ── Sidebar ── */}
      <aside
        style={{
          width: collapsed ? "var(--sidebar-collapsed)" : "var(--sidebar-w)",
          backgroundColor: "var(--bg-surface)",
          borderRight: "1px solid var(--border-base)",
        }}
        className="flex flex-col h-full transition-all duration-300 flex-shrink-0 z-20"
      >
        {/* Logo / Brand */}
        <div
          className="flex items-center justify-between px-3 py-4 flex-shrink-0"
          style={{ borderBottom: "1px solid var(--border-subtle)" }}
        >
          {!collapsed && (
            <div className="flex items-center gap-2.5 min-w-0">
              {/* Wordmark logo */}
              <div
                className="w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0"
                style={{ backgroundColor: "var(--accent)", color: "var(--text-inverse)" }}
              >
                <Activity size={14} />
              </div>
              <div className="min-w-0">
                <div
                  className="text-sm font-semibold leading-none tracking-tight"
                  style={{ color: "var(--text-primary)" }}
                >
                  PeopleSense
                </div>
                <div
                  className="text-2xs mt-0.5 truncate"
                  style={{ color: "var(--text-muted)" }}
                >
                  Crowd Intelligence
                </div>
              </div>
            </div>
          )}
          {collapsed && (
            <div
              className="w-7 h-7 rounded-lg flex items-center justify-center mx-auto"
              style={{ backgroundColor: "var(--accent)", color: "var(--text-inverse)" }}
            >
              <Activity size={14} />
            </div>
          )}
          <button
            onClick={() => setCollapsed((c) => !c)}
            className="p-1 rounded-md transition-colors flex-shrink-0"
            style={{ color: "var(--text-muted)" }}
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {collapsed ? <ChevronRight size={13} /> : <ChevronLeft size={13} />}
          </button>
        </div>

        {/* Nav groups */}
        <nav className="flex-1 overflow-y-auto py-3 px-2 space-y-4">
          {groups.map((group) => {
            const items = grouped[group];
            if (!items?.length) return null;
            return (
              <div key={group}>
                {!collapsed && (
                  <p
                    className="px-3 mb-1 text-2xs font-semibold uppercase tracking-widest"
                    style={{ color: "var(--text-muted)" }}
                  >
                    {GROUP_LABELS[group]}
                  </p>
                )}
                <div className="space-y-0.5">
                  {items.map((item) => {
                    const active = location.pathname === item.to;
                    return (
                      <Link
                        key={item.to}
                        to={item.to}
                        title={collapsed ? t(item.labelKey) : undefined}
                        className="flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm transition-all duration-100 group relative"
                        style={{
                          backgroundColor: active ? "var(--accent-dim)" : "transparent",
                          color: active ? "var(--accent)" : "var(--text-secondary)",
                          fontWeight: active ? 500 : 400,
                        }}
                        onMouseEnter={(e) => {
                          if (!active) (e.currentTarget as HTMLElement).style.backgroundColor = "var(--bg-subtle)";
                        }}
                        onMouseLeave={(e) => {
                          if (!active) (e.currentTarget as HTMLElement).style.backgroundColor = "transparent";
                        }}
                      >
                        <span className="flex-shrink-0">{item.icon}</span>
                        {!collapsed && (
                          <span className="truncate text-sm">{t(item.labelKey)}</span>
                        )}
                        {item.badge && item.badge > 0 && (
                          <span
                            className={`${
                              collapsed ? "absolute top-1 right-1" : "ml-auto"
                            } text-white text-2xs rounded-full min-w-[16px] h-4 flex items-center justify-center px-1 font-medium`}
                            style={{ backgroundColor: "var(--danger)", fontSize: "10px" }}
                          >
                            {item.badge > 99 ? "99+" : item.badge}
                          </span>
                        )}
                        {active && (
                          <span
                            className="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-4 rounded-r-full"
                            style={{ backgroundColor: "var(--accent)" }}
                          />
                        )}
                      </Link>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </nav>

        {/* Sidebar footer */}
        <div
          className="p-2 space-y-1 flex-shrink-0"
          style={{ borderTop: "1px solid var(--border-subtle)" }}
        >
          {/* Theme toggle */}
          <button
            onClick={toggleMode}
            className="flex items-center gap-2.5 px-3 py-2 w-full rounded-lg text-sm transition-all duration-100"
            style={{ color: "var(--text-muted)" }}
            onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "var(--bg-subtle)")}
            onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
            aria-label={isOps ? "Switch to light mode" : "Switch to ops mode"}
          >
            {isOps ? <Sun size={15} /> : <Moon size={15} />}
            {!collapsed && (
              <span className="text-xs">{isOps ? "Light Mode" : "Ops Mode"}</span>
            )}
          </button>

          {/* User info */}
          {!collapsed && user && (
            <div className="px-3 py-2 rounded-lg" style={{ backgroundColor: "var(--bg-subtle)" }}>
              <p
                className="text-xs font-medium truncate leading-tight"
                style={{ color: "var(--text-primary)" }}
              >
                {user.full_name}
              </p>
              <p className="text-2xs truncate mt-0.5" style={{ color: "var(--text-muted)" }}>
                {user.email}
              </p>
            </div>
          )}

          {/* Sign out */}
          <button
            onClick={handleSignOut}
            className="flex items-center gap-2.5 px-3 py-2 w-full rounded-lg text-sm transition-all duration-100"
            style={{ color: "var(--danger)" }}
            onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "var(--danger-dim)")}
            onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
          >
            <LogOut size={15} />
            {!collapsed && <span className="text-xs">{t("signOut")}</span>}
          </button>
        </div>
      </aside>

      {/* ── Main content ── */}
      <main className="flex-1 overflow-y-auto">
        <Outlet />
      </main>
    </div>
  );
}
