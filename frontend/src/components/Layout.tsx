import { NavLink, Outlet, useNavigate } from "react-router-dom";
import { useAuthStore } from "../auth/store";

const navItems = [
  { to: "/", label: "Dashboard", icon: "📊" },
  { to: "/image", label: "Image Detect", icon: "🖼️" },
  { to: "/video", label: "Video Detect", icon: "🎬" },
  { to: "/stream", label: "Live Stream", icon: "📡" },
  { to: "/cameras", label: "Cameras", icon: "🎥" },
  { to: "/analytics", label: "Analytics", icon: "📈" },
  { to: "/alerts", label: "Alerts", icon: "🔔" },
  { to: "/settings", label: "Settings", icon: "⚙️" },
];

export default function Layout() {
  const navigate = useNavigate();
  const { user, clear } = useAuthStore();

  const logout = () => {
    clear();
    navigate("/login");
  };

  return (
    <div className="flex h-screen bg-slate-50">
      <aside className="w-60 bg-brand-900 text-white flex flex-col">
        <div className="px-6 py-5 border-b border-white/10">
          <div className="text-xl font-semibold">PeopleSense</div>
          <div className="text-xs text-brand-100/70">Crowd analytics platform</div>
        </div>
        <nav className="flex-1 px-3 py-4 space-y-1">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) =>
                `flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition ${
                  isActive
                    ? "bg-white/15 text-white"
                    : "text-brand-100/80 hover:bg-white/10 hover:text-white"
                }`
              }
            >
              <span aria-hidden>{item.icon}</span>
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="px-4 py-4 border-t border-white/10 text-xs">
          <div className="font-medium">{user?.full_name ?? "—"}</div>
          <div className="text-brand-100/60 truncate">{user?.email}</div>
          <button
            onClick={logout}
            className="mt-3 w-full rounded-md bg-white/10 hover:bg-white/20 px-3 py-1.5 text-xs font-medium"
          >
            Sign out
          </button>
        </div>
      </aside>
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
