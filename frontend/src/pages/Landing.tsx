import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BarChart2,
  Bell,
  Camera,
  CheckCircle2,
  ChevronRight,
  Globe,
  Radio,
  Shield,
  TrendingUp,
  Users,
  Zap,
} from "lucide-react";
import { ROUTES } from "../constants";

// ─── Tiny hook: detect when element enters viewport ───────────────────────────
function useInView(threshold = 0.15) {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([e]) => { if (e.isIntersecting) { setVisible(true); obs.disconnect(); } },
      { threshold }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, [threshold]);
  return { ref, visible };
}

// ─── Animated counter ─────────────────────────────────────────────────────────
function Counter({ target, suffix = "" }: { target: number; suffix?: string }) {
  const [val, setVal] = useState(0);
  const { ref, visible } = useInView(0.3);
  useEffect(() => {
    if (!visible) return;
    const dur = 1400;
    const start = performance.now();
    const tick = (now: number) => {
      const p = Math.min((now - start) / dur, 1);
      const ease = 1 - Math.pow(1 - p, 3);
      setVal(Math.round(ease * target));
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }, [visible, target]);
  return <span ref={ref}>{val.toLocaleString()}{suffix}</span>;
}

// ─── Mock live dashboard widget ───────────────────────────────────────────────
function DashboardMock() {
  const bars = [42, 68, 55, 80, 61, 93, 74, 58, 82, 67, 91, 76];
  const [active, setActive] = useState(11);
  useEffect(() => {
    const id = setInterval(() => setActive((a) => (a + 1) % bars.length), 900);
    return () => clearInterval(id);
  }, []);
  return (
    <div
      className="rounded-2xl overflow-hidden select-none"
      style={{
        backgroundColor: "#0d1420",
        border: "1px solid #1e2a3d",
        boxShadow: "0 32px 80px rgba(0,0,0,0.6), 0 0 0 1px rgba(0,194,168,0.06)",
      }}
    >
      {/* Title bar */}
      <div className="flex items-center gap-2 px-4 py-3 border-b" style={{ borderColor: "#1a2436" }}>
        <span className="w-3 h-3 rounded-full" style={{ backgroundColor: "#ff5f57" }} />
        <span className="w-3 h-3 rounded-full" style={{ backgroundColor: "#febc2e" }} />
        <span className="w-3 h-3 rounded-full" style={{ backgroundColor: "#28c840" }} />
        <span className="ml-3 text-xs font-mono" style={{ color: "#4a5c78" }}>
          peoplesense — live dashboard
        </span>
        <span className="ml-auto flex items-center gap-1.5 text-xs" style={{ color: "#00c2a8" }}>
          <span className="w-1.5 h-1.5 rounded-full bg-[#00c2a8] animate-pulse" />
          Live
        </span>
      </div>

      {/* KPI row */}
      <div className="grid grid-cols-3 gap-px" style={{ backgroundColor: "#1a2436" }}>
        {[
          { label: "Current Count", value: "247", color: "#00c2a8" },
          { label: "Peak Today", value: "891", color: "#f59e0b" },
          { label: "Cameras Live", value: "12 / 14", color: "#34d399" },
        ].map((k) => (
          <div key={k.label} className="px-4 py-3.5" style={{ backgroundColor: "#0d1420" }}>
            <p className="text-2xs uppercase tracking-widest mb-1" style={{ color: "#4a5c78", fontSize: "9px" }}>
              {k.label}
            </p>
            <p className="text-xl font-mono font-medium" style={{ color: k.color, letterSpacing: "-0.03em" }}>
              {k.value}
            </p>
          </div>
        ))}
      </div>

      {/* Chart */}
      <div className="px-4 pt-4 pb-2">
        <p className="text-xs mb-3" style={{ color: "#4a5c78" }}>People count — last 12 buckets</p>
        <div className="flex items-end gap-1.5" style={{ height: 72 }}>
          {bars.map((h, i) => (
            <div key={i} className="flex-1 rounded-sm transition-all duration-500 relative" style={{
              height: `${h}%`,
              backgroundColor: i === active ? "#00c2a8" : "rgba(0,194,168,0.18)",
              boxShadow: i === active ? "0 0 12px rgba(0,194,168,0.5)" : "none",
            }} />
          ))}
        </div>
        <div className="flex justify-between mt-1.5">
          {["12h", "10h", "8h", "6h", "4h", "2h", "now"].map((t) => (
            <span key={t} style={{ color: "#2a3a52", fontSize: "8px", fontFamily: "DM Mono" }}>{t}</span>
          ))}
        </div>
      </div>

      {/* Alert row */}
      <div className="mx-4 mb-4 mt-2 rounded-lg px-3 py-2.5 flex items-center gap-2.5"
        style={{ backgroundColor: "rgba(245,158,11,0.08)", border: "1px solid rgba(245,158,11,0.2)" }}>
        <AlertTriangle size={12} style={{ color: "#f59e0b", flexShrink: 0 }} />
        <span className="text-xs" style={{ color: "#f59e0b" }}>
          Gate A density exceeded threshold · 3 min ago
        </span>
      </div>
    </div>
  );
}

// ─── Mock alert feed widget ───────────────────────────────────────────────────
function AlertFeedMock() {
  const items = [
    { time: "14:32", cam: "Main Entrance", count: 312, type: "critical" },
    { time: "14:18", cam: "Platform 3",    count: 198, type: "warn" },
    { time: "14:05", cam: "Food Court",    count: 87,  type: "ok" },
    { time: "13:51", cam: "Gate B",        count: 243, type: "warn" },
  ];
  const colors = { critical: "#f87171", warn: "#f59e0b", ok: "#34d399" };
  return (
    <div className="rounded-2xl overflow-hidden"
      style={{ backgroundColor: "#0d1420", border: "1px solid #1e2a3d", boxShadow: "0 32px 80px rgba(0,0,0,0.5)" }}>
      <div className="px-4 py-3 border-b flex items-center justify-between" style={{ borderColor: "#1a2436" }}>
        <span className="text-xs font-semibold" style={{ color: "#e8edf5" }}>Alert Feed</span>
        <span className="text-2xs px-2 py-0.5 rounded-md font-medium"
          style={{ backgroundColor: "rgba(248,113,113,0.12)", color: "#f87171", border: "1px solid rgba(248,113,113,0.2)" }}>
          2 active
        </span>
      </div>
      {items.map((item, i) => (
        <div key={i} className="flex items-center gap-3 px-4 py-3 border-b" style={{ borderColor: "#0f1825" }}>
          <span className="w-1.5 h-1.5 rounded-full flex-shrink-0"
            style={{ backgroundColor: colors[item.type as keyof typeof colors] }} />
          <div className="flex-1 min-w-0">
            <p className="text-xs font-medium" style={{ color: "#c8d5e8" }}>{item.cam}</p>
            <p className="text-2xs font-mono" style={{ color: "#4a5c78" }}>{item.time}</p>
          </div>
          <span className="text-sm font-mono font-medium" style={{ color: colors[item.type as keyof typeof colors] }}>
            {item.count}
          </span>
        </div>
      ))}
    </div>
  );
}

// ─── Feature section row ──────────────────────────────────────────────────────
function FeatureRow({
  eyebrow,
  title,
  description,
  bullets,
  visual,
  reverse = false,
}: {
  eyebrow: string;
  title: string;
  description: string;
  bullets: string[];
  visual: React.ReactNode;
  reverse?: boolean;
}) {
  const { ref, visible } = useInView();
  return (
    <div
      ref={ref}
      className={`grid grid-cols-1 lg:grid-cols-2 gap-16 items-center transition-all duration-700 ${
        visible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-8"
      } ${reverse ? "lg:grid-flow-dense" : ""}`}
    >
      <div className={reverse ? "lg:col-start-2" : ""}>
        <p className="text-xs font-semibold uppercase tracking-widest mb-3" style={{ color: "#00c2a8" }}>
          {eyebrow}
        </p>
        <h3 className="text-3xl font-bold leading-tight mb-4" style={{ color: "#e8edf5" }}>
          {title}
        </h3>
        <p className="text-base leading-relaxed mb-6" style={{ color: "#8a9bb5" }}>
          {description}
        </p>
        <ul className="space-y-2.5">
          {bullets.map((b) => (
            <li key={b} className="flex items-start gap-2.5 text-sm" style={{ color: "#8a9bb5" }}>
              <CheckCircle2 size={15} className="flex-shrink-0 mt-0.5" style={{ color: "#00c2a8" }} />
              {b}
            </li>
          ))}
        </ul>
      </div>
      <div className={reverse ? "lg:col-start-1 lg:row-start-1" : ""}>{visual}</div>
    </div>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────
export default function Landing() {
  const heroRef = useRef<HTMLHeadingElement>(null);

  // Subtle parallax on hero text
  useEffect(() => {
    const handler = () => {
      if (heroRef.current) {
        heroRef.current.style.transform = `translateY(${window.scrollY * 0.08}px)`;
      }
    };
    window.addEventListener("scroll", handler, { passive: true });
    return () => window.removeEventListener("scroll", handler);
  }, []);

  return (
    <div
      className="min-h-screen overflow-x-hidden"
      style={{ backgroundColor: "#080c12", color: "#e8edf5", fontFamily: "'DM Sans', sans-serif" }}
    >
      {/* ── Navbar ── */}
      <header
        className="fixed top-0 left-0 right-0 z-50 px-6 py-0"
        style={{ borderBottom: "1px solid rgba(30,42,61,0.8)", backdropFilter: "blur(20px)", backgroundColor: "rgba(8,12,18,0.85)" }}
      >
        <div className="max-w-6xl mx-auto flex items-center justify-between h-14">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg flex items-center justify-center" style={{ backgroundColor: "#00c2a8" }}>
              <Activity size={14} style={{ color: "#080c12" }} />
            </div>
            <span className="font-semibold text-sm tracking-tight" style={{ color: "#e8edf5" }}>PeopleSense</span>
          </div>

          <nav className="hidden md:flex items-center gap-1">
            {["Features", "Use Cases", "Pricing"].map((item) => (
              <a
                key={item}
                href={`#${item.toLowerCase().replace(" ", "-")}`}
                className="px-3 py-1.5 text-sm rounded-lg transition-colors"
                style={{ color: "#8a9bb5" }}
                onMouseEnter={(e) => (e.currentTarget.style.color = "#e8edf5")}
                onMouseLeave={(e) => (e.currentTarget.style.color = "#8a9bb5")}
              >
                {item}
              </a>
            ))}
          </nav>

          <div className="flex items-center gap-2">
            <Link
              to={ROUTES.LOGIN}
              className="px-3 py-1.5 text-sm rounded-lg transition-colors"
              style={{ color: "#8a9bb5" }}
            >
              Sign in
            </Link>
            <Link
              to={ROUTES.REGISTER}
              className="flex items-center gap-1.5 px-4 py-1.5 text-sm font-medium rounded-lg transition-all"
              style={{ backgroundColor: "#00c2a8", color: "#080c12" }}
            >
              Get started
              <ArrowRight size={13} />
            </Link>
          </div>
        </div>
      </header>

      {/* ── Hero ── */}
      <section className="relative flex flex-col items-center justify-center text-center min-h-screen px-6 pb-20 pt-24 overflow-hidden">
        {/* Background glow */}
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            background: "radial-gradient(ellipse 80% 60% at 50% -10%, rgba(0,194,168,0.12) 0%, transparent 70%)",
          }}
        />
        {/* Grid texture */}
        <div
          className="absolute inset-0 pointer-events-none opacity-[0.03]"
          style={{
            backgroundImage: "linear-gradient(rgba(0,194,168,1) 1px, transparent 1px), linear-gradient(90deg, rgba(0,194,168,1) 1px, transparent 1px)",
            backgroundSize: "60px 60px",
          }}
        />

        <div className="relative max-w-5xl mx-auto">
          {/* Badge */}
          <div
            className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium mb-8"
            style={{ backgroundColor: "rgba(0,194,168,0.08)", border: "1px solid rgba(0,194,168,0.2)", color: "#00c2a8" }}
          >
            <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ backgroundColor: "#00c2a8" }} />
            Now monitoring 40M+ crowd events per month
          </div>

          {/* Headline */}
          <h1
            ref={heroRef}
            className="font-bold leading-[1.05] tracking-tight mb-6"
            style={{ fontSize: "clamp(2.8rem, 7vw, 5.5rem)", color: "#e8edf5" }}
          >
            Crowd intelligence
            <br />
            <span
              style={{
                background: "linear-gradient(90deg, #00c2a8 0%, #4af0d8 50%, #00c2a8 100%)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
                backgroundSize: "200% auto",
                animation: "shimmer 3s linear infinite",
              }}
            >
              for safer spaces.
            </span>
          </h1>

          <p
            className="text-lg leading-relaxed mb-10 mx-auto"
            style={{ color: "#8a9bb5", maxWidth: "580px" }}
          >
            Real-time people counting, density alerts, and predictive analytics
            for temples, transit hubs, hospitals, and large venues.
            Privacy-first. No face data. Ever.
          </p>

          {/* CTAs */}
          <div className="flex flex-wrap items-center justify-center gap-3 mb-20">
            <Link
              to={ROUTES.REGISTER}
              className="flex items-center gap-2 px-6 py-3 rounded-xl font-semibold text-sm transition-all duration-200"
              style={{ backgroundColor: "#00c2a8", color: "#080c12", boxShadow: "0 0 32px rgba(0,194,168,0.3)" }}
              onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.boxShadow = "0 0 48px rgba(0,194,168,0.5)"; }}
              onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.boxShadow = "0 0 32px rgba(0,194,168,0.3)"; }}
            >
              Start for free
              <ArrowRight size={15} />
            </Link>
            <Link
              to={ROUTES.LOGIN}
              className="flex items-center gap-2 px-6 py-3 rounded-xl font-medium text-sm transition-all duration-200"
              style={{ border: "1px solid #1e2a3d", color: "#8a9bb5", backgroundColor: "rgba(255,255,255,0.02)" }}
              onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.borderColor = "#2d3f58"; (e.currentTarget as HTMLElement).style.color = "#e8edf5"; }}
              onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.borderColor = "#1e2a3d"; (e.currentTarget as HTMLElement).style.color = "#8a9bb5"; }}
            >
              View live demo
            </Link>
          </div>

          {/* Hero visual — dashboard mock */}
          <div className="relative mx-auto" style={{ maxWidth: "780px" }}>
            {/* Glow behind card */}
            <div
              className="absolute inset-0 rounded-2xl blur-3xl"
              style={{ background: "radial-gradient(ellipse at 50% 80%, rgba(0,194,168,0.15), transparent 70%)", transform: "scaleY(0.6) translateY(30%)" }}
            />
            <DashboardMock />
          </div>
        </div>
      </section>

      {/* ── Social proof numbers ── */}
      <section className="px-6 py-20" style={{ borderTop: "1px solid #1a2436", borderBottom: "1px solid #1a2436" }}>
        <div className="max-w-4xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-8 text-center">
          {[
            { target: 40,   suffix: "M+", label: "Events processed / month" },
            { target: 500,  suffix: "ms", label: "Average detection latency" },
            { target: 8,    suffix: "+",  label: "Industry verticals" },
            { target: 99.9, suffix: "%",  label: "Uptime SLA" },
          ].map((s) => (
            <div key={s.label}>
              <p
                className="font-mono font-bold mb-1"
                style={{ fontSize: "clamp(2rem,4vw,3rem)", color: "#e8edf5", letterSpacing: "-0.04em" }}
              >
                <Counter target={s.target} suffix={s.suffix} />
              </p>
              <p className="text-xs" style={{ color: "#4a5c78" }}>{s.label}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── Feature sections ── */}
      <section id="features" className="px-6 py-28 max-w-6xl mx-auto space-y-32">
        <FeatureRow
          eyebrow="Real-time Monitoring"
          title={"Know exactly who is in your space,\nright now."}
          description="Sub-second head counts from any IP camera. Live density overlays, multi-camera grid, and instant status across your entire facility — all in one dashboard."
          bullets={[
            "Works with any RTSP / HTTP camera stream",
            "Sub-500ms detection latency via WebSocket",
            "Density heat maps per zone",
            "Mobile-ready for field operators",
          ]}
          visual={<DashboardMock />}
        />

        <FeatureRow
          eyebrow="Smart Alerts"
          title={"Get notified before a\nsituation becomes critical."}
          description="Configurable threshold and anomaly alerts delivered to Slack, MS Teams, email, or custom webhooks — in under five seconds of the event firing."
          bullets={[
            "Threshold and z-score anomaly triggers",
            "Slack, Teams, email, and webhook delivery",
            "Per-camera cooldown and escalation rules",
            "Alert history with full audit trail",
          ]}
          visual={<AlertFeedMock />}
          reverse
        />

        <FeatureRow
          eyebrow="Predictive Analytics"
          title={"Forecast surges before\nthey happen."}
          description="Hourly trend charts, 7-day rolling averages, and anomaly detection powered by z-score analysis. Export PDF or CSV reports for operations and compliance teams."
          bullets={[
            "7, 14, 30, and 90-day trend windows",
            "Peak forecasting with confidence bands",
            "Per-camera breakdown in PDF reports",
            "CSV export for custom BI pipelines",
          ]}
          visual={
            <div
              className="rounded-2xl p-6 overflow-hidden"
              style={{ backgroundColor: "#0d1420", border: "1px solid #1e2a3d", boxShadow: "0 32px 80px rgba(0,0,0,0.5)" }}
            >
              <p className="text-xs mb-4" style={{ color: "#4a5c78" }}>People count — last 30 days</p>
              <svg viewBox="0 0 400 120" className="w-full">
                <defs>
                  <linearGradient id="chartGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#00c2a8" stopOpacity="0.3" />
                    <stop offset="100%" stopColor="#00c2a8" stopOpacity="0" />
                  </linearGradient>
                </defs>
                {/* Grid lines */}
                {[0, 30, 60, 90].map((y) => (
                  <line key={y} x1="0" y1={y} x2="400" y2={y} stroke="#1a2436" strokeWidth="1" />
                ))}
                {/* Area fill */}
                <path
                  d="M0,90 C20,85 30,70 50,65 S80,40 100,50 S130,30 160,35 S200,20 220,25 S260,15 280,20 S320,35 350,30 S380,20 400,18 L400,120 L0,120 Z"
                  fill="url(#chartGrad)"
                />
                {/* Line */}
                <path
                  d="M0,90 C20,85 30,70 50,65 S80,40 100,50 S130,30 160,35 S200,20 220,25 S260,15 280,20 S320,35 350,30 S380,20 400,18"
                  fill="none"
                  stroke="#00c2a8"
                  strokeWidth="2"
                />
                {/* Peak dot */}
                <circle cx="280" cy="20" r="4" fill="#00c2a8" />
                <circle cx="280" cy="20" r="8" fill="rgba(0,194,168,0.2)" />
                {/* Peak label */}
                <text x="292" y="16" fill="#00c2a8" fontSize="9" fontFamily="DM Mono">peak 891</text>
              </svg>
              {/* Stats row */}
              <div className="grid grid-cols-3 gap-3 mt-4">
                {[
                  { label: "Avg daily peak", value: "612" },
                  { label: "Total detections", value: "284K" },
                  { label: "Anomalies flagged", value: "7" },
                ].map((s) => (
                  <div key={s.label} className="rounded-xl px-3 py-2.5" style={{ backgroundColor: "#111820" }}>
                    <p className="text-2xs mb-1" style={{ color: "#4a5c78", fontSize: "9px" }}>{s.label}</p>
                    <p className="text-base font-mono font-medium" style={{ color: "#e8edf5" }}>{s.value}</p>
                  </div>
                ))}
              </div>
            </div>
          }
        />
      </section>

      {/* ── Use cases ── */}
      <section id="use-cases" className="px-6 py-24" style={{ backgroundColor: "#0a0e16", borderTop: "1px solid #1a2436" }}>
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-14">
            <p className="text-xs font-semibold uppercase tracking-widest mb-3" style={{ color: "#00c2a8" }}>Use Cases</p>
            <h2 className="text-3xl font-bold" style={{ color: "#e8edf5" }}>
              Built for every high-traffic environment
            </h2>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-px" style={{ backgroundColor: "#1a2436" }}>
            {[
              { label: "Temple & Religious Sites", sub: "Pilgrimage safety", icon: <Shield size={16} /> },
              { label: "Public Transit",           sub: "Platform density",  icon: <Radio size={16} /> },
              { label: "Retail & Malls",           sub: "Footfall analytics",icon: <TrendingUp size={16} /> },
              { label: "Hospitals & Clinics",      sub: "Emergency zones",   icon: <Activity size={16} /> },
              { label: "Schools & Universities",   sub: "Campus occupancy",  icon: <Users size={16} /> },
              { label: "Stadiums & Events",        sub: "Gate crowd flow",   icon: <BarChart2 size={16} /> },
              { label: "Workplaces",               sub: "Space utilization", icon: <Camera size={16} /> },
              { label: "Tourism & Museums",        sub: "Visitor experience",icon: <Globe size={16} /> },
            ].map((v) => (
              <div
                key={v.label}
                className="px-5 py-6 transition-colors duration-150 cursor-default group"
                style={{ backgroundColor: "#0a0e16" }}
                onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "#0e1520")}
                onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "#0a0e16")}
              >
                <div
                  className="w-8 h-8 rounded-lg flex items-center justify-center mb-3 transition-colors"
                  style={{ backgroundColor: "rgba(0,194,168,0.08)", color: "#4a5c78" }}
                >
                  {v.icon}
                </div>
                <p className="text-sm font-medium leading-tight mb-1" style={{ color: "#c8d5e8" }}>{v.label}</p>
                <p className="text-xs" style={{ color: "#3a4e68" }}>{v.sub}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Pricing ── */}
      <section id="pricing" className="px-6 py-28">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-14">
            <p className="text-xs font-semibold uppercase tracking-widest mb-3" style={{ color: "#00c2a8" }}>Pricing</p>
            <h2 className="text-3xl font-bold mb-3" style={{ color: "#e8edf5" }}>
              Transparent plans that scale with you
            </h2>
            <p className="text-sm" style={{ color: "#4a5c78" }}>
              Start free. Upgrade when your deployment grows.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {[
              {
                tier: "Free",
                price: "$0",
                period: "",
                cameras: "3 cameras",
                alerts: "5 alert rules",
                features: ["Basic analytics", "In-app notifications", "30-day retention"],
                cta: "Get started",
                highlight: false,
              },
              {
                tier: "Pro",
                price: "$29",
                period: "/mo",
                cameras: "25 cameras",
                alerts: "50 alert rules",
                features: ["PDF & CSV exports", "Public status page", "Slack / Teams alerts", "90-day retention", "API tokens"],
                cta: "Start free trial",
                highlight: true,
              },
              {
                tier: "Enterprise",
                price: "Custom",
                period: "",
                cameras: "Unlimited cameras",
                alerts: "Unlimited alerts",
                features: ["Dedicated infrastructure", "SLA support", "SSO / SAML", "Custom retention", "On-premise"],
                cta: "Contact sales",
                highlight: false,
              },
            ].map((p) => (
              <div
                key={p.tier}
                className="relative rounded-2xl p-6 flex flex-col"
                style={{
                  backgroundColor: p.highlight ? "rgba(0,194,168,0.04)" : "#0d1420",
                  border: p.highlight ? "1px solid rgba(0,194,168,0.35)" : "1px solid #1a2436",
                  boxShadow: p.highlight ? "0 0 60px rgba(0,194,168,0.08)" : "none",
                }}
              >
                {p.highlight && (
                  <div
                    className="absolute -top-px left-1/2 -translate-x-1/2 px-4 py-0.5 rounded-b-lg text-2xs font-semibold uppercase tracking-wider"
                    style={{ backgroundColor: "#00c2a8", color: "#080c12" }}
                  >
                    Most popular
                  </div>
                )}

                <div className="mb-6 mt-2">
                  <p className="text-xs font-semibold uppercase tracking-widest mb-3" style={{ color: "#4a5c78" }}>
                    {p.tier}
                  </p>
                  <div className="flex items-end gap-1.5 mb-1.5">
                    <span className="font-mono font-bold" style={{ fontSize: "2.5rem", color: "#e8edf5", letterSpacing: "-0.04em", lineHeight: 1 }}>
                      {p.price}
                    </span>
                    {p.period && <span className="text-sm pb-1.5" style={{ color: "#4a5c78" }}>{p.period}</span>}
                  </div>
                  <p className="text-xs" style={{ color: "#3a4e68" }}>{p.cameras} · {p.alerts}</p>
                </div>

                <ul className="space-y-2.5 flex-1 mb-6">
                  {p.features.map((f) => (
                    <li key={f} className="flex items-start gap-2 text-xs" style={{ color: "#8a9bb5" }}>
                      <CheckCircle2 size={12} className="flex-shrink-0 mt-0.5" style={{ color: "#00c2a8" }} />
                      {f}
                    </li>
                  ))}
                </ul>

                <Link
                  to={ROUTES.REGISTER}
                  className="flex items-center justify-center gap-1.5 py-2.5 rounded-xl text-sm font-medium transition-all"
                  style={
                    p.highlight
                      ? { backgroundColor: "#00c2a8", color: "#080c12" }
                      : { backgroundColor: "#111820", color: "#8a9bb5", border: "1px solid #1a2436" }
                  }
                  onMouseEnter={(e) => {
                    if (!p.highlight) { (e.currentTarget as HTMLElement).style.borderColor = "#2d3f58"; (e.currentTarget as HTMLElement).style.color = "#e8edf5"; }
                  }}
                  onMouseLeave={(e) => {
                    if (!p.highlight) { (e.currentTarget as HTMLElement).style.borderColor = "#1a2436"; (e.currentTarget as HTMLElement).style.color = "#8a9bb5"; }
                  }}
                >
                  {p.cta}
                  <ChevronRight size={13} />
                </Link>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Privacy callout ── */}
      <section className="px-6 py-16" style={{ borderTop: "1px solid #1a2436", borderBottom: "1px solid #1a2436", backgroundColor: "#0a0e16" }}>
        <div className="max-w-4xl mx-auto flex flex-col md:flex-row items-center justify-between gap-8">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-2xl flex items-center justify-center flex-shrink-0"
              style={{ backgroundColor: "rgba(0,194,168,0.08)", border: "1px solid rgba(0,194,168,0.15)" }}>
              <Shield size={22} style={{ color: "#00c2a8" }} />
            </div>
            <div>
              <p className="font-semibold mb-1" style={{ color: "#e8edf5" }}>Privacy-first by design</p>
              <p className="text-sm" style={{ color: "#4a5c78" }}>
                No face data stored. No PII collected. GDPR-compliant with configurable retention.
              </p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-2xl flex items-center justify-center flex-shrink-0"
              style={{ backgroundColor: "rgba(0,194,168,0.08)", border: "1px solid rgba(0,194,168,0.15)" }}>
              <Zap size={22} style={{ color: "#00c2a8" }} />
            </div>
            <div>
              <p className="font-semibold mb-1" style={{ color: "#e8edf5" }}>API-first platform</p>
              <p className="text-sm" style={{ color: "#4a5c78" }}>
                Full REST + WebSocket API. Customer-managed tokens with granular scope control.
              </p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-2xl flex items-center justify-center flex-shrink-0"
              style={{ backgroundColor: "rgba(0,194,168,0.08)", border: "1px solid rgba(0,194,168,0.15)" }}>
              <Globe size={22} style={{ color: "#00c2a8" }} />
            </div>
            <div>
              <p className="font-semibold mb-1" style={{ color: "#e8edf5" }}>Public status pages</p>
              <p className="text-sm" style={{ color: "#4a5c78" }}>
                Shareable live crowd feeds for your community. No login required to view.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ── Final CTA ── */}
      <section className="px-6 py-32 text-center relative overflow-hidden">
        <div
          className="absolute inset-0 pointer-events-none"
          style={{ background: "radial-gradient(ellipse 60% 50% at 50% 50%, rgba(0,194,168,0.07) 0%, transparent 70%)" }}
        />
        <div className="relative max-w-2xl mx-auto">
          <h2
            className="font-bold leading-tight tracking-tight mb-5"
            style={{ fontSize: "clamp(2rem, 5vw, 3.5rem)", color: "#e8edf5" }}
          >
            Deploy in minutes.
            <br />
            <span style={{ color: "#4a5c78" }}>No hardware changes required.</span>
          </h2>
          <p className="text-base mb-10" style={{ color: "#4a5c78" }}>
            Works with any IP camera. Free forever on up to 3 cameras.
          </p>
          <div className="flex flex-wrap items-center justify-center gap-3">
            <Link
              to={ROUTES.REGISTER}
              className="flex items-center gap-2 px-8 py-3.5 rounded-xl font-semibold text-sm transition-all"
              style={{ backgroundColor: "#00c2a8", color: "#080c12", boxShadow: "0 0 40px rgba(0,194,168,0.25)" }}
            >
              Create free account
              <ArrowRight size={15} />
            </Link>
            <a
              href="#features"
              className="flex items-center gap-2 px-8 py-3.5 rounded-xl font-medium text-sm transition-all"
              style={{ border: "1px solid #1e2a3d", color: "#8a9bb5" }}
            >
              See how it works
            </a>
          </div>
        </div>
      </section>

      {/* ── Footer ── */}
      <footer className="px-6 py-10" style={{ borderTop: "1px solid #1a2436" }}>
        <div className="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded-md flex items-center justify-center" style={{ backgroundColor: "#00c2a8" }}>
              <Activity size={12} style={{ color: "#080c12" }} />
            </div>
            <span className="text-sm font-semibold" style={{ color: "#3a4e68" }}>PeopleSense</span>
          </div>
          <p className="text-xs" style={{ color: "#2a3a52" }}>
            © {new Date().getFullYear()} PeopleSense. Privacy-first crowd intelligence.
          </p>
          <div className="flex items-center gap-5 text-xs" style={{ color: "#2a3a52" }}>
            {["Privacy Policy", "Terms", "API Docs", "Status"].map((l) => (
              <a key={l} href="#" className="transition-colors hover:text-white">{l}</a>
            ))}
          </div>
        </div>
      </footer>

      {/* Shimmer animation */}
      <style>{`
        @keyframes shimmer {
          0%   { background-position: 0% center; }
          100% { background-position: 200% center; }
        }
      `}</style>
    </div>
  );
}