/**
 * Landing page — dark/light mode aware, Stripe-grade design.
 *
 * Theme: "Cyberpunk/Precision instrument" — deep slate darks, teal accent,
 * monospace data readouts, crisp grid lines. Matches the ops dashboard.
 *
 * Light mode: clean white with same accent — feels like Linear/Vercel.
 * Dark mode:  deep #080c12 with teal glow — feels like Arc/DaVinci.
 *
 * Memorable element: the live-animating dashboard mock in the hero that
 * actually looks like the real app.
 */
import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  Activity, AlertTriangle, ArrowRight, BarChart2,
  Camera, CheckCircle2, ChevronRight, Globe, Moon, Radio,
  Shield, Sun, TrendingUp, Users, Zap,
} from "lucide-react";
import { ROUTES } from "../constants";
import { useThemeMode } from "../hooks";

// ── Intersection observer hook ────────────────────────────────────────────────
function useInView(threshold = 0.12) {
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

// ── Animated counter ──────────────────────────────────────────────────────────
function Counter({ target, suffix = "" }: { target: number; suffix?: string }) {
  const [val, setVal] = useState(0);
  const { ref, visible } = useInView(0.3);
  useEffect(() => {
    if (!visible) return;
    const dur = 1600;
    const start = performance.now();
    const tick = (now: number) => {
      const p = Math.min((now - start) / dur, 1);
      const ease = 1 - Math.pow(1 - p, 4);
      setVal(Math.round(ease * target));
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }, [visible, target]);
  return <span ref={ref}>{val.toLocaleString()}{suffix}</span>;
}

// ── Floating particles background ─────────────────────────────────────────────
function Particles({ dark }: { dark: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    let animId: number;
    const resize = () => {
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
    };
    resize();
    window.addEventListener("resize", resize);

    const particles = Array.from({ length: 55 }, () => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      r: Math.random() * 1.2 + 0.3,
      vx: (Math.random() - 0.5) * 0.3,
      vy: (Math.random() - 0.5) * 0.3,
      alpha: Math.random() * 0.4 + 0.1,
    }));

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const color = dark ? "0,194,168" : "0,194,168";
      for (const p of particles) {
        p.x += p.vx; p.y += p.vy;
        if (p.x < 0) p.x = canvas.width;
        if (p.x > canvas.width) p.x = 0;
        if (p.y < 0) p.y = canvas.height;
        if (p.y > canvas.height) p.y = 0;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${color},${p.alpha * (dark ? 1 : 0.5)})`;
        ctx.fill();
      }
      // Draw connecting lines
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 100) {
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.strokeStyle = `rgba(${color},${(1 - dist / 100) * (dark ? 0.08 : 0.04)})`;
            ctx.stroke();
          }
        }
      }
      animId = requestAnimationFrame(draw);
    };
    draw();
    return () => { cancelAnimationFrame(animId); window.removeEventListener("resize", resize); };
  }, [dark]);
  return <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none" />;
}

// ── Live dashboard mock ───────────────────────────────────────────────────────
function DashboardMock({ dark }: { dark: boolean }) {
  const bg     = dark ? "#0d1420" : "#ffffff";
  const border = dark ? "#1e2a3d" : "#e8eaef";
  const label  = dark ? "#4a5c78" : "#9aa5b4";

  const bars = [42, 68, 55, 80, 61, 93, 74, 58, 82, 67, 91, 76];
  const [active, setActive] = useState(11);
  useEffect(() => {
    const id = setInterval(() => setActive((a) => (a + 1) % bars.length), 900);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="rounded-2xl overflow-hidden select-none"
      style={{ backgroundColor: bg, border: `1px solid ${border}`,
        boxShadow: dark
          ? "0 32px 80px rgba(0,0,0,0.6), 0 0 0 1px rgba(0,194,168,0.06)"
          : "0 32px 80px rgba(0,0,0,0.1), 0 0 0 1px rgba(0,194,168,0.08)" }}>
      {/* Titlebar */}
      <div className="flex items-center gap-2 px-4 py-3 border-b" style={{ borderColor: border }}>
        <span className="w-3 h-3 rounded-full" style={{ backgroundColor: "#ff5f57" }} />
        <span className="w-3 h-3 rounded-full" style={{ backgroundColor: "#febc2e" }} />
        <span className="w-3 h-3 rounded-full" style={{ backgroundColor: "#28c840" }} />
        <span className="ml-3 text-xs font-mono" style={{ color: label }}>peoplesense — live</span>
        <span className="ml-auto flex items-center gap-1.5 text-xs" style={{ color: "#00c2a8" }}>
          <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ backgroundColor: "#00c2a8" }} />Live
        </span>
      </div>
      {/* KPIs */}
      <div className="grid grid-cols-3 gap-px" style={{ backgroundColor: border }}>
        {[
          { label: "Current Count", value: "247", color: "#00c2a8" },
          { label: "Peak Today",    value: "891", color: "#f59e0b" },
          { label: "Cameras Live",  value: "12/14", color: "#34d399" },
        ].map((k) => (
          <div key={k.label} className="px-4 py-3.5" style={{ backgroundColor: bg }}>
            <p className="uppercase tracking-widest mb-1" style={{ color: label, fontSize: "9px" }}>{k.label}</p>
            <p className="text-xl font-mono font-medium" style={{ color: k.color, letterSpacing: "-0.03em" }}>{k.value}</p>
          </div>
        ))}
      </div>
      {/* Chart */}
      <div className="px-4 pt-4 pb-2">
        <p className="text-xs mb-3" style={{ color: label }}>People count — last 12 buckets</p>
        <div className="flex items-end gap-1.5" style={{ height: 72 }}>
          {bars.map((h, i) => (
            <div key={i} className="flex-1 rounded-sm transition-all duration-500" style={{
              height: `${h}%`,
              backgroundColor: i === active ? "#00c2a8" : (dark ? "rgba(0,194,168,0.15)" : "rgba(0,194,168,0.12)"),
              boxShadow: i === active ? "0 0 12px rgba(0,194,168,0.5)" : "none",
            }} />
          ))}
        </div>
      </div>
      {/* Alert row */}
      <div className="mx-4 mb-4 mt-2 rounded-lg px-3 py-2.5 flex items-center gap-2.5"
        style={{ backgroundColor: "rgba(245,158,11,0.08)", border: "1px solid rgba(245,158,11,0.2)" }}>
        <AlertTriangle size={12} style={{ color: "#f59e0b", flexShrink: 0 }} />
        <span className="text-xs" style={{ color: "#f59e0b" }}>Gate A density exceeded threshold · 3 min ago</span>
      </div>
    </div>
  );
}

// ── Alert feed mock ───────────────────────────────────────────────────────────
function AlertFeedMock({ dark }: { dark: boolean }) {
  const bg     = dark ? "#0d1420" : "#ffffff";
  const border = dark ? "#1e2a3d" : "#e8eaef";
  const rowDiv = dark ? "#0f1825" : "#f5f6f8";
  const nameC  = dark ? "#c8d5e8" : "#1a2233";
  const timeC  = dark ? "#4a5c78" : "#9aa5b4";
  const headC  = dark ? "#e8edf5" : "#0d1117";
  const items = [
    { time: "14:32", cam: "Main Entrance", count: 312, type: "critical" },
    { time: "14:18", cam: "Platform 3",    count: 198, type: "warn" },
    { time: "14:05", cam: "Food Court",    count: 87,  type: "ok" },
    { time: "13:51", cam: "Gate B",        count: 243, type: "warn" },
  ];
  const colors = { critical: "#f87171", warn: "#f59e0b", ok: "#34d399" };
  return (
    <div className="rounded-2xl overflow-hidden"
      style={{ backgroundColor: bg, border: `1px solid ${border}`,
        boxShadow: dark ? "0 32px 80px rgba(0,0,0,0.5)" : "0 32px 80px rgba(0,0,0,0.08)" }}>
      <div className="px-4 py-3 border-b flex items-center justify-between" style={{ borderColor: border }}>
        <span className="text-xs font-semibold" style={{ color: headC }}>Alert Feed</span>
        <span className="text-2xs px-2 py-0.5 rounded-md font-medium"
          style={{ backgroundColor: "rgba(248,113,113,0.12)", color: "#f87171", border: "1px solid rgba(248,113,113,0.2)" }}>
          2 active
        </span>
      </div>
      {items.map((item, i) => (
        <div key={i} className="flex items-center gap-3 px-4 py-3 border-b" style={{ borderColor: rowDiv }}>
          <span className="w-1.5 h-1.5 rounded-full flex-shrink-0"
            style={{ backgroundColor: colors[item.type as keyof typeof colors] }} />
          <div className="flex-1 min-w-0">
            <p className="text-xs font-medium" style={{ color: nameC }}>{item.cam}</p>
            <p className="text-2xs font-mono" style={{ color: timeC }}>{item.time}</p>
          </div>
          <span className="text-sm font-mono font-medium" style={{ color: colors[item.type as keyof typeof colors] }}>
            {item.count}
          </span>
        </div>
      ))}
    </div>
  );
}

// ── Feature row ───────────────────────────────────────────────────────────────
function FeatureRow({ eyebrow, title, description, bullets, visual, reverse = false, dark }:
  { eyebrow: string; title: string; description: string; bullets: string[]; visual: React.ReactNode; reverse?: boolean; dark: boolean }) {
  const { ref, visible } = useInView();
  const textColor  = dark ? "#e8edf5" : "#0d1117";
  const subColor   = dark ? "#8a9bb5" : "#4a5568";
  return (
    <div ref={ref}
      className={`grid grid-cols-1 lg:grid-cols-2 gap-16 items-center transition-all duration-700 ${
        visible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-10"
      } ${reverse ? "lg:grid-flow-dense" : ""}`}>
      <div className={reverse ? "lg:col-start-2" : ""}>
        <p className="text-xs font-semibold uppercase tracking-widest mb-3" style={{ color: "#00c2a8" }}>{eyebrow}</p>
        <h3 className="text-3xl font-bold leading-tight mb-4" style={{ color: textColor }}>{title}</h3>
        <p className="text-base leading-relaxed mb-6" style={{ color: subColor }}>{description}</p>
        <ul className="space-y-2.5">
          {bullets.map((b) => (
            <li key={b} className="flex items-start gap-2.5 text-sm" style={{ color: subColor }}>
              <CheckCircle2 size={15} className="flex-shrink-0 mt-0.5" style={{ color: "#00c2a8" }} />{b}
            </li>
          ))}
        </ul>
      </div>
      <div className={reverse ? "lg:col-start-1 lg:row-start-1" : ""}>{visual}</div>
    </div>
  );
}

// ── Main ──────────────────────────────────────────────────────────────────────
export default function Landing() {
  const [themeMode, toggleTheme] = useThemeMode();
  const dark = themeMode === "ops";
  const heroRef = useRef<HTMLHeadingElement>(null);
  const [scrolled, setScrolled] = useState(false);

  // Parallax + scrolled nav state
  useEffect(() => {
    const handler = () => {
      setScrolled(window.scrollY > 40);
      if (heroRef.current) {
        heroRef.current.style.transform = `translateY(${window.scrollY * 0.06}px)`;
      }
    };
    window.addEventListener("scroll", handler, { passive: true });
    return () => window.removeEventListener("scroll", handler);
  }, []);

  // Colors that switch between modes
  const bg        = dark ? "#080c12"           : "#f9fafb";
  const bgSurf    = dark ? "#0d1420"           : "#ffffff";
  const bgAlt     = dark ? "#0a0e16"           : "#f3f4f6";
  const border    = dark ? "#1a2436"           : "#e5e7eb";
  const textH     = dark ? "#e8edf5"           : "#0d1117";
  const textSub   = dark ? "#8a9bb5"           : "#4a5568";
  const textMuted = dark ? "#4a5c78"           : "#9aa5b4";
  const navBg     = dark
    ? `rgba(8,12,18,${scrolled ? "0.95" : "0.7"})`
    : `rgba(249,250,251,${scrolled ? "0.97" : "0.8"})`;
  const glowClr   = dark ? "rgba(0,194,168,0.10)" : "rgba(0,194,168,0.06)";

  return (
    <div className="min-h-screen overflow-x-hidden" style={{ backgroundColor: bg, color: textH, fontFamily: "'DM Sans', sans-serif", transition: "background-color 0.3s, color 0.3s" }}>

      {/* ── Navbar ────────────────────────────────────────────────────────── */}
      <header className="fixed top-0 left-0 right-0 z-50 px-6 py-0"
        style={{ borderBottom: `1px solid ${scrolled ? border : "transparent"}`, backdropFilter: "blur(20px)", backgroundColor: navBg, transition: "all 0.3s" }}>
        <div className="max-w-6xl mx-auto flex items-center justify-between h-14">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg flex items-center justify-center" style={{ backgroundColor: "#00c2a8" }}>
              <Activity size={14} style={{ color: "#080c12" }} />
            </div>
            <span className="font-semibold text-sm tracking-tight" style={{ color: textH }}>PeopleSense</span>
          </div>

          <nav className="hidden md:flex items-center gap-1">
            {["Features", "Use Cases", "Pricing"].map((item) => (
              <a key={item} href={`#${item.toLowerCase().replace(" ", "-")}`}
                className="px-3 py-1.5 text-sm rounded-lg transition-colors"
                style={{ color: textMuted }}
                onMouseEnter={(e) => (e.currentTarget.style.color = textH)}
                onMouseLeave={(e) => (e.currentTarget.style.color = textMuted)}>
                {item}
              </a>
            ))}
          </nav>

          <div className="flex items-center gap-2">
            {/* Theme toggle */}
            <button onClick={toggleTheme}
              className="w-8 h-8 rounded-lg flex items-center justify-center transition-colors"
              style={{ backgroundColor: "transparent", border: `1px solid ${border}`, color: textMuted }}
              onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.color = textH; (e.currentTarget as HTMLElement).style.borderColor = textMuted; }}
              onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.color = textMuted; (e.currentTarget as HTMLElement).style.borderColor = border; }}
              aria-label="Toggle theme">
              {dark ? <Sun size={14} /> : <Moon size={14} />}
            </button>
            <Link to={ROUTES.LOGIN} className="px-3 py-1.5 text-sm rounded-lg transition-colors" style={{ color: textMuted }}
              onMouseEnter={(e) => (e.currentTarget.style.color = textH)}
              onMouseLeave={(e) => (e.currentTarget.style.color = textMuted)}>
              Sign in
            </Link>
            <Link to={ROUTES.REGISTER}
              className="flex items-center gap-1.5 px-4 py-1.5 text-sm font-medium rounded-lg"
              style={{ backgroundColor: "#00c2a8", color: "#080c12" }}>
              Get started <ArrowRight size={13} />
            </Link>
          </div>
        </div>
      </header>

      {/* ── Hero ──────────────────────────────────────────────────────────── */}
      <section className="relative flex flex-col items-center justify-center text-center min-h-screen px-6 pb-20 pt-24 overflow-hidden">
        <Particles dark={dark} />
        <div className="absolute inset-0 pointer-events-none"
          style={{ background: `radial-gradient(ellipse 80% 60% at 50% -10%, ${glowClr} 0%, transparent 70%)` }} />
        {/* Subtle grid */}
        <div className="absolute inset-0 pointer-events-none"
          style={{ backgroundImage: `linear-gradient(${dark ? "rgba(0,194,168,0.03)" : "rgba(0,194,168,0.06)"} 1px, transparent 1px), linear-gradient(90deg, ${dark ? "rgba(0,194,168,0.03)" : "rgba(0,194,168,0.06)"} 1px, transparent 1px)`, backgroundSize: "60px 60px" }} />

        <div className="relative max-w-5xl mx-auto">
          {/* Live badge */}
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium mb-8"
            style={{ backgroundColor: dark ? "rgba(0,194,168,0.08)" : "rgba(0,194,168,0.1)", border: "1px solid rgba(0,194,168,0.25)", color: "#00c2a8" }}>
            <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ backgroundColor: "#00c2a8" }} />
            Now monitoring 40M+ crowd events per month
          </div>

          {/* Headline */}
          <h1 ref={heroRef} className="font-bold leading-[1.05] tracking-tight mb-6"
            style={{ fontSize: "clamp(2.8rem, 7vw, 5.5rem)", color: textH }}>
            Crowd intelligence
            <br />
            <span style={{ background: "linear-gradient(90deg, #00c2a8 0%, #4af0d8 50%, #00c2a8 100%)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundSize: "200% auto", animation: "shimmer 3s linear infinite" }}>
              for safer spaces.
            </span>
          </h1>

          <p className="text-lg leading-relaxed mb-10 mx-auto" style={{ color: textSub, maxWidth: "580px" }}>
            Real-time people counting, density alerts, and predictive analytics for temples,
            transit hubs, hospitals, and large venues. Privacy-first. No face data. Ever.
          </p>

          {/* CTAs */}
          <div className="flex flex-wrap items-center justify-center gap-3 mb-20">
            <Link to={ROUTES.REGISTER}
              className="flex items-center gap-2 px-6 py-3 rounded-xl font-semibold text-sm transition-all duration-200"
              style={{ backgroundColor: "#00c2a8", color: "#080c12", boxShadow: "0 0 32px rgba(0,194,168,0.3)" }}
              onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.boxShadow = "0 0 48px rgba(0,194,168,0.5)"; (e.currentTarget as HTMLElement).style.transform = "translateY(-1px)"; }}
              onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.boxShadow = "0 0 32px rgba(0,194,168,0.3)"; (e.currentTarget as HTMLElement).style.transform = "translateY(0)"; }}>
              Start for free <ArrowRight size={15} />
            </Link>
            <Link to={ROUTES.LOGIN}
              className="flex items-center gap-2 px-6 py-3 rounded-xl font-medium text-sm transition-all duration-200"
              style={{ border: `1px solid ${border}`, color: textSub, backgroundColor: dark ? "rgba(255,255,255,0.02)" : "rgba(0,0,0,0.02)" }}
              onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.borderColor = textMuted; (e.currentTarget as HTMLElement).style.color = textH; }}
              onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.borderColor = border; (e.currentTarget as HTMLElement).style.color = textSub; }}>
              View live demo
            </Link>
          </div>

          {/* Hero visual */}
          <div className="relative mx-auto" style={{ maxWidth: "780px" }}>
            <div className="absolute inset-0 rounded-2xl blur-3xl"
              style={{ background: "radial-gradient(ellipse at 50% 80%, rgba(0,194,168,0.12), transparent 70%)", transform: "scaleY(0.6) translateY(30%)" }} />
            <DashboardMock dark={dark} />
          </div>
        </div>
      </section>

      {/* ── Stats ─────────────────────────────────────────────────────────── */}
      <section className="px-6 py-20" style={{ borderTop: `1px solid ${border}`, borderBottom: `1px solid ${border}`, backgroundColor: bgAlt }}>
        <div className="max-w-4xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-8 text-center">
          {[
            { target: 40,   suffix: "M+", label: "Events / month" },
            { target: 500,  suffix: "ms", label: "Detection latency" },
            { target: 8,    suffix: "+",  label: "Industry verticals" },
            { target: 99.9, suffix: "%",  label: "Uptime SLA" },
          ].map((s) => (
            <div key={s.label}>
              <p className="font-mono font-bold mb-1" style={{ fontSize: "clamp(2rem,4vw,3rem)", color: textH, letterSpacing: "-0.04em" }}>
                <Counter target={s.target} suffix={s.suffix} />
              </p>
              <p className="text-xs" style={{ color: textMuted }}>{s.label}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── Features ──────────────────────────────────────────────────────── */}
      <section id="features" className="px-6 py-28 max-w-6xl mx-auto space-y-32">
        <FeatureRow dark={dark} eyebrow="Real-time Monitoring"
          title="Know exactly who is in your space, right now."
          description="Sub-second head counts from any IP camera. Live density overlays, multi-camera grid, and instant status across your entire facility — all in one dashboard."
          bullets={["Works with any RTSP / HTTP camera stream", "Sub-500ms detection latency via WebSocket", "Density heat maps per zone", "Mobile-ready for field operators"]}
          visual={<DashboardMock dark={dark} />}
        />
        <FeatureRow dark={dark} eyebrow="Smart Alerts" reverse
          title="Get notified before a situation becomes critical."
          description="Configurable threshold and anomaly alerts delivered to Slack, MS Teams, email, or custom webhooks — in under five seconds of the event firing."
          bullets={["Threshold and z-score anomaly triggers", "Slack, Teams, email, and webhook delivery", "Per-camera cooldown and escalation rules", "Alert history with full audit trail"]}
          visual={<AlertFeedMock dark={dark} />}
        />
        <FeatureRow dark={dark} eyebrow="Predictive Analytics"
          title="Forecast surges before they happen."
          description="Hourly trend charts, 7-day rolling averages, and anomaly detection powered by z-score analysis. Export PDF or CSV reports for operations and compliance teams."
          bullets={["7, 14, 30, and 90-day trend windows", "Peak forecasting with confidence bands", "Per-camera breakdown in PDF reports", "CSV export for custom BI pipelines"]}
          visual={
            <div className="rounded-2xl p-6 overflow-hidden"
              style={{ backgroundColor: bgSurf, border: `1px solid ${border}`, boxShadow: dark ? "0 32px 80px rgba(0,0,0,0.5)" : "0 32px 80px rgba(0,0,0,0.08)" }}>
              <p className="text-xs mb-4" style={{ color: textMuted }}>People count — last 30 days</p>
              <svg viewBox="0 0 400 120" className="w-full">
                <defs>
                  <linearGradient id="chartGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#00c2a8" stopOpacity="0.25" />
                    <stop offset="100%" stopColor="#00c2a8" stopOpacity="0" />
                  </linearGradient>
                </defs>
                {[0, 30, 60, 90].map((y) => (
                  <line key={y} x1="0" y1={y} x2="400" y2={y} stroke={border} strokeWidth="1" />
                ))}
                <path d="M0,90 C20,85 30,70 50,65 S80,40 100,50 S130,30 160,35 S200,20 220,25 S260,15 280,20 S320,35 350,30 S380,20 400,18 L400,120 L0,120 Z" fill="url(#chartGrad)" />
                <path d="M0,90 C20,85 30,70 50,65 S80,40 100,50 S130,30 160,35 S200,20 220,25 S260,15 280,20 S320,35 350,30 S380,20 400,18" fill="none" stroke="#00c2a8" strokeWidth="2" />
                <circle cx="280" cy="20" r="4" fill="#00c2a8" />
                <circle cx="280" cy="20" r="8" fill="rgba(0,194,168,0.2)" />
                <text x="292" y="16" fill="#00c2a8" fontSize="9" fontFamily="DM Mono">peak 891</text>
              </svg>
              <div className="grid grid-cols-3 gap-3 mt-4">
                {[{ label: "Avg daily peak", value: "612" }, { label: "Total detections", value: "284K" }, { label: "Anomalies flagged", value: "7" }].map((s) => (
                  <div key={s.label} className="rounded-xl px-3 py-2.5" style={{ backgroundColor: bgAlt }}>
                    <p className="text-2xs mb-1" style={{ color: textMuted, fontSize: "9px" }}>{s.label}</p>
                    <p className="text-base font-mono font-medium" style={{ color: textH }}>{s.value}</p>
                  </div>
                ))}
              </div>
            </div>
          }
        />
      </section>

      {/* ── Use cases ─────────────────────────────────────────────────────── */}
      <section id="use-cases" className="px-6 py-24" style={{ backgroundColor: bgAlt, borderTop: `1px solid ${border}` }}>
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-14">
            <p className="text-xs font-semibold uppercase tracking-widest mb-3" style={{ color: "#00c2a8" }}>Use Cases</p>
            <h2 className="text-3xl font-bold" style={{ color: textH }}>Built for every high-traffic environment</h2>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-px" style={{ backgroundColor: border }}>
            {[
              { label: "Temple & Religious Sites", sub: "Pilgrimage safety",   icon: <Shield size={16} /> },
              { label: "Public Transit",           sub: "Platform density",    icon: <Radio size={16} /> },
              { label: "Retail & Malls",           sub: "Footfall analytics",  icon: <TrendingUp size={16} /> },
              { label: "Hospitals & Clinics",      sub: "Emergency zones",     icon: <Activity size={16} /> },
              { label: "Schools & Universities",   sub: "Campus occupancy",    icon: <Users size={16} /> },
              { label: "Stadiums & Events",        sub: "Gate crowd flow",     icon: <BarChart2 size={16} /> },
              { label: "Workplaces",               sub: "Space utilization",   icon: <Camera size={16} /> },
              { label: "Tourism & Museums",        sub: "Visitor experience",  icon: <Globe size={16} /> },
            ].map((v) => (
              <div key={v.label} className="px-5 py-6 transition-colors duration-150 cursor-default"
                style={{ backgroundColor: bgSurf }}
                onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = bgAlt)}
                onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = bgSurf)}>
                <div className="w-8 h-8 rounded-lg flex items-center justify-center mb-3"
                  style={{ backgroundColor: "rgba(0,194,168,0.08)", color: textMuted }}>
                  {v.icon}
                </div>
                <p className="text-sm font-medium leading-tight mb-1" style={{ color: textH }}>{v.label}</p>
                <p className="text-xs" style={{ color: textMuted }}>{v.sub}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Pricing ───────────────────────────────────────────────────────── */}
      <section id="pricing" className="px-6 py-28">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-14">
            <p className="text-xs font-semibold uppercase tracking-widest mb-3" style={{ color: "#00c2a8" }}>Pricing</p>
            <h2 className="text-3xl font-bold mb-3" style={{ color: textH }}>Transparent plans that scale with you</h2>
            <p className="text-sm" style={{ color: textMuted }}>Start free. Upgrade when your deployment grows.</p>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {[
              { tier: "Free",       price: "$0",     period: "",    cameras: "3 cameras",         alerts: "5 alert rules",        features: ["Basic analytics", "In-app notifications", "30-day retention"], cta: "Get started",      highlight: false },
              { tier: "Pro",        price: "$29",    period: "/mo", cameras: "25 cameras",        alerts: "50 alert rules",       features: ["PDF & CSV exports", "Public status page", "Slack / Teams", "90-day retention", "API tokens"], cta: "Start free trial", highlight: true },
              { tier: "Enterprise", price: "Custom", period: "",    cameras: "Unlimited cameras", alerts: "Unlimited alerts",     features: ["Dedicated infra", "SLA support", "SSO / SAML", "Custom retention", "On-premise"], cta: "Contact sales",   highlight: false },
            ].map((p) => (
              <div key={p.tier} className="relative rounded-2xl p-6 flex flex-col transition-transform duration-200"
                style={{ backgroundColor: p.highlight ? (dark ? "rgba(0,194,168,0.04)" : "rgba(0,194,168,0.03)") : bgSurf,
                  border: p.highlight ? "1.5px solid rgba(0,194,168,0.4)" : `1px solid ${border}`,
                  boxShadow: p.highlight ? "0 0 60px rgba(0,194,168,0.08)" : "none" }}
                onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.transform = "translateY(-3px)"; }}
                onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.transform = "translateY(0)"; }}>
                {p.highlight && (
                  <div className="absolute -top-px left-1/2 -translate-x-1/2 px-4 py-0.5 rounded-b-lg text-2xs font-semibold uppercase tracking-wider"
                    style={{ backgroundColor: "#00c2a8", color: "#080c12" }}>Most popular</div>
                )}
                <div className="mb-6 mt-2">
                  <p className="text-xs font-semibold uppercase tracking-widest mb-3" style={{ color: textMuted }}>{p.tier}</p>
                  <div className="flex items-end gap-1.5 mb-1.5">
                    <span className="font-mono font-bold" style={{ fontSize: "2.5rem", color: textH, letterSpacing: "-0.04em", lineHeight: 1 }}>{p.price}</span>
                    {p.period && <span className="text-sm pb-1.5" style={{ color: textMuted }}>{p.period}</span>}
                  </div>
                  <p className="text-xs" style={{ color: textMuted }}>{p.cameras} · {p.alerts}</p>
                </div>
                <ul className="space-y-2.5 flex-1 mb-6">
                  {p.features.map((f) => (
                    <li key={f} className="flex items-start gap-2 text-xs" style={{ color: textSub }}>
                      <CheckCircle2 size={12} className="flex-shrink-0 mt-0.5" style={{ color: "#00c2a8" }} />{f}
                    </li>
                  ))}
                </ul>
                <Link to={ROUTES.REGISTER}
                  className="flex items-center justify-center gap-1.5 py-2.5 rounded-xl text-sm font-medium transition-all"
                  style={p.highlight
                    ? { backgroundColor: "#00c2a8", color: "#080c12" }
                    : { backgroundColor: dark ? "#111820" : "#f5f6f8", color: textSub, border: `1px solid ${border}` }}
                  onMouseEnter={(e) => { if (!p.highlight) { (e.currentTarget as HTMLElement).style.color = textH; (e.currentTarget as HTMLElement).style.borderColor = textMuted; } }}
                  onMouseLeave={(e) => { if (!p.highlight) { (e.currentTarget as HTMLElement).style.color = textSub; (e.currentTarget as HTMLElement).style.borderColor = border; } }}>
                  {p.cta} <ChevronRight size={13} />
                </Link>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Trust strip ───────────────────────────────────────────────────── */}
      <section className="px-6 py-16" style={{ borderTop: `1px solid ${border}`, borderBottom: `1px solid ${border}`, backgroundColor: bgAlt }}>
        <div className="max-w-4xl mx-auto flex flex-col md:flex-row items-center justify-between gap-8">
          {[
            { icon: <Shield size={22} />, title: "Privacy-first by design", sub: "No face data stored. No PII collected. GDPR-compliant." },
            { icon: <Zap size={22} />,    title: "API-first platform",      sub: "Full REST + WebSocket API. Customer-managed tokens." },
            { icon: <Globe size={22} />,  title: "Public status pages",     sub: "Shareable live crowd feeds. No login required to view." },
          ].map((item) => (
            <div key={item.title} className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-2xl flex items-center justify-center flex-shrink-0"
                style={{ backgroundColor: "rgba(0,194,168,0.08)", border: "1px solid rgba(0,194,168,0.15)" }}>
                <span style={{ color: "#00c2a8" }}>{item.icon}</span>
              </div>
              <div>
                <p className="font-semibold mb-1" style={{ color: textH }}>{item.title}</p>
                <p className="text-sm" style={{ color: textMuted }}>{item.sub}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ── Final CTA ─────────────────────────────────────────────────────── */}
      <section className="px-6 py-32 text-center relative overflow-hidden">
        <div className="absolute inset-0 pointer-events-none"
          style={{ background: `radial-gradient(ellipse 60% 50% at 50% 50%, ${glowClr} 0%, transparent 70%)` }} />
        <div className="relative max-w-2xl mx-auto">
          <h2 className="font-bold leading-tight tracking-tight mb-5"
            style={{ fontSize: "clamp(2rem, 5vw, 3.5rem)", color: textH }}>
            Deploy in minutes.<br />
            <span style={{ color: textMuted }}>No hardware changes required.</span>
          </h2>
          <p className="text-base mb-10" style={{ color: textMuted }}>
            Works with any IP camera. Free forever on up to 3 cameras.
          </p>
          <div className="flex flex-wrap items-center justify-center gap-3">
            <Link to={ROUTES.REGISTER}
              className="flex items-center gap-2 px-8 py-3.5 rounded-xl font-semibold text-sm transition-all"
              style={{ backgroundColor: "#00c2a8", color: "#080c12", boxShadow: "0 0 40px rgba(0,194,168,0.25)" }}
              onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.transform = "translateY(-2px)"; (e.currentTarget as HTMLElement).style.boxShadow = "0 0 56px rgba(0,194,168,0.4)"; }}
              onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.transform = "translateY(0)"; (e.currentTarget as HTMLElement).style.boxShadow = "0 0 40px rgba(0,194,168,0.25)"; }}>
              Create free account <ArrowRight size={15} />
            </Link>
            <a href="#features"
              className="flex items-center gap-2 px-8 py-3.5 rounded-xl font-medium text-sm transition-all"
              style={{ border: `1px solid ${border}`, color: textSub }}
              onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.color = textH; (e.currentTarget as HTMLElement).style.borderColor = textMuted; }}
              onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.color = textSub; (e.currentTarget as HTMLElement).style.borderColor = border; }}>
              See how it works
            </a>
          </div>
        </div>
      </section>

      {/* ── Footer ────────────────────────────────────────────────────────── */}
      <footer className="px-6 py-10" style={{ borderTop: `1px solid ${border}` }}>
        <div className="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded-md flex items-center justify-center" style={{ backgroundColor: "#00c2a8" }}>
              <Activity size={12} style={{ color: "#080c12" }} />
            </div>
            <span className="text-sm font-semibold" style={{ color: textMuted }}>PeopleSense</span>
          </div>
          <p className="text-xs" style={{ color: textMuted }}>
            © {new Date().getFullYear()} PeopleSense. Privacy-first crowd intelligence.
          </p>
          <div className="flex items-center gap-5 text-xs" style={{ color: textMuted }}>
            {["Privacy Policy", "Terms", "API Docs", "Status"].map((l) => (
              <a key={l} href="#" className="transition-colors"
                onMouseEnter={(e) => (e.currentTarget.style.color = textH)}
                onMouseLeave={(e) => (e.currentTarget.style.color = textMuted)}>{l}</a>
            ))}
          </div>
        </div>
      </footer>

      <style>{`
        @keyframes shimmer { 0% { background-position: 0% center; } 100% { background-position: 200% center; } }
        @media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
      `}</style>
    </div>
  );
}