/** @type {import('tailwindcss').Config} */
export default {
  // Use "ops" class for the ops/dark mode instead of "dark"
  darkMode: ["class", '[class="ops"]'],
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        border: "var(--border-base)",
        "border-subtle": "var(--border-subtle)",
        base:    "var(--bg-base)",
        surface: "var(--bg-surface)",
        elevated:"var(--bg-elevated)",
        subtle:  "var(--bg-subtle)",
        muted:   "var(--bg-muted)",
        primary:   "var(--text-primary)",
        secondary: "var(--text-secondary)",
        "text-muted": "var(--text-muted)",
        accent:  "var(--accent)",
        "accent-dim": "var(--accent-dim)",
        warn:    "var(--warn)",
        danger:  "var(--danger)",
        success: "var(--success)",
      },
      fontFamily: {
        sans: ["DM Sans", "-apple-system", "BlinkMacSystemFont", "sans-serif"],
        mono: ["DM Mono", "SF Mono", "Fira Code", "monospace"],
      },
      fontSize: {
        "2xs": ["0.65rem", { lineHeight: "1rem" }],
      },
      boxShadow: {
        card: "0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04)",
        "card-lg": "0 4px 16px rgba(0,0,0,0.08)",
        accent: "0 0 0 3px rgba(0, 194, 168, 0.2)",
      },
      animation: {
        "fade-in": "fadeIn 0.2s ease",
        "slide-up": "slideUp 0.2s ease",
      },
      keyframes: {
        fadeIn:  { from: { opacity: 0 }, to: { opacity: 1 } },
        slideUp: { from: { opacity: 0, transform: "translateY(6px)" }, to: { opacity: 1, transform: "translateY(0)" } },
      },
    },
  },
  plugins: [],
};