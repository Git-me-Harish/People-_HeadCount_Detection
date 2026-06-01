/**
 * PeopleSense Design System
 * Precision-instrument aesthetic — built for real-time operations.
 */

import React from "react";
import { Loader2, X } from "lucide-react";

// ── Button ────────────────────────────────────────────────────────────────────

type ButtonVariant = "primary" | "secondary" | "danger" | "ghost" | "outline";
type ButtonSize = "sm" | "md" | "lg";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  loading?: boolean;
  icon?: React.ReactNode;
}

const variantClasses: Record<ButtonVariant, string> = {
  primary:
    "bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-[var(--text-inverse)] font-semibold shadow-sm disabled:opacity-40",
  secondary:
    "bg-[var(--bg-subtle)] border border-[var(--border-base)] text-[var(--text-primary)] hover:bg-[var(--bg-muted)] disabled:opacity-40",
  danger:
    "bg-[var(--danger)] hover:opacity-90 text-white font-semibold shadow-sm disabled:opacity-40",
  ghost:
    "text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)] hover:text-[var(--text-primary)] disabled:opacity-40",
  outline:
    "border border-[var(--accent)] text-[var(--accent)] hover:bg-[var(--accent-dim)] disabled:opacity-40",
};

const sizeClasses: Record<ButtonSize, string> = {
  sm: "px-3 py-1.5 text-xs rounded-md gap-1.5",
  md: "px-4 py-2 text-sm rounded-lg gap-2",
  lg: "px-5 py-2.5 text-sm rounded-lg gap-2",
};

export function Button({
  variant = "primary",
  size = "md",
  loading = false,
  icon,
  children,
  disabled,
  className = "",
  ...rest
}: ButtonProps) {
  return (
    <button
      disabled={disabled || loading}
      className={`inline-flex items-center justify-center font-medium transition-all duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)] focus-visible:ring-offset-1 focus-visible:ring-offset-[var(--bg-surface)] disabled:cursor-not-allowed ${variantClasses[variant]} ${sizeClasses[size]} ${className}`}
      {...rest}
    >
      {loading ? <Loader2 size={13} className="animate-spin" /> : icon}
      {children}
    </button>
  );
}

// ── Card ──────────────────────────────────────────────────────────────────────

interface CardProps {
  children: React.ReactNode;
  className?: string;
  padding?: boolean;
  hover?: boolean;
}

export function Card({ children, className = "", padding = true, hover = false }: CardProps) {
  return (
    <div
      className={`bg-[var(--bg-surface)] border border-[var(--border-base)] rounded-xl shadow-card ${
        padding ? "p-5" : ""
      } ${hover ? "hover:border-[var(--accent)] transition-colors duration-150 cursor-pointer" : ""} ${className}`}
    >
      {children}
    </div>
  );
}

// ── Section Header (inside cards) ─────────────────────────────────────────────

export function SectionHeader({
  title,
  action,
}: {
  title: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="flex items-center justify-between px-5 py-3.5 border-b border-[var(--border-subtle)]">
      <span className="text-xs font-semibold uppercase tracking-widest text-[var(--text-muted)]">
        {title}
      </span>
      {action}
    </div>
  );
}

// ── Badge ─────────────────────────────────────────────────────────────────────

type BadgeColor = "accent" | "green" | "yellow" | "red" | "gray" | "blue";

const badgeColors: Record<BadgeColor, string> = {
  accent: "bg-[var(--accent-dim)] text-[var(--accent)] border border-[var(--accent)]/20",
  green:  "bg-[var(--success-dim)] text-[var(--success)] border border-[var(--success)]/20",
  yellow: "bg-[var(--warn-dim)] text-[var(--warn)] border border-[var(--warn)]/20",
  red:    "bg-[var(--danger-dim)] text-[var(--danger)] border border-[var(--danger)]/20",
  gray:   "bg-[var(--bg-subtle)] text-[var(--text-muted)] border border-[var(--border-base)]",
  blue:   "bg-blue-500/10 text-blue-400 border border-blue-500/20",
};

// Map old colors to new
const colorMap: Record<string, BadgeColor> = {
  indigo: "accent",
  green: "green",
  yellow: "yellow",
  red: "red",
  gray: "gray",
  purple: "blue",
  accent: "accent",
  blue: "blue",
};

export function Badge({
  children,
  color = "gray",
}: {
  children: React.ReactNode;
  color?: string;
}) {
  const mapped = colorMap[color] ?? "gray";
  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded-md text-2xs font-medium tracking-wide ${badgeColors[mapped as BadgeColor]}`}
    >
      {children}
    </span>
  );
}

// ── Input ─────────────────────────────────────────────────────────────────────

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  helper?: string;
}

export function Input({ label, error, helper, className = "", id, ...rest }: InputProps) {
  const inputId = id ?? label?.toLowerCase().replace(/\s+/g, "-");
  return (
    <div className="space-y-1.5">
      {label && (
        <label
          htmlFor={inputId}
          className="block text-xs font-medium text-[var(--text-secondary)] uppercase tracking-wider"
        >
          {label}
        </label>
      )}
      <input
        id={inputId}
        className={`block w-full rounded-lg border text-sm px-3 py-2.5 bg-[var(--bg-subtle)] text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:outline-none focus:ring-2 focus:ring-[var(--accent)] focus:ring-offset-0 focus:border-[var(--accent)] transition-all font-sans ${
          error
            ? "border-[var(--danger)] focus:ring-[var(--danger)]"
            : "border-[var(--border-base)]"
        } ${className}`}
        {...rest}
      />
      {error && <p className="text-xs text-[var(--danger)]">{error}</p>}
      {helper && !error && <p className="text-xs text-[var(--text-muted)]">{helper}</p>}
    </div>
  );
}

// ── Select ────────────────────────────────────────────────────────────────────

interface SelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
  label?: string;
  options: Array<{ value: string; label: string }>;
}

export function Select({ label, options, className = "", ...rest }: SelectProps) {
  return (
    <div className="space-y-1.5">
      {label && (
        <label className="block text-xs font-medium text-[var(--text-secondary)] uppercase tracking-wider">
          {label}
        </label>
      )}
      <select
        className={`block w-full rounded-lg border border-[var(--border-base)] text-sm px-3 py-2.5 bg-[var(--bg-subtle)] text-[var(--text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--accent)] focus:border-[var(--accent)] transition-all ${className}`}
        {...rest}
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </div>
  );
}

// ── Toggle ────────────────────────────────────────────────────────────────────

interface ToggleProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label?: string;
  disabled?: boolean;
}

export function Toggle({ checked, onChange, label, disabled }: ToggleProps) {
  return (
    <label className="flex items-center gap-3 cursor-pointer group">
      <button
        role="switch"
        aria-checked={checked}
        disabled={disabled}
        onClick={() => onChange(!checked)}
        className={`relative w-9 h-5 rounded-full transition-colors duration-200 focus-visible:ring-2 focus-visible:ring-[var(--accent)] ${
          checked ? "bg-[var(--accent)]" : "bg-[var(--bg-muted)] border border-[var(--border-base)]"
        } disabled:opacity-40 disabled:cursor-not-allowed`}
      >
        <span
          className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow-sm transition-transform duration-200 ${
            checked ? "translate-x-4" : "translate-x-0"
          }`}
        />
      </button>
      {label && (
        <span className="text-sm text-[var(--text-secondary)]">{label}</span>
      )}
    </label>
  );
}

// ── Spinner ───────────────────────────────────────────────────────────────────

export function Spinner({ size = 18 }: { size?: number }) {
  return <Loader2 size={size} className="animate-spin text-[var(--accent)]" />;
}

// ── Empty state ───────────────────────────────────────────────────────────────

export function EmptyState({
  icon,
  title,
  description,
  action,
}: {
  icon?: React.ReactNode;
  title: string;
  description?: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-center">
      {icon && (
        <div className="mb-4 text-[var(--border-base)] w-10 h-10 flex items-center justify-center">
          {icon}
        </div>
      )}
      <h3 className="text-sm font-semibold text-[var(--text-secondary)] mb-1">{title}</h3>
      {description && (
        <p className="text-xs text-[var(--text-muted)] mb-5 max-w-xs leading-relaxed">{description}</p>
      )}
      {action}
    </div>
  );
}

// ── PageHeader ────────────────────────────────────────────────────────────────

export function PageHeader({
  title,
  subtitle,
  actions,
}: {
  title: string;
  subtitle?: string;
  actions?: React.ReactNode;
}) {
  return (
    <div className="flex items-start justify-between mb-6">
      <div>
        <h1 className="text-lg font-semibold text-[var(--text-primary)] tracking-tight">{title}</h1>
        {subtitle && (
          <p className="text-xs text-[var(--text-muted)] mt-0.5 font-normal">{subtitle}</p>
        )}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  );
}

// ── Modal ─────────────────────────────────────────────────────────────────────

interface ModalProps {
  open: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
  footer?: React.ReactNode;
  width?: string;
}

export function Modal({ open, onClose, title, children, footer, width = "max-w-lg" }: ModalProps) {
  if (!open) return null;
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 animate-fade-in"
      role="dialog"
      aria-modal="true"
      aria-label={title}
    >
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
        aria-hidden="true"
      />
      <div
        className={`relative bg-[var(--bg-surface)] rounded-xl shadow-card-lg w-full ${width} border border-[var(--border-base)] animate-slide-up`}
      >
        <div className="flex items-center justify-between px-6 py-4 border-b border-[var(--border-subtle)]">
          <h2 className="text-sm font-semibold text-[var(--text-primary)]">{title}</h2>
          <button
            onClick={onClose}
            className="text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors p-1 rounded-md hover:bg-[var(--bg-subtle)]"
            aria-label="Close"
          >
            <X size={14} />
          </button>
        </div>
        <div className="p-6">{children}</div>
        {footer && (
          <div className="flex justify-end gap-3 px-6 py-4 border-t border-[var(--border-subtle)]">
            {footer}
          </div>
        )}
      </div>
    </div>
  );
}

// ── KPI Card ──────────────────────────────────────────────────────────────────

export function KpiCard({
  label,
  value,
  sub,
  icon,
  color = "accent",
  trend,
}: {
  label: string;
  value: string | number;
  sub?: string;
  icon?: React.ReactNode;
  color?: "accent" | "green" | "amber" | "red";
  trend?: { value: number; label: string };
}) {
  const iconColors: Record<string, string> = {
    accent: "bg-[var(--accent-dim)] text-[var(--accent)]",
    green:  "bg-[var(--success-dim)] text-[var(--success)]",
    amber:  "bg-[var(--warn-dim)] text-[var(--warn)]",
    red:    "bg-[var(--danger-dim)] text-[var(--danger)]",
    indigo: "bg-[var(--accent-dim)] text-[var(--accent)]",
  };

  return (
    <Card className="group">
      <div className="flex items-start justify-between">
        <div className="flex-1 min-w-0">
          <p className="text-2xs font-semibold text-[var(--text-muted)] uppercase tracking-widest mb-2">
            {label}
          </p>
          <p className="text-3xl font-mono font-medium text-[var(--text-primary)] stat-value tabular-nums leading-none">
            {value}
          </p>
          {sub && (
            <p className="text-2xs text-[var(--text-muted)] mt-1.5">{sub}</p>
          )}
          {trend && (
            <p
              className={`text-2xs mt-1.5 font-medium ${
                trend.value >= 0 ? "text-[var(--success)]" : "text-[var(--danger)]"
              }`}
            >
              {trend.value >= 0 ? "+" : ""}{trend.value}% {trend.label}
            </p>
          )}
        </div>
        {icon && (
          <div
            className={`w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0 ${iconColors[color] ?? iconColors.accent}`}
          >
            {icon}
          </div>
        )}
      </div>
    </Card>
  );
}

// ── Density dot ───────────────────────────────────────────────────────────────

export function DensityDot({ count }: { count: number }) {
  const color =
    count >= 200
      ? "bg-[var(--danger)]"
      : count >= 100
      ? "bg-orange-500"
      : count >= 50
      ? "bg-[var(--warn)]"
      : "bg-[var(--success)]";
  return <span className={`inline-block w-1.5 h-1.5 rounded-full ${color} flex-shrink-0`} />;
}

// ── Status pill ───────────────────────────────────────────────────────────────

export function StatusPill({
  active,
  labelOn = "Active",
  labelOff = "Inactive",
}: {
  active: boolean;
  labelOn?: string;
  labelOff?: string;
}) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md text-2xs font-medium ${
        active
          ? "bg-[var(--success-dim)] text-[var(--success)]"
          : "bg-[var(--bg-subtle)] text-[var(--text-muted)]"
      }`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${active ? "bg-[var(--success)] animate-pulse" : "bg-[var(--text-muted)]"}`} />
      {active ? labelOn : labelOff}
    </span>
  );
}

// ── Data row (for lists) ──────────────────────────────────────────────────────

export function DataRow({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`flex items-center gap-4 px-5 py-3 border-b border-[var(--border-subtle)] last:border-0 hover:bg-[var(--bg-subtle)] transition-colors ${className}`}
    >
      {children}
    </div>
  );
}

// ── Alert banner ──────────────────────────────────────────────────────────────

export function AlertBanner({
  type = "info",
  children,
}: {
  type?: "info" | "warn" | "error" | "success";
  children: React.ReactNode;
}) {
  const styles = {
    info:    "bg-[var(--accent-dim)] border-[var(--accent)]/30 text-[var(--accent)]",
    warn:    "bg-[var(--warn-dim)] border-[var(--warn)]/30 text-[var(--warn)]",
    error:   "bg-[var(--danger-dim)] border-[var(--danger)]/30 text-[var(--danger)]",
    success: "bg-[var(--success-dim)] border-[var(--success)]/30 text-[var(--success)]",
  };
  return (
    <div className={`rounded-lg border px-4 py-3 text-sm ${styles[type]}`}>
      {children}
    </div>
  );
}