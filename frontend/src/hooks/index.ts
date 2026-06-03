/**
 * Custom React hooks — data fetching, polling, notifications.
 */

import { useCallback, useEffect, useState } from "react";
import { NOTIFICATION_POLL_INTERVAL_MS } from "../constants";
import { notificationsApi } from "../api/client";
import { useAuthStore } from "../auth/store";

// ── Generic async data hook ───────────────────────────────────────────────────

export interface UseAsyncState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  refetch: () => void;
}

export function useAsync<T>(
  fetcher: () => Promise<{ data: T }>,
  deps: unknown[] = [],
  { enabled = true }: { enabled?: boolean } = {}
): UseAsyncState<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetch = useCallback(async () => {
    if (!enabled) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetcher();
      setData(res.data);
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
        ?? "Request failed";
      setError(typeof msg === "string" ? msg : JSON.stringify(msg));
    } finally {
      setLoading(false);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, ...deps]);

  useEffect(() => {
    fetch();
  }, [fetch]);

  return { data, loading, error, refetch: fetch };
}

// ── Polling hook ─────────────────────────────────────────────────────────[[...]

export function usePolling<T>(
  fetcher: () => Promise<{ data: T }>,
  intervalMs: number,
  deps: unknown[] = []
): UseAsyncState<T> {
  const state = useAsync<T>(fetcher, deps);
  const { refetch } = state;

  useEffect(() => {
    const id = setInterval(refetch, intervalMs);
    return () => clearInterval(id);
  }, [refetch, intervalMs]);

  return state;
}

// ── Unread notification count ─────────────────────────────────────────────────

export function useUnreadCount() {
  const { token } = useAuthStore();
  const [count, setCount] = useState(0);

  const fetch = useCallback(async () => {
    if (!token) return;
    try {
      const res = await notificationsApi.countUnread();
      setCount(res.data.unread_count);
    } catch {
      // silent — badge is non-critical
    }
  }, [token]);

  useEffect(() => {
    fetch();
    const id = setInterval(fetch, NOTIFICATION_POLL_INTERVAL_MS);
    return () => clearInterval(id);
  }, [fetch]);

  return { count, refresh: fetch };
}

// ── Theme mode: "light" or "ops" ──────────────────────────────────────────────

export type ThemeMode = "light" | "ops";

export function useThemeMode(): [ThemeMode, () => void] {
  const [mode, setMode] = useState<ThemeMode>(() => {
    const stored = localStorage.getItem("ps_theme_mode") as ThemeMode | null;
    if (stored === "light" || stored === "ops") return stored;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "ops" : "light";
  });

  useEffect(() => {
    const html = document.documentElement;
    html.classList.remove("light", "ops", "dark");
    html.classList.add(mode);
    // Keep "dark" class in sync for any tailwind dark: utilities
    if (mode === "ops") html.classList.add("dark");
    localStorage.setItem("ps_theme_mode", mode);
  }, [mode]);

  const toggle = useCallback(() => {
    setMode((m) => (m === "light" ? "ops" : "light"));
  }, []);

  return [mode, toggle];
}

// Legacy alias — some pages may still call useDarkMode
export function useDarkMode(): [boolean, () => void] {
  const [mode, toggle] = useThemeMode();
  return [mode === "ops", toggle];
}

// ── Local storage state ───────────────────────────────────────────────────────[...]

export function useLocalStorage<T>(key: string, initial: T): [T, (val: T) => void] {
  const [value, setValue] = useState<T>(() => {
    try {
      const stored = localStorage.getItem(key);
      return stored !== null ? (JSON.parse(stored) as T) : initial;
    } catch {
      return initial;
    }
  });

  const set = useCallback(
    (val: T) => {
      localStorage.setItem(key, JSON.stringify(val));
      setValue(val);
    },
    [key]
  );

  return [value, set];
}

// ── Debounce ──────────────────────────────────────────────────────────[[...]

export function useDebounce<T>(value: T, delayMs: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const id = setTimeout(() => setDebounced(value), delayMs);
    return () => clearTimeout(id);
  }, [value, delayMs]);
  return debounced;
}
