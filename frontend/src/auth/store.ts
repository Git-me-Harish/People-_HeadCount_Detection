import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { User } from "../api/client";

interface AuthState {
  token: string | null;
  user: User | null;
  setSession: (token: string, user?: User | null) => void;
  setUser: (user: User | null) => void;
  clear: () => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      token: null,
      user: null,
      setSession: (token, user) => set({ token, user: user ?? null }),
      setUser: (user) => set({ user }),
      clear: () => set({ token: null, user: null }),
    }),
    { name: "peoplesense-auth" },
  ),
);
