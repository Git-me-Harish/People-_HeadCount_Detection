import { create } from "zustand";
import type { User } from "../types";

interface AuthState {
  token: string | null;
  user: User | null;
  setToken: (token: string) => void;
  setUser: (user: User) => void;
  clear: () => void;
  isAdmin: () => boolean;
}

export const useAuthStore = create<AuthState>((set, get) => ({
  token: localStorage.getItem("ps_token"),
  user: null,

  setToken: (token) => {
    localStorage.setItem("ps_token", token);
    set({ token });
  },

  setUser: (user) => set({ user }),

  clear: () => {
    localStorage.removeItem("ps_token");
    set({ token: null, user: null });
  },

  isAdmin: () => get().user?.role === "admin",
}));
