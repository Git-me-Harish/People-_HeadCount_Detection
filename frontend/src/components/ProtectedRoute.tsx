import { Navigate } from "react-router-dom";
import type { ReactNode } from "react";
import { useAuthStore } from "../auth/store";

export default function ProtectedRoute({ children }: { children: ReactNode }) {
  const token = useAuthStore((s) => s.token);
  if (!token) return <Navigate to="/login" replace />;
  return <>{children}</>;
}
