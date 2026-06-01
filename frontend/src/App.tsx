import { useEffect } from "react";
import { Navigate, Route, Routes, useNavigate } from "react-router-dom";
import Layout from "./components/Layout";
import ProtectedRoute from "./components/ProtectedRoute";
import { authApi } from "./api/client";
import { useAuthStore } from "./auth/store";
import { ROUTES } from "./constants";
import "./i18n";

// Pages
import Landing from "./pages/Landing";
import Login from "./pages/Login";
import Register from "./pages/Register";
import Onboarding from "./pages/Onboarding";
import Dashboard from "./pages/Dashboard";
import ImageDetect from "./pages/ImageDetect";
import VideoDetect from "./pages/VideoDetect";
import LiveStream from "./pages/LiveStream";
import Cameras from "./pages/Cameras";
import Analytics from "./pages/Analytics";
import Alerts from "./pages/Alerts";
import Settings from "./pages/Settings";
import Notifications from "./pages/Notifications";
import ApiTokens from "./pages/ApiTokens";
import Reports from "./pages/Reports";
import PublicPageSettings from "./pages/PublicPageSettings";
import AuditLog from "./pages/AuditLog";
import PlanUsage from "./pages/PlanUsage";
import PublicView from "./pages/PublicView";

function App() {
  const { token, user, setUser, clear } = useAuthStore();
  const navigate = useNavigate();

  useEffect(() => {
    if (token && !user) {
      authApi
        .me()
        .then((r) => setUser(r.data))
        .catch(() => {
          clear();
          navigate(ROUTES.LOGIN);
        });
    }
  }, [token, user, setUser, clear, navigate]);

  return (
    <Routes>
      {/* Public unauthenticated routes */}
      <Route path="/" element={<Landing />} />
      <Route path="/login" element={<Login />} />
      <Route path="/register" element={<Register />} />
      <Route path="/public/:slug" element={<PublicView />} />

      {/* Onboarding (authenticated but outside layout) */}
      <Route
        path="/onboarding"
        element={
          <ProtectedRoute>
            <Onboarding />
          </ProtectedRoute>
        }
      />

      {/* App shell with sidebar */}
      <Route
        path="/dashboard"
        element={
          <ProtectedRoute>
            <Layout />
          </ProtectedRoute>
        }
      >
        <Route index element={<Dashboard />} />
      </Route>

      <Route
        path="/"
        element={
          <ProtectedRoute>
            <Layout />
          </ProtectedRoute>
        }
      >
        <Route path="cameras" element={<Cameras />} />
        <Route path="live" element={<LiveStream />} />
        <Route path="detect/image" element={<ImageDetect />} />
        <Route path="detect/video" element={<VideoDetect />} />
        <Route path="analytics" element={<Analytics />} />
        <Route path="alerts" element={<Alerts />} />
        <Route path="notifications" element={<Notifications />} />
        <Route path="reports" element={<Reports />} />
        <Route path="public-page" element={<PublicPageSettings />} />
        <Route path="api-tokens" element={<ApiTokens />} />
        <Route path="audit" element={<AuditLog />} />
        <Route path="plan" element={<PlanUsage />} />
        <Route path="settings" element={<Settings />} />
      </Route>

      <Route path="*" element={<Navigate to="/" />} />
    </Routes>
  );
}

export default App;
