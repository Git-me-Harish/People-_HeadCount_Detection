import { useEffect } from "react";
import { Navigate, Route, Routes, useNavigate } from "react-router-dom";
import Layout from "./components/Layout";
import ProtectedRoute from "./components/ProtectedRoute";
import { api } from "./api/client";
import { useAuthStore } from "./auth/store";
import Login from "./pages/Login";
import Register from "./pages/Register";
import Dashboard from "./pages/Dashboard";
import ImageDetect from "./pages/ImageDetect";
import VideoDetect from "./pages/VideoDetect";
import LiveStream from "./pages/LiveStream";
import Cameras from "./pages/Cameras";
import Analytics from "./pages/Analytics";
import Alerts from "./pages/Alerts";
import Settings from "./pages/Settings";

function App() {
  const { token, user, setUser, clear } = useAuthStore();
  const navigate = useNavigate();

  useEffect(() => {
    if (token && !user) {
      api
        .get("/auth/me")
        .then((r) => setUser(r.data))
        .catch(() => {
          clear();
          navigate("/login");
        });
    }
  }, [token, user, setUser, clear, navigate]);

  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/register" element={<Register />} />
      <Route
        path="/"
        element={
          <ProtectedRoute>
            <Layout />
          </ProtectedRoute>
        }
      >
        <Route index element={<Dashboard />} />
        <Route path="image" element={<ImageDetect />} />
        <Route path="video" element={<VideoDetect />} />
        <Route path="stream" element={<LiveStream />} />
        <Route path="cameras" element={<Cameras />} />
        <Route path="analytics" element={<Analytics />} />
        <Route path="alerts" element={<Alerts />} />
        <Route path="settings" element={<Settings />} />
      </Route>
      <Route path="*" element={<Navigate to="/" />} />
    </Routes>
  );
}

export default App;
