import { Navigate, Route, Routes } from "react-router-dom";
import { BrainCircuit } from "lucide-react";
import ProtectedRoute from "./components/ProtectedRoute";
import { useAuth } from "./context/AuthContext";
import AuthPage from "./pages/AuthPage";
import ChatPage from "./pages/ChatPage";

function HomeRedirect() {
  const { authReady, isAuthenticated } = useAuth();

  if (!authReady) {
    return <div className="status-banner">Checking session...</div>;
  }

  return <Navigate to={isAuthenticated ? "/chat" : "/auth"} replace />;
}

export default function App() {
  return (
    <div className="app-shell">
      <div className="ambient-glow glow-top" aria-hidden="true" />
      <div className="ambient-glow glow-bottom" aria-hidden="true" />

      <header className="app-header">
        <div className="brand-mark" aria-hidden="true">
          <BrainCircuit size={18} />
        </div>
        <div>
          <h1>Course Copilot</h1>
          <p className="header-subtitle">Focused answers from your learning material.</p>
        </div>
      </header>

      <main className="app-main">
        <Routes>
          <Route path="/" element={<HomeRedirect />} />
          <Route path="/auth" element={<AuthPage />} />
          <Route
            path="/chat"
            element={
              <ProtectedRoute>
                <ChatPage />
              </ProtectedRoute>
            }
          />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}
