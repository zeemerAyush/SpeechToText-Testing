import type { ReactNode } from "react";
import { HashRouter, Navigate, NavLink, Route, Routes } from "react-router-dom";
import { AuthProvider, useAuth } from "./auth";
import Login from "./pages/Login";
import NewRecording from "./pages/NewRecording";
import Recordings from "./pages/Recordings";
import TranscriptPage from "./pages/TranscriptPage";

function Protected({ children, roles }: { children: ReactNode; roles?: string[] }) {
  const { session } = useAuth();
  if (!session) return <Navigate to="/login" replace />;
  if (roles && !roles.includes(session.role)) return <Navigate to="/" replace />;
  return <>{children}</>;
}

function Shell({ children }: { children: ReactNode }) {
  const { session, signOut } = useAuth();
  const canUpload = session?.role === "admin" || session?.role === "uploader";
  return (
    <>
      <header className="topbar">
        <strong className="brand">Transcription</strong>
        <nav>
          <NavLink to="/" end>
            Recordings
          </NavLink>
          {canUpload && <NavLink to="/new">New recording</NavLink>}
        </nav>
        <span className="spacer" />
        <span className="user">
          {session?.username} <span className="role">{session?.role}</span>
        </span>
        <button className="link" onClick={signOut}>
          Sign out
        </button>
      </header>
      <main className="page">{children}</main>
    </>
  );
}

export default function App() {
  // HashRouter keeps page URLs (#/...) separate from the API paths (/recordings/...) on the same server.
  return (
    <AuthProvider>
      <HashRouter>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route
            path="/"
            element={
              <Protected>
                <Shell>
                  <Recordings />
                </Shell>
              </Protected>
            }
          />
          <Route
            path="/new"
            element={
              <Protected roles={["admin", "uploader"]}>
                <Shell>
                  <NewRecording />
                </Shell>
              </Protected>
            }
          />
          <Route
            path="/recordings/:id"
            element={
              <Protected>
                <Shell>
                  <TranscriptPage />
                </Shell>
              </Protected>
            }
          />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </HashRouter>
    </AuthProvider>
  );
}
