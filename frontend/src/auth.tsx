import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import * as api from "./api";
import type { Session } from "./types";

interface AuthValue {
  session: Session | null;
  signIn: (username: string, password: string) => Promise<void>;
  signOut: () => void;
}

const AuthContext = createContext<AuthValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session | null>(() => api.loadSession());

  const signOut = useCallback(() => {
    api.clearSession();
    setSession(null);
  }, []);

  // An expired/invalid token anywhere in the app sends the user back to the login page.
  useEffect(() => {
    api.setUnauthorizedHandler(() => setSession(null));
    return () => api.setUnauthorizedHandler(null);
  }, []);

  const signIn = useCallback(async (username: string, password: string) => {
    const s = await api.login(username, password);
    api.saveSession(s);
    setSession(s);
  }, []);

  const value = useMemo(() => ({ session, signIn, signOut }), [session, signIn, signOut]);
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside <AuthProvider>");
  return ctx;
}
