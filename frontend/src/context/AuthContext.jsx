import Cookies from "js-cookie";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import {
  checkAuthSession,
  loginUser,
  logoutUser,
  registerUser,
} from "../api/authApi";
import { DEFAULT_BACKEND_URL } from "../api/http";

const AuthContext = createContext(null);

const BACKEND_URL_KEY = "frontend_backend_url";
const AUTH_FLAG_COOKIE = "frontend_auth";
const USER_EMAIL_COOKIE = "frontend_user_email";

function normalizeBackendUrl(rawUrl) {
  const value = String(rawUrl || "").trim();
  return value || DEFAULT_BACKEND_URL;
}

export function AuthProvider({ children }) {
  const [backendUrl, setBackendUrlState] = useState(() => {
    const remembered = window.localStorage.getItem(BACKEND_URL_KEY);
    return normalizeBackendUrl(remembered);
  });
  const [authReady, setAuthReady] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [userEmail, setUserEmail] = useState(() => Cookies.get(USER_EMAIL_COOKIE) || "");

  const clearClientAuth = useCallback(() => {
    Cookies.remove(AUTH_FLAG_COOKIE);
    Cookies.remove(USER_EMAIL_COOKIE);
    setIsAuthenticated(false);
    setUserEmail("");
  }, []);

  const applyClientAuth = useCallback((email) => {
    Cookies.set(AUTH_FLAG_COOKIE, "1", { sameSite: "Lax" });
    if (email && String(email).trim()) {
      Cookies.set(USER_EMAIL_COOKIE, String(email).trim().toLowerCase(), { sameSite: "Lax" });
      setUserEmail(String(email).trim().toLowerCase());
    } else {
      Cookies.remove(USER_EMAIL_COOKIE);
      setUserEmail("");
    }
    setIsAuthenticated(true);
  }, []);

  const setBackendUrl = useCallback((url) => {
    setBackendUrlState(normalizeBackendUrl(url));
  }, []);

  useEffect(() => {
    window.localStorage.setItem(BACKEND_URL_KEY, backendUrl);
  }, [backendUrl]);

  const refreshAuthFromServer = useCallback(async () => {
    setAuthReady(false);

    const result = await checkAuthSession(backendUrl);

    if (result.ok) {
      const rememberedEmail = Cookies.get(USER_EMAIL_COOKIE) || "";
      applyClientAuth(rememberedEmail);
      setAuthReady(true);
      return;
    }

    // Keep the user session during transient backend/network issues.
    if (Cookies.get(AUTH_FLAG_COOKIE) === "1") {
      setIsAuthenticated(true);
      setUserEmail(Cookies.get(USER_EMAIL_COOKIE) || "");
      setAuthReady(true);
      return;
    }

    clearClientAuth();
    setAuthReady(true);
  }, [applyClientAuth, backendUrl, clearClientAuth]);

  useEffect(() => {
    let active = true;

    const run = async () => {
      await refreshAuthFromServer();
      if (!active) {
        return;
      }
    };

    run();

    return () => {
      active = false;
    };
  }, [refreshAuthFromServer]);

  const login = useCallback(
    async ({ email, password }) => {
      const result = await loginUser(backendUrl, email, password);
      if (result.ok) {
        applyClientAuth(email);
      }
      return result;
    },
    [applyClientAuth, backendUrl],
  );

  const register = useCallback(
    async ({ username, email, password }) => {
      const result = await registerUser(backendUrl, username, email, password);
      if (result.ok) {
        applyClientAuth(email);
      }
      return result;
    },
    [applyClientAuth, backendUrl],
  );

  const logout = useCallback(async () => {
    const result = await logoutUser(backendUrl);
    clearClientAuth();
    return result.ok ? result : { ok: true, message: "Successfully logged out" };
  }, [backendUrl, clearClientAuth]);

  const value = useMemo(
    () => ({
      authReady,
      isAuthenticated,
      userEmail,
      backendUrl,
      setBackendUrl,
      login,
      register,
      logout,
      refreshAuthFromServer,
    }),
    [
      authReady,
      backendUrl,
      isAuthenticated,
      login,
      logout,
      refreshAuthFromServer,
      register,
      setBackendUrl,
      userEmail,
    ],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used inside AuthProvider");
  }
  return context;
}
