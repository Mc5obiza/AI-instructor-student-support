import { Navigate } from "react-router-dom";
import { useState } from "react";
import { motion } from "framer-motion";
import { LockKeyhole, UserPlus, ShieldCheck, BookOpenText } from "lucide-react";
import { useAuth } from "../context/AuthContext";

export default function AuthPage() {
  const { authReady, isAuthenticated, login, register } = useAuth();

  const [activeTab, setActiveTab] = useState("login");
  const [loginEmail, setLoginEmail] = useState("");
  const [loginPassword, setLoginPassword] = useState("");
  const [signupUsername, setSignupUsername] = useState("");
  const [signupEmail, setSignupEmail] = useState("");
  const [signupPassword, setSignupPassword] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [infoMessage, setInfoMessage] = useState("Login or create an account to start chatting.");

  if (authReady && isAuthenticated) {
    return <Navigate to="/chat" replace />;
  }

  const handleLoginSubmit = async (event) => {
    event.preventDefault();
    setErrorMessage("");
    setInfoMessage("");
    setIsSubmitting(true);

    const result = await login({
      email: loginEmail.trim().toLowerCase(),
      password: loginPassword,
    });

    setIsSubmitting(false);

    if (result.ok) {
      setInfoMessage(result.message || "Login successful");
      return;
    }

    setErrorMessage(result.error || "Login failed");
  };

  const handleSignupSubmit = async (event) => {
    event.preventDefault();
    setErrorMessage("");
    setInfoMessage("");
    setIsSubmitting(true);

    const result = await register({
      username: signupUsername.trim(),
      email: signupEmail.trim().toLowerCase(),
      password: signupPassword,
    });

    setIsSubmitting(false);

    if (result.ok) {
      setInfoMessage(result.message || "Account created");
      return;
    }

    setErrorMessage(result.error || "Account creation failed");
  };

  return (
    <section className="auth-layout">
      <motion.aside
        className="auth-panel"
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35 }}
      >
        <p className="auth-eyebrow">AI LEARNING ASSISTANT</p>
        <h2>Study with confidence.</h2>
        <p className="muted-text">
          Ask questions, get context-aware answers, and keep your sessions organized.
        </p>

        <div className="feature-list">
          <p className="feature-chip">
            <ShieldCheck size={14} />
            Secure sessions
          </p>
          <p className="feature-chip">
            <BookOpenText size={14} />
            Course-focused answers
          </p>
        </div>
      </motion.aside>

      <motion.section
        className="auth-card"
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35, delay: 0.05 }}
      >
        <div className="tab-row">
          <button
            type="button"
            className={`tab-button ${activeTab === "login" ? "active" : ""}`}
            onClick={() => setActiveTab("login")}
          >
            <LockKeyhole size={15} />
            Login
          </button>
          <button
            type="button"
            className={`tab-button ${activeTab === "signup" ? "active" : ""}`}
            onClick={() => setActiveTab("signup")}
          >
            <UserPlus size={15} />
            Sign Up
          </button>
        </div>

        {activeTab === "login" ? (
          <form className="auth-form" onSubmit={handleLoginSubmit}>
            <label htmlFor="login-email">Email</label>
            <input
              id="login-email"
              type="email"
              value={loginEmail}
              onChange={(event) => setLoginEmail(event.target.value)}
              required
            />

            <label htmlFor="login-password">Password</label>
            <input
              id="login-password"
              type="password"
              value={loginPassword}
              onChange={(event) => setLoginPassword(event.target.value)}
              required
            />

            <button type="submit" disabled={isSubmitting}>
              {isSubmitting ? "Please wait..." : "Login"}
            </button>
          </form>
        ) : (
          <form className="auth-form" onSubmit={handleSignupSubmit}>
            <label htmlFor="signup-username">Username</label>
            <input
              id="signup-username"
              type="text"
              value={signupUsername}
              onChange={(event) => setSignupUsername(event.target.value)}
              required
            />

            <label htmlFor="signup-email">Email</label>
            <input
              id="signup-email"
              type="email"
              value={signupEmail}
              onChange={(event) => setSignupEmail(event.target.value)}
              required
            />

            <label htmlFor="signup-password">Password</label>
            <input
              id="signup-password"
              type="password"
              value={signupPassword}
              onChange={(event) => setSignupPassword(event.target.value)}
              required
            />

            <button type="submit" disabled={isSubmitting}>
              {isSubmitting ? "Please wait..." : "Create Account"}
            </button>
          </form>
        )}

        {errorMessage ? <p className="error-text">{errorMessage}</p> : null}
        {infoMessage ? <p className="info-text">{infoMessage}</p> : null}
      </motion.section>
    </section>
  );
}
