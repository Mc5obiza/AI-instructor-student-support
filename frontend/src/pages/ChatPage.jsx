import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import { AnimatePresence, motion } from "framer-motion";
import {
  Bot,
  LogOut,
  MessageSquareText,
  Plus,
  SendHorizontal,
  Sparkles,
  Trash2,
  UserCircle2,
} from "lucide-react";
import {
  createNewChatSession,
  deleteChatSession,
  fetchChatMessages,
  fetchChatSessions,
  streamPrompt,
} from "../api/chatApi";
import { useAuth } from "../context/AuthContext";

function truncateTitle(title) {
  const raw = String(title || "Untitled Chat");
  if (raw.length <= 32) {
    return raw;
  }
  return `${raw.slice(0, 29)}...`;
}

function isLikelyCodeLine(line) {
  const trimmed = String(line || "").trim();
  return (
    /\b[A-Za-z_]\w*(?:\[[^\]\n]+\])?(?:\.[A-Za-z_]\w*)+\s*\(/.test(trimmed) ||
    /\b(?:inplace|axis|value|dtype)\s*=\s*(?:True|False|None|\d+|["'])/.test(trimmed) ||
    /^(?:def|class|return|import|from|for|while|if|else|elif|try|except|print)\b/.test(trimmed)
  );
}

function isLikelyEquationLine(line) {
  const trimmed = String(line || "").trim();
  if (!trimmed) {
    return false;
  }

  if (
    isLikelyCodeLine(trimmed) ||
    trimmed.includes("$$") ||
    trimmed.includes("\\begin{") ||
    trimmed.includes("\\end{") ||
    trimmed.startsWith("```") ||
    trimmed.endsWith(":")
  ) {
    return false;
  }

  if (!trimmed.includes("=")) {
    return false;
  }

  const hasLeftHandVariable = /(?:\\[A-Za-z]+(?:\{[^{}]*\})*|[A-Za-z][A-Za-z0-9_{}^]*)\s*=/.test(trimmed);
  const hasMathSignals = /[+\-*/^()]/.test(trimmed) || /_[A-Za-z0-9]/.test(trimmed) || /\\[A-Za-z]+/.test(trimmed);
  return hasLeftHandVariable && hasMathSignals;
}

function splitPackedEquations(line) {
  const trimmed = String(line || "").trim();
  if (!trimmed || trimmed.includes("$$") || trimmed.includes("\\begin{") || trimmed.includes("\\end{")) {
    return [line];
  }

  const packedMatches = trimmed.match(
    /(?:\\[A-Za-z]+(?:\{[^{}]*\})*|[A-Za-z][A-Za-z0-9_{}^]*)\s*=\s*.+?(?=(?:\s+(?:\\[A-Za-z]+(?:\{[^{}]*\})*|[A-Za-z][A-Za-z0-9_{}^]*)\s*=)|$)/g,
  );

  if (packedMatches && packedMatches.length > 1) {
    return packedMatches.map((item) => item.trim()).filter(Boolean);
  }

  return [line];
}

function toDisplayMathBlock(expression) {
  const trimmed = String(expression || "").trim();
  return `$$\n${trimmed}\n$$`;
}

function normalizeMathMarkdown(content) {
  const text = String(content || "");
  const lines = text.split(/\r?\n/);
  const normalized = [];
  let insideCodeFence = false;
  let insideGeneratedCode = false;
  let insideDisplayMath = false;

  const appendPlainLine = (line) => {
    if (isLikelyCodeLine(line)) {
      if (!insideGeneratedCode) {
        normalized.push("```python");
        insideGeneratedCode = true;
      }
      normalized.push(line);
      return;
    }

    const expandedLines = splitPackedEquations(line);
    for (const expanded of expandedLines) {
      if (isLikelyEquationLine(expanded)) {
        normalized.push(toDisplayMathBlock(expanded));
        normalized.push("");
      } else {
        if (insideGeneratedCode && expanded.trim()) {
          normalized.push("```");
          insideGeneratedCode = false;
        }
        normalized.push(expanded);
      }
    }
  };

  for (const line of lines) {
    if (
      insideGeneratedCode &&
      line.trim() &&
      !isLikelyCodeLine(line) &&
      !/^\s*(`{3,}|~{3,})/.test(line)
    ) {
      normalized.push("```");
      insideGeneratedCode = false;
    }

    if (/^\s*(`{3,}|~{3,})/.test(line)) {
      insideCodeFence = !insideCodeFence;
      normalized.push(line);
      continue;
    }

    if (insideCodeFence) {
      normalized.push(line);
      continue;
    }

    if (insideDisplayMath) {
      const closingIndex = line.indexOf("$$");
      if (closingIndex === -1) {
        normalized.push(line);
        continue;
      }

      const mathContent = line.slice(0, closingIndex).trimEnd();
      if (mathContent) {
        normalized.push(mathContent);
      }
      normalized.push("$$");
      insideDisplayMath = false;
      const trailingText = line.slice(closingIndex + 2).trim();
      if (trailingText) {
        appendPlainLine(trailingText);
      }
      continue;
    }

    const displayMathPattern = /\$\$([\s\S]*?)\$\$/g;
    let previousIndex = 0;
    let match = displayMathPattern.exec(line);
    if (match) {
      while (match) {
        const leadingText = line.slice(previousIndex, match.index).trim();
        if (leadingText) {
          appendPlainLine(leadingText);
        }
        normalized.push(toDisplayMathBlock(match[1]));
        previousIndex = match.index + match[0].length;
        match = displayMathPattern.exec(line);
      }

      const trailingText = line.slice(previousIndex).trim();
      if (trailingText) {
        appendPlainLine(trailingText);
      }
      continue;
    }

    const openingIndex = line.indexOf("$$");
    if (openingIndex !== -1 && line.slice(0, openingIndex).trim() === "") {
      normalized.push("$$");
      const mathContent = line.slice(openingIndex + 2).trim();
      if (mathContent) {
        normalized.push(mathContent);
      }
      insideDisplayMath = true;
      continue;
    }

    appendPlainLine(line);
  }

  if (insideGeneratedCode) {
    normalized.push("```");
  }

  while (normalized.length > 0 && normalized[normalized.length - 1] === "") {
    normalized.pop();
  }

  return normalized.join("\n");
}

export default function ChatPage() {
  const navigate = useNavigate();
  const { backendUrl, logout } = useAuth();

  const [chatSessionsState, setChatSessionsState] = useState({});
  const [chatOrder, setChatOrder] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [prompt, setPrompt] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [isSyncing, setIsSyncing] = useState(false);
  const [pendingDeleteSession, setPendingDeleteSession] = useState(null);
  const [deletingSessionId, setDeletingSessionId] = useState(null);
  const [deleteError, setDeleteError] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [infoMessage, setInfoMessage] = useState("");

  const sessionsRef = useRef({});
  const streamingSessionRef = useRef(null);
  const messagesBoxRef = useRef(null);
  const deleteDialogRef = useRef(null);
  const chatSessions = chatSessionsState;

  const setChatSessions = useCallback((updaterOrValue) => {
    setChatSessionsState((previous) => {
      const next =
        typeof updaterOrValue === "function"
          ? updaterOrValue(previous)
          : updaterOrValue;
      sessionsRef.current = next;
      return next;
    });
  }, []);

  const syncSessionsFromBackend = useCallback(async () => {
    setIsSyncing(true);

    const result = await fetchChatSessions(backendUrl);
    setIsSyncing(false);

    // Avoid clobbering live token-by-token updates while a stream is active.
    if (streamingSessionRef.current) {
      return true;
    }

    if (!result.ok) {
      setErrorMessage(result.error || "Could not sync sessions");
      return false;
    }

    const previousSessions = sessionsRef.current;
    const nextSessions = {};
    const nextOrder = [];

    for (const item of result.sessions) {
      if (!item || typeof item !== "object") {
        continue;
      }

      const sessionId = String(item.session_id || "").trim();
      if (!sessionId) {
        continue;
      }

      const backendTitle = String(item.title || "Chat session");
      const previous = previousSessions[sessionId] || {};
      const previousTitle = String(previous.title || "").trim();
      const title =
        previousTitle &&
        backendTitle === "Chat session"
          ? previousTitle
          : backendTitle;
      const previousMessages = Array.isArray(previous.messages) ? previous.messages : [];

      nextSessions[sessionId] = {
        title,
        messages: previousMessages,
        loaded: Boolean(previous.loaded || previousMessages.length > 0),
      };
      nextOrder.push(sessionId);
    }

    sessionsRef.current = nextSessions;
    setChatSessions(nextSessions);
    setChatOrder(nextOrder);
    setActiveSessionId((previousActive) => {
      if (previousActive && nextSessions[previousActive]) {
        return previousActive;
      }
      return nextOrder[0] || null;
    });

    return true;
  }, [backendUrl]);

  useEffect(() => {
    let active = true;

    const run = async () => {
      const ok = await syncSessionsFromBackend();
      if (!active) {
        return;
      }
      if (ok) {
        setErrorMessage("");
      }
    };

    run();

    return () => {
      active = false;
    };
  }, [syncSessionsFromBackend]);

  useEffect(() => {
    if (!activeSessionId) {
      return;
    }

    const selected = chatSessions[activeSessionId];
    if (!selected || selected.loaded) {
      return;
    }

    if (streamingSessionRef.current === activeSessionId) {
      return;
    }

    let active = true;

    const run = async () => {
      const result = await fetchChatMessages(backendUrl, activeSessionId);
      if (!active) {
        return;
      }

      if (!result.ok) {
        setErrorMessage(result.error || "Could not load messages");
        return;
      }

      setChatSessions((previous) => {
        const existing = previous[activeSessionId] || { title: "Chat session", messages: [], loaded: false };
        return {
          ...previous,
          [activeSessionId]: {
            ...existing,
            messages: result.messages,
            loaded: true,
          },
        };
      });
    };

    run();

    return () => {
      active = false;
    };
  }, [activeSessionId, backendUrl, chatSessions]);

  const handleLogout = async () => {
    const result = await logout();
    setInfoMessage(result.message || "Successfully logged out");
    navigate("/auth", { replace: true });
  };

  const handleNewChat = async () => {
    setErrorMessage("");
    setInfoMessage("");

    if (!activeSessionId) {
      setInfoMessage("Current chat is empty. No new chat created.");
      return;
    }

    const activeSession = chatSessions[activeSessionId];
    if (activeSession && activeSession.loaded && Array.isArray(activeSession.messages) && activeSession.messages.length === 0) {
      setInfoMessage("Current chat is empty. No new chat created.");
      return;
    }

    const messagesResult = await fetchChatMessages(backendUrl, activeSessionId);
    if (!messagesResult.ok) {
      setErrorMessage(messagesResult.error || "Could not validate current chat");
      return;
    }

    setChatSessions((previous) => {
      const existing = previous[activeSessionId] || { title: "Chat session", messages: [], loaded: false };
      return {
        ...previous,
        [activeSessionId]: {
          ...existing,
          messages: messagesResult.messages,
          loaded: true,
        },
      };
    });

    if (messagesResult.messages.length === 0) {
      setInfoMessage("Current chat is empty. No new chat created.");
      return;
    }

    // Switch to a draft chat. The backend session is created only when the user sends a message.
    setActiveSessionId(null);
    setInfoMessage("New chat ready. Session will be created when you send a message.");
  };

  const requestDeleteSession = (sessionId, title) => {
    if (streamingSessionRef.current === sessionId) {
      return;
    }

    setErrorMessage("");
    setInfoMessage("");
    setDeleteError("");
    setPendingDeleteSession({ sessionId, title });
  };

  const handleDeleteSession = async () => {
    if (!pendingDeleteSession) {
      return;
    }

    const { sessionId } = pendingDeleteSession;
    setDeletingSessionId(sessionId);

    const result = await deleteChatSession(backendUrl, sessionId);

    if (!result.ok) {
      setDeletingSessionId(null);
      setDeleteError(result.error || "Could not delete session");
      return;
    }

    setDeletingSessionId(null);
    setPendingDeleteSession(null);
    const remainingOrder = chatOrder.filter((item) => item !== sessionId);
    setChatSessions((previous) => {
      const next = { ...previous };
      delete next[sessionId];
      return next;
    });
    setChatOrder(remainingOrder);
    setActiveSessionId((activeId) =>
      activeId === sessionId ? remainingOrder[0] || null : activeId,
    );
    setInfoMessage(result.message || "Session deleted");
  };

  const currentMessages = useMemo(() => {
    if (!activeSessionId || !chatSessions[activeSessionId]) {
      return [];
    }
    return Array.isArray(chatSessions[activeSessionId].messages)
      ? chatSessions[activeSessionId].messages
      : [];
  }, [activeSessionId, chatSessions]);

  useEffect(() => {
    const element = messagesBoxRef.current;
    if (!element) {
      return;
    }

    const distanceToBottom = element.scrollHeight - element.scrollTop - element.clientHeight;
    const shouldStickToBottom = distanceToBottom < 140 || isStreaming;
    if (shouldStickToBottom) {
      element.scrollTop = element.scrollHeight;
    }
  }, [currentMessages, isStreaming]);

  const ensureSessionForPrompt = useCallback(
    async (rawPrompt) => {
      if (activeSessionId) {
        return activeSessionId;
      }

      const created = await createNewChatSession(backendUrl);
      if (!created.ok || !created.sessionId) {
        setErrorMessage(created.error || "Could not create session for message");
        return null;
      }

      const sessionId = created.sessionId;
      setChatSessions((previous) => ({
        ...previous,
        [sessionId]: {
          title: rawPrompt.slice(0, 60) || "New Chat",
          messages: [],
          loaded: true,
        },
      }));
      setChatOrder((previous) => (previous.includes(sessionId) ? previous : [sessionId, ...previous]));
      setActiveSessionId(sessionId);
      return sessionId;
    },
    [activeSessionId, backendUrl],
  );

  const handleSendPrompt = async (event) => {
    event.preventDefault();
    if (isStreaming) {
      return;
    }

    const cleanPrompt = prompt.trim();
    if (!cleanPrompt) {
      return;
    }

    setErrorMessage("");
    setInfoMessage("");
    setPrompt("");

    const targetSessionId = await ensureSessionForPrompt(cleanPrompt);
    if (!targetSessionId) {
      return;
    }

    streamingSessionRef.current = targetSessionId;
    setIsStreaming(true);

    setChatSessions((previous) => {
      const existing = previous[targetSessionId] || { title: "New Chat", messages: [], loaded: true };
      const nextTitle = String(existing.title || "").trim() || "New Chat";

      return {
        ...previous,
        [targetSessionId]: {
          ...existing,
          title: nextTitle,
          loaded: true,
          messages: [
            ...existing.messages,
            { role: "user", content: cleanPrompt },
            { role: "assistant", content: "" },
          ],
        },
      };
    });

    setChatOrder((previous) => {
      const without = previous.filter((item) => item !== targetSessionId);
      return [targetSessionId, ...without];
    });
    setActiveSessionId(targetSessionId);

    let accumulated = "";

    try {
      const streamResult = await streamPrompt({
        baseUrl: backendUrl,
        prompt: cleanPrompt,
        sessionId: targetSessionId,
        onToken: (token) => {
          accumulated += token;
          setChatSessions((previous) => {
            const existing = previous[targetSessionId];
            if (!existing) {
              return previous;
            }

            const nextMessages = Array.isArray(existing.messages) ? [...existing.messages] : [];
            const lastIndex = nextMessages.length - 1;
            const lastRole = lastIndex >= 0 ? String(nextMessages[lastIndex]?.role || "") : "";

            if (lastRole === "assistant") {
              nextMessages[lastIndex] = {
                role: "assistant",
                content: accumulated,
              };
            } else {
              nextMessages.push({
                role: "assistant",
                content: accumulated,
              });
            }

            return {
              ...previous,
              [targetSessionId]: {
                ...existing,
                loaded: true,
                messages: nextMessages,
              },
            };
          });
        },
      });

      if (streamResult?.sessionTitle) {
        setChatSessions((previous) => {
          const existing = previous[targetSessionId];
          if (!existing) {
            return previous;
          }

          return {
            ...previous,
            [targetSessionId]: {
              ...existing,
              title: String(streamResult.sessionTitle),
            },
          };
        });
      }
    } catch (error) {
      const message = String(error?.message || "Request failed");
      setErrorMessage(message);
      setChatSessions((previous) => {
        const existing = previous[targetSessionId];
        if (!existing) {
          return previous;
        }

        const nextMessages = Array.isArray(existing.messages) ? [...existing.messages] : [];
        const lastIndex = nextMessages.length - 1;
        const lastRole = lastIndex >= 0 ? String(nextMessages[lastIndex]?.role || "") : "";

        if (lastRole === "assistant") {
          nextMessages[lastIndex] = {
            role: "assistant",
            content: message,
          };
        } else {
          nextMessages.push({
            role: "assistant",
            content: message,
          });
        }

        return {
          ...previous,
          [targetSessionId]: {
            ...existing,
            loaded: true,
            messages: nextMessages,
          },
        };
      });
    } finally {
      streamingSessionRef.current = null;
      setIsStreaming(false);
      await syncSessionsFromBackend();
    }
  };

  return (
    <section className="chat-layout">
      <aside className="chat-sidebar" aria-label="Chat history">
        <div className="sidebar-heading">
          <div>
            <p className="sidebar-eyebrow">YOUR WORKSPACE</p>
            <h2>
              <MessageSquareText size={17} />
              Chats
            </h2>
          </div>
          {chatOrder.length > 0 ? (
            <span className="chat-count">{chatOrder.length}</span>
          ) : null}
        </div>

        <button type="button" className="primary-button" onClick={handleNewChat}>
          <Plus size={15} />
          New Chat
        </button>

        {chatOrder.length === 0 ? (
          <p className="sidebar-empty">Your conversations will show up here.</p>
        ) : null}

        <div className="chat-list" aria-label="Previous conversations">
          {chatOrder.map((sessionId) => {
            const item = chatSessions[sessionId] || {};
            const title = truncateTitle(item.title || "Untitled Chat");
            const active = sessionId === activeSessionId;

            return (
              <div className={`chat-item-row ${active ? "active" : ""}`} key={sessionId}>
                <button
                  type="button"
                  className="chat-item"
                  onClick={() => setActiveSessionId(sessionId)}
                  aria-pressed={active}
                >
                  {title}
                </button>
                <button
                  type="button"
                  className="delete-chat-button"
                  onClick={() => requestDeleteSession(sessionId, item.title || "Untitled Chat")}
                  disabled={deletingSessionId === sessionId || streamingSessionRef.current === sessionId}
                  aria-label={`Delete ${item.title || "Untitled Chat"}`}
                  title="Delete conversation"
                >
                  <Trash2 size={15} />
                </button>
              </div>
            );
          })}
        </div>
      </aside>

      <div className="chat-main">
        <div className="chat-topbar">
          <p className="session-badge">
            <Sparkles size={14} />
            Private learning session
          </p>
          <button type="button" className="secondary-button" onClick={handleLogout}>
            <LogOut size={15} />
            Logout
          </button>
        </div>

        {errorMessage ? <p className="error-text">{errorMessage}</p> : null}
        {infoMessage ? <p className="info-text">{infoMessage}</p> : null}
        {isSyncing ? <p className="muted-text">Syncing chats...</p> : null}

        <div className="messages-box" ref={messagesBoxRef} aria-live="polite">
          {currentMessages.length === 0 ? (
            <div className="chat-empty-state">
              <div className="empty-state-icon" aria-hidden="true">
                <Sparkles size={22} />
              </div>
              <p className="empty-state-eyebrow">YOUR STUDY COMPANION</p>
              <h2>What are you learning today?</h2>
              <p className="empty-state-copy">
                Ask a question about your course material and get a clear, grounded explanation.
              </p>
              <div className="suggested-prompts" aria-label="Suggested questions">
                {[
                  "When should I use a decision tree?",
                  "Explain gradient descent simply",
                  "How do I handle missing data?",
                ].map((suggestion) => (
                  <button
                    key={suggestion}
                    type="button"
                    className="suggestion-button"
                    onClick={() => setPrompt(suggestion)}
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <AnimatePresence initial={false}>
              {currentMessages.map((message, index) => (
                <motion.div
                  layout
                  key={`${message.role}-${index}`}
                  className={`message-row ${message.role}`}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.18 }}
                >
                  <div className="message-bubble">
                    <strong>
                      {message.role === "user" ? <UserCircle2 size={14} /> : <Bot size={14} />}
                      {message.role === "user" ? "You" : "Assistant"}
                    </strong>
                    <div className="message-markdown">
                      <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>
                        {normalizeMathMarkdown(String(message.content || ""))}
                      </ReactMarkdown>
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          )}
        </div>

        <form className="prompt-form" onSubmit={handleSendPrompt}>
          <input
            type="text"
            aria-label="Ask a question about your course material"
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            placeholder="Ask anything about your course material..."
            disabled={isStreaming}
          />
          <button type="submit" disabled={isStreaming}>
            <SendHorizontal size={15} />
            {isStreaming ? "Sending..." : "Send"}
          </button>
        </form>
      </div>
      {pendingDeleteSession ? (
        <div
          className="confirmation-backdrop"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget && !deletingSessionId) {
              setPendingDeleteSession(null);
              setDeleteError("");
            }
          }}
        >
          <section
            className="confirmation-dialog"
            ref={deleteDialogRef}
            role="alertdialog"
            aria-modal="true"
            aria-labelledby="delete-dialog-title"
            aria-describedby="delete-dialog-description"
            onKeyDown={(event) => {
              if (event.key === "Escape" && !deletingSessionId) {
                setPendingDeleteSession(null);
                setDeleteError("");
              } else if (event.key === "Tab") {
                const buttons = deleteDialogRef.current?.querySelectorAll("button:not(:disabled)");
                if (!buttons?.length) {
                  return;
                }

                const firstButton = buttons[0];
                const lastButton = buttons[buttons.length - 1];
                if (event.shiftKey && document.activeElement === firstButton) {
                  event.preventDefault();
                  lastButton.focus();
                } else if (!event.shiftKey && document.activeElement === lastButton) {
                  event.preventDefault();
                  firstButton.focus();
                }
              }
            }}
          >
            <div className="confirmation-icon" aria-hidden="true">
              <Trash2 size={20} />
            </div>
            <h2 id="delete-dialog-title">Delete this conversation?</h2>
            <p id="delete-dialog-description">
              <strong>{pendingDeleteSession.title}</strong> and its messages will be permanently deleted.
            </p>
            {deleteError ? <p className="confirmation-error" role="alert">{deleteError}</p> : null}
            <div className="confirmation-actions">
              <button
                type="button"
                className="confirmation-cancel"
                onClick={() => {
                  setPendingDeleteSession(null);
                  setDeleteError("");
                }}
                disabled={Boolean(deletingSessionId)}
                autoFocus
              >
                Cancel
              </button>
              <button
                type="button"
                className="confirmation-delete"
                onClick={handleDeleteSession}
                disabled={Boolean(deletingSessionId)}
              >
                <Trash2 size={15} />
                {deletingSessionId ? "Deleting..." : "Delete conversation"}
              </button>
            </div>
          </section>
        </div>
      ) : null}
    </section>
  );
}
