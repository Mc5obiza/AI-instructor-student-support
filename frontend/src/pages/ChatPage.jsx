import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { AnimatePresence, motion } from "framer-motion";
import {
  Bot,
  LogOut,
  MessageSquareText,
  Plus,
  SendHorizontal,
  Sparkles,
  UserCircle2,
} from "lucide-react";
import {
  createNewChatSession,
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

export default function ChatPage() {
  const navigate = useNavigate();
  const { backendUrl, logout } = useAuth();

  const [chatSessionsState, setChatSessionsState] = useState({});
  const [chatOrder, setChatOrder] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [prompt, setPrompt] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [isSyncing, setIsSyncing] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [infoMessage, setInfoMessage] = useState("");

  const sessionsRef = useRef({});
  const streamingSessionRef = useRef(null);
  const messagesBoxRef = useRef(null);
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
      <aside className="chat-sidebar">
        <h2>
          <MessageSquareText size={16} />
          Chats
        </h2>

        <button type="button" className="primary-button" onClick={handleNewChat}>
          <Plus size={15} />
          New Chat
        </button>

        {chatOrder.length === 0 ? <p className="muted-text">No chats yet.</p> : null}

        <div className="chat-list">
          {chatOrder.map((sessionId) => {
            const item = chatSessions[sessionId] || {};
            const title = truncateTitle(item.title || "Untitled Chat");
            const active = sessionId === activeSessionId;

            return (
              <button
                key={sessionId}
                type="button"
                className={`chat-item ${active ? "active" : ""}`}
                onClick={() => setActiveSessionId(sessionId)}
              >
                {title}
              </button>
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

        <div className="messages-box" ref={messagesBoxRef}>
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
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{String(message.content || "")}</ReactMarkdown>
                  </div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>

        <form className="prompt-form" onSubmit={handleSendPrompt}>
          <input
            type="text"
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            placeholder="Type your question..."
            disabled={isStreaming}
          />
          <button type="submit" disabled={isStreaming}>
            <SendHorizontal size={15} />
            {isStreaming ? "Sending..." : "Send"}
          </button>
        </form>
      </div>
    </section>
  );
}
