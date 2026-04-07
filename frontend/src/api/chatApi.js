import axios from "axios";
import { apiBase, responseDetail } from "./http";

export async function fetchChatSessions(baseUrl) {
  try {
    const response = await axios.get(`${apiBase(baseUrl)}/chat/sessions`, {
      withCredentials: true,
      validateStatus: () => true,
    });

    if (response.status >= 200 && response.status < 300) {
      const sessions = Array.isArray(response.data?.sessions) ? response.data.sessions : [];
      return { ok: true, sessions };
    }

    return { ok: false, error: responseDetail(response) };
  } catch (error) {
    return { ok: false, error: String(error?.message || "Could not fetch sessions") };
  }
}

export async function fetchChatMessages(baseUrl, sessionId) {
  try {
    const response = await axios.get(`${apiBase(baseUrl)}/chat/session/${sessionId}/messages`, {
      withCredentials: true,
      validateStatus: () => true,
    });

    if (response.status >= 200 && response.status < 300) {
      const rawMessages = Array.isArray(response.data?.messages) ? response.data.messages : [];
      const messages = rawMessages
        .filter((item) => item && typeof item === "object")
        .map((item) => ({
          role: String(item.role || "assistant"),
          content: String(item.content || ""),
        }));
      return { ok: true, messages };
    }

    return { ok: false, error: responseDetail(response) };
  } catch (error) {
    return { ok: false, error: String(error?.message || "Could not fetch messages") };
  }
}

export async function createNewChatSession(baseUrl) {
  try {
    const response = await axios.post(
      `${apiBase(baseUrl)}/chat/session/new`,
      {},
      {
        withCredentials: true,
        validateStatus: () => true,
      },
    );

    if (response.status >= 200 && response.status < 300) {
      return {
        ok: true,
        message: String(response.data?.message || "New session created"),
        sessionId: String(response.data?.session_id || "").trim() || null,
      };
    }

    return { ok: false, error: responseDetail(response), sessionId: null };
  } catch (error) {
    return { ok: false, error: String(error?.message || "Could not create session"), sessionId: null };
  }
}

export async function streamPrompt({ baseUrl, prompt, sessionId, onToken }) {
  const payload = { prompt };
  if (sessionId && String(sessionId).trim()) {
    payload.session_id = String(sessionId).trim();
  }

  const response = await fetch(`${apiBase(baseUrl)}/ask/stream`, {
    method: "POST",
    credentials: "include",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      if (body && typeof body.detail === "string" && body.detail.trim()) {
        detail = body.detail;
      }
    } catch (error) {
      // Keep fallback detail when response body is not JSON.
    }
    throw new Error(detail);
  }

  if (!response.body) {
    throw new Error("Empty stream response");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let eventBuffer = "";
  let finalSessionTitle = null;

  const handleSseEvent = (rawEvent) => {
    const dataLines = rawEvent
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trim())
      .filter(Boolean);

    if (dataLines.length === 0) {
      return false;
    }

    let event;
    try {
      event = JSON.parse(dataLines.join("\n"));
    } catch (error) {
      return false;
    }

    const type = String(event?.type || "");
    if (type === "token") {
      if (typeof onToken === "function") {
        onToken(String(event?.token || ""));
      }
      return false;
    }

    if (type === "error") {
      throw new Error(String(event?.error || "Unknown backend error"));
    }

    if (type === "done") {
      const maybeTitle = String(event?.session_title || "").trim();
      if (maybeTitle) {
        finalSessionTitle = maybeTitle;
      }
      return true;
    }

    return false;
  };

  while (true) {
    const { done, value } = await reader.read();

    eventBuffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    eventBuffer = eventBuffer.replace(/\r\n/g, "\n");

    let separatorIndex = eventBuffer.indexOf("\n\n");
    while (separatorIndex !== -1) {
      const rawEvent = eventBuffer.slice(0, separatorIndex);
      eventBuffer = eventBuffer.slice(separatorIndex + 2);

      if (handleSseEvent(rawEvent)) {
        return { sessionTitle: finalSessionTitle };
      }

      separatorIndex = eventBuffer.indexOf("\n\n");
    }

    if (done) {
      const trailingEvent = eventBuffer.trim();
      if (trailingEvent && handleSseEvent(trailingEvent)) {
        return { sessionTitle: finalSessionTitle };
      }
      break;
    }
  }

  return { sessionTitle: finalSessionTitle };
}
