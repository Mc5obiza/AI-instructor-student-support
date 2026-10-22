function inferBackendUrl() {
  if (typeof window === "undefined") {
    return "http://localhost:8000";
  }
  const host = String(window.location.hostname || "localhost").trim() || "localhost";
  const protocol = String(window.location.protocol || "http:").trim() || "http:";
  return `${protocol}//${host}:8000`;
}

export const DEFAULT_BACKEND_URL = String(
  import.meta.env.VITE_BACKEND_URL || inferBackendUrl(),
).trim();

export function apiBase(url) {
  const raw = String(url || DEFAULT_BACKEND_URL).trim();
  return raw.replace(/\/+$/, "");
}

export function responseDetail(response) {
  const payload = response?.data;

  if (payload && typeof payload === "object") {
    if (typeof payload.detail === "string" && payload.detail.trim()) {
      return payload.detail;
    }
    if (Array.isArray(payload.detail)) {
      const validationMessages = payload.detail
        .filter((item) => item && typeof item.msg === "string")
        .map((item) => {
          const location = Array.isArray(item.loc) ? item.loc.at(-1) : null;
          return location ? `${location}: ${item.msg}` : item.msg;
        });
      if (validationMessages.length > 0) {
        return validationMessages.join("; ");
      }
    }
    if (typeof payload.message === "string" && payload.message.trim()) {
      return payload.message;
    }
  }

  if (typeof response?.statusText === "string" && response.statusText.trim()) {
    return response.statusText;
  }

  return `HTTP ${response?.status ?? "Unknown"}`;
}
