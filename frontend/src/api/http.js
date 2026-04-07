export const DEFAULT_BACKEND_URL = "http://127.0.0.1:8000";

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
    if (typeof payload.message === "string" && payload.message.trim()) {
      return payload.message;
    }
  }

  if (typeof response?.statusText === "string" && response.statusText.trim()) {
    return response.statusText;
  }

  return `HTTP ${response?.status ?? "Unknown"}`;
}
