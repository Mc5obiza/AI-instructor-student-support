import axios from "axios";
import { apiBase, responseDetail } from "./http";

export async function registerUser(baseUrl, username, email, password) {
  try {
    const response = await axios.post(
      `${apiBase(baseUrl)}/auth/register`,
      { username, email, password },
      {
        withCredentials: true,
        validateStatus: () => true,
      },
    );

    if (response.status === 201) {
      return { ok: true, message: "Account created" };
    }

    return { ok: false, error: responseDetail(response) };
  } catch (error) {
    return { ok: false, error: String(error?.message || "Register request failed") };
  }
}

export async function loginUser(baseUrl, email, password) {
  try {
    const body = new URLSearchParams();
    body.set("username", email);
    body.set("password", password);

    const response = await axios.post(`${apiBase(baseUrl)}/auth/login`, body, {
      withCredentials: true,
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
      },
      validateStatus: () => true,
    });

    if (response.status >= 200 && response.status < 300) {
      return { ok: true, message: "Login successful" };
    }

    return { ok: false, error: responseDetail(response) };
  } catch (error) {
    return { ok: false, error: String(error?.message || "Login request failed") };
  }
}

export async function logoutUser(baseUrl) {
  try {
    const response = await axios.get(`${apiBase(baseUrl)}/logout`, {
      withCredentials: true,
      validateStatus: () => true,
    });

    if (response.status >= 200 && response.status < 300) {
      return { ok: true, message: responseDetail(response) };
    }

    return { ok: false, error: responseDetail(response) };
  } catch (error) {
    return { ok: false, error: String(error?.message || "Logout request failed") };
  }
}

export async function checkAuthSession(baseUrl) {
  try {
    const response = await axios.get(`${apiBase(baseUrl)}/chat/sessions`, {
      withCredentials: true,
      validateStatus: () => true,
    });

    if (response.status >= 200 && response.status < 300) {
      return { ok: true, sessions: Array.isArray(response.data?.sessions) ? response.data.sessions : [] };
    }

    return { ok: false, error: responseDetail(response) };
  } catch (error) {
    return { ok: false, error: String(error?.message || "Auth check failed") };
  }
}
