from __future__ import annotations

import json
import base64
from typing import Any, Iterator

import requests
import streamlit as st
from streamlit_cookies_controller import CookieController  # pyright: ignore[reportMissingImports]


DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"
AUTH_COOKIE_NAME = "access_token"
CHAT_SESSION_COOKIE_NAME = "chat_session_id"


def _api_base(url: str) -> str:
    return url.rstrip("/")


def _get_http_session() -> requests.Session:
    if "http_session" not in st.session_state:
        st.session_state.http_session = requests.Session()
    return st.session_state.http_session


def _get_cookie_controller() -> CookieController:
    if "cookie_controller" not in st.session_state:
        st.session_state.cookie_controller = CookieController()
    return st.session_state.cookie_controller


def _refresh_cookie_controller() -> None:
    controller = _get_cookie_controller()
    try:
        controller.refresh()
    except Exception:
        # Keep app usable even if the cookie component is temporarily unavailable.
        pass


def _response_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text or f"HTTP {response.status_code}"

    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail
        message = payload.get("message")
        if isinstance(message, str) and message.strip():
            return message
    return response.text or f"HTTP {response.status_code}"


def _session_cookie_value(cookie_name: str) -> str | None:
    session = _get_http_session()
    cookie_value = session.cookies.get(cookie_name)
    if cookie_value:
        return str(cookie_value)

    for cookie in session.cookies:
        if cookie.name == cookie_name and cookie.value:
            return str(cookie.value)

    return None


def _decode_jwt_payload_unverified(token: str) -> dict[str, Any]:
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return {}

        payload = parts[1]
        padding = "=" * (-len(payload) % 4)
        decoded = base64.urlsafe_b64decode((payload + padding).encode("utf-8")).decode("utf-8")
        data = json.loads(decoded)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _set_user_state_from_token(token: str | None) -> None:
    if not token:
        st.session_state.pop("auth_token", None)
        st.session_state.pop("user_email", None)
        return

    st.session_state.auth_token = token
    claims = _decode_jwt_payload_unverified(token)
    email = str(claims.get("sub", "")).strip().lower()
    if email:
        st.session_state.user_email = email
    else:
        st.session_state.pop("user_email", None)


def _browser_cookie_value(cookie_name: str) -> str | None:
    controller = _get_cookie_controller()
    raw_value = controller.get(cookie_name)
    if raw_value is None:
        return None

    value = str(raw_value).strip()
    if value.startswith('"') and value.endswith('"') and len(value) >= 2:
        value = value[1:-1].strip()
    if value.lower().startswith("bearer "):
        value = value[7:].strip()
    return value or None


def _set_browser_cookie(cookie_name: str, cookie_value: str) -> None:
    if not cookie_name.strip() or not cookie_value.strip():
        return

    controller = _get_cookie_controller()
    controller.set(cookie_name, cookie_value)


def _clear_browser_cookie(cookie_name: str) -> None:
    if not cookie_name.strip():
        return

    controller = _get_cookie_controller()
    try:
        controller.remove(cookie_name)
    except Exception:
        pass


def _persist_cookie_to_browser(cookie_name: str) -> None:
    cookie_value = _session_cookie_value(cookie_name)
    if not cookie_value:
        return

    if _browser_cookie_value(cookie_name) == cookie_value:
        return

    _set_browser_cookie(cookie_name=cookie_name, cookie_value=cookie_value)


def _sync_cookie_from_browser(cookie_name: str) -> None:
    if _session_cookie_value(cookie_name):
        return

    browser_value = _browser_cookie_value(cookie_name)
    if not browser_value:
        return

    session = _get_http_session()
    session.cookies.set(cookie_name, browser_value)


def _sync_session_cookies_from_browser() -> None:
    _sync_cookie_from_browser(AUTH_COOKIE_NAME)
    _sync_cookie_from_browser(CHAT_SESSION_COOKIE_NAME)


def _remove_session_cookie(cookie_name: str) -> None:
    session = _get_http_session()

    try:
        session.cookies.pop(cookie_name, None)
    except Exception:
        pass

    try:
        session.cookies.set(cookie_name, "", expires=0)
    except Exception:
        pass


def restore_auth_from_cookie(base_url: str) -> tuple[bool, str]:
    _refresh_cookie_controller()
    _sync_session_cookies_from_browser()
    auth_token = _session_cookie_value(AUTH_COOKIE_NAME)
    if not auth_token:
        _set_user_state_from_token(None)
        return False, ""

    _set_user_state_from_token(auth_token)

    session = _get_http_session()
    try:
        response = session.get(f"{_api_base(base_url)}/chat/sessions", timeout=20)
    except requests.RequestException as exc:
        # Keep user logged in while backend connectivity is transiently unavailable.
        return True, f"Auth check skipped: {exc}"

    if response.ok:
        return True, ""

    if response.status_code in {401, 403}:
        _remove_session_cookie(AUTH_COOKIE_NAME)
        _remove_session_cookie(CHAT_SESSION_COOKIE_NAME)
        _clear_browser_cookie(AUTH_COOKIE_NAME)
        _clear_browser_cookie(CHAT_SESSION_COOKIE_NAME)
        _set_user_state_from_token(None)
        st.session_state.http_session = requests.Session()
        _refresh_cookie_controller()
        return False, _response_detail(response)

    # Do not force logout on non-auth backend errors (5xx, etc.).
    return True, _response_detail(response)


def _current_chat_session_id() -> str | None:
    return _session_cookie_value(CHAT_SESSION_COOKIE_NAME)


def register_user(base_url: str, username: str, email: str, password: str) -> tuple[bool, str]:
    session = _get_http_session()
    response = session.post(
        f"{_api_base(base_url)}/auth/register",
        json={"username": username, "email": email, "password": password},
        timeout=20,
    )
    if response.status_code == 201:
        _set_user_state_from_token(_session_cookie_value(AUTH_COOKIE_NAME))
        return True, "Account created"
    return False, _response_detail(response)


def login_user(base_url: str, email: str, password: str) -> tuple[bool, str]:
    session = _get_http_session()
    response = session.post(
        f"{_api_base(base_url)}/auth/login",
        data={"username": email, "password": password},
        timeout=20,
    )
    if response.ok:
        _set_user_state_from_token(_session_cookie_value(AUTH_COOKIE_NAME))
        return True, "Login successful"
    return False, _response_detail(response)


def logout_user(base_url: str) -> tuple[bool, str]:
    session = _get_http_session()
    response = session.get(f"{_api_base(base_url)}/logout", timeout=20)
    if response.ok:
        st.session_state.http_session = requests.Session()
        _set_user_state_from_token(None)
        _refresh_cookie_controller()
        return True, _response_detail(response)
    return False, _response_detail(response)


def create_new_chat_session(base_url: str) -> tuple[bool, str, str | None]:
    session = _get_http_session()
    response = session.post(f"{_api_base(base_url)}/chat/session/new", timeout=20)
    if not response.ok:
        return False, _response_detail(response), None

    try:
        payload = response.json()
    except ValueError:
        payload = {}

    session_id = str(payload.get("session_id", "")).strip() or _current_chat_session_id()
    return True, "New session created", session_id


def fetch_chat_sessions(base_url: str) -> tuple[bool, list[dict[str, Any]], str]:
    session = _get_http_session()
    response = session.get(f"{_api_base(base_url)}/chat/sessions", timeout=20)
    if not response.ok:
        return False, [], _response_detail(response)

    try:
        payload = response.json()
    except ValueError:
        return False, [], "Invalid sessions response"

    sessions = payload.get("sessions", []) if isinstance(payload, dict) else []
    if not isinstance(sessions, list):
        return False, [], "Invalid sessions payload"
    return True, sessions, ""


def fetch_chat_messages(base_url: str, session_id: str) -> tuple[bool, list[dict[str, str]], str]:
    session = _get_http_session()
    response = session.get(f"{_api_base(base_url)}/chat/session/{session_id}/messages", timeout=20)
    if not response.ok:
        return False, [], _response_detail(response)

    try:
        payload = response.json()
    except ValueError:
        return False, [], "Invalid messages response"

    messages = payload.get("messages", []) if isinstance(payload, dict) else []
    if not isinstance(messages, list):
        return False, [], "Invalid messages payload"

    clean_messages: list[dict[str, str]] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "assistant"))
        content = str(item.get("content", ""))
        clean_messages.append({"role": role, "content": content})
    return True, clean_messages, ""


def sync_sessions_from_backend(base_url: str) -> tuple[bool, str]:
    success, sessions, error_message = fetch_chat_sessions(base_url=base_url)
    if not success:
        return False, error_message

    existing = st.session_state.chat_sessions
    new_sessions: dict[str, dict[str, Any]] = {}
    new_order: list[str] = []

    for item in sessions:
        if not isinstance(item, dict):
            continue
        session_id = str(item.get("session_id", "")).strip()
        if not session_id:
            continue

        backend_title = str(item.get("title", "Chat session") or "Chat session")
        previous = existing.get(session_id, {})
        previous_title = str(previous.get("title", "")).strip()
        title = backend_title
        if previous_title and previous_title not in {"Chat session", "New Chat"} and backend_title == "Chat session":
            title = previous_title

        messages = previous.get("messages", [])
        if not isinstance(messages, list):
            messages = []

        new_sessions[session_id] = {
            "title": title,
            "messages": messages,
            "loaded": bool(previous.get("loaded", False)),
        }
        new_order.append(session_id)

    st.session_state.chat_sessions = new_sessions
    st.session_state.chat_order = new_order
    if st.session_state.active_session_id not in new_sessions:
        st.session_state.active_session_id = new_order[0] if new_order else None
    return True, ""


def stream_events(base_url: str, prompt: str, session_id: str | None) -> Iterator[dict[str, Any]]:
    """Yield parsed SSE events from the backend stream endpoint."""
    session = _get_http_session()
    payload: dict[str, Any] = {"prompt": prompt}
    if session_id and session_id.strip():
        payload["session_id"] = session_id.strip()

    with session.post(
        f"{_api_base(base_url)}/ask/stream",
        json=payload,
        stream=True,
        timeout=(5, 600),
    ) as response:
        if response.status_code == 401:
            raise PermissionError(_response_detail(response))
        response.raise_for_status()
        for raw_line in response.iter_lines(decode_unicode=True):
            if raw_line is None:
                continue

            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            line = line.strip()
            if not line or not line.startswith("data:"):
                continue

            event_json = line[len("data:") :].strip()
            if not event_json:
                continue

            try:
                yield json.loads(event_json)
            except json.JSONDecodeError:
                continue


def main() -> None:
    st.set_page_config(page_title="RAG Stream Test", layout="centered")
    st.title("RAG Stream Test")

    if "is_authenticated" not in st.session_state:
        st.session_state.is_authenticated = False
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {}
    if "chat_order" not in st.session_state:
        st.session_state.chat_order = []
    if "active_session_id" not in st.session_state:
        st.session_state.active_session_id = None

    backend_url = st.text_input("Backend URL", value=DEFAULT_BACKEND_URL)

    restored_auth, _ = restore_auth_from_cookie(base_url=backend_url)
    if restored_auth:
        st.session_state.is_authenticated = True
    elif st.session_state.is_authenticated and not _session_cookie_value(AUTH_COOKIE_NAME):
        st.session_state.is_authenticated = False

    if st.session_state.is_authenticated:
        _persist_cookie_to_browser(AUTH_COOKIE_NAME)
    if _session_cookie_value(CHAT_SESSION_COOKIE_NAME):
        _persist_cookie_to_browser(CHAT_SESSION_COOKIE_NAME)

    if not st.session_state.is_authenticated:
        login_tab, signup_tab = st.tabs(["Login", "Sign Up"])

        with login_tab:
            with st.form("login_form", clear_on_submit=False):
                login_email = st.text_input("Email", key="login_email")
                login_password = st.text_input("Password", type="password", key="login_password")
                login_submit = st.form_submit_button("Login")

            if login_submit:
                success, message = login_user(
                    base_url=backend_url,
                    email=login_email.strip().lower(),
                    password=login_password,
                )
                if success:
                    st.session_state.is_authenticated = True
                    st.session_state.chat_sessions = {}
                    st.session_state.chat_order = []
                    st.session_state.active_session_id = None
                    _persist_cookie_to_browser(AUTH_COOKIE_NAME)
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

        with signup_tab:
            with st.form("signup_form", clear_on_submit=False):
                signup_username = st.text_input("Username", key="signup_username")
                signup_email = st.text_input("Email", key="signup_email")
                signup_password = st.text_input("Password", type="password", key="signup_password")
                signup_submit = st.form_submit_button("Create Account")

            if signup_submit:
                success, message = register_user(
                    base_url=backend_url,
                    username=signup_username.strip(),
                    email=signup_email.strip().lower(),
                    password=signup_password,
                )
                if success:
                    st.session_state.is_authenticated = True
                    st.session_state.chat_sessions = {}
                    st.session_state.chat_order = []
                    st.session_state.active_session_id = None
                    _persist_cookie_to_browser(AUTH_COOKIE_NAME)
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

        st.info("Login or create an account to start chatting.")
        return

    top_col1, top_col2 = st.columns([3, 1])
    with top_col1:
        st.caption("Authenticated")
    with top_col2:
        if st.button("Logout"):
            success, message = logout_user(base_url=backend_url)
            if success:
                st.session_state.is_authenticated = False
                st.session_state.chat_sessions = {}
                st.session_state.chat_order = []
                st.session_state.active_session_id = None
                st.success(message)
                st.rerun()
            else:
                st.error(message)

    sync_ok, sync_message = sync_sessions_from_backend(base_url=backend_url)
    if not sync_ok:
        st.warning(sync_message)

    with st.sidebar:
        st.subheader("Chats")

        if st.button("New Chat", use_container_width=True):
            active_id = st.session_state.active_session_id
            can_create_new_chat = True

            if active_id:
                msg_ok, messages_list, msg_error = fetch_chat_messages(
                    base_url=backend_url,
                    session_id=active_id,
                )
                if not msg_ok:
                    st.error(msg_error)
                    can_create_new_chat = False
                else:
                    if active_id in st.session_state.chat_sessions:
                        st.session_state.chat_sessions[active_id]["messages"] = messages_list
                        st.session_state.chat_sessions[active_id]["loaded"] = True

                    if len(messages_list) == 0:
                        st.info("Current chat is empty. No new chat created.")
                        can_create_new_chat = False

            if can_create_new_chat:
                success, message, session_id = create_new_chat_session(base_url=backend_url)
                if success:
                    _persist_cookie_to_browser(CHAT_SESSION_COOKIE_NAME)
                    sync_sessions_from_backend(base_url=backend_url)
                    if session_id and session_id not in st.session_state.chat_sessions:
                        st.session_state.chat_sessions[session_id] = {
                            "title": "New Chat",
                            "messages": [],
                            "loaded": True,
                        }
                        st.session_state.chat_order.insert(0, session_id)
                    st.session_state.active_session_id = session_id
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

        if not st.session_state.chat_order:
            st.caption("No chats yet.")

        for session_id in st.session_state.chat_order:
            session_data = st.session_state.chat_sessions.get(session_id, {})
            title = str(session_data.get("title", "Untitled Chat"))
            button_label = title if len(title) <= 32 else f"{title[:29]}..."
            if st.button(button_label, key=f"session_{session_id}", use_container_width=True):
                st.session_state.active_session_id = session_id
                st.rerun()

    active_session_id = st.session_state.active_session_id
    current_messages: list[dict[str, str]] = []
    if active_session_id and active_session_id in st.session_state.chat_sessions:
        active_session = st.session_state.chat_sessions[active_session_id]
        if not bool(active_session.get("loaded", False)):
            success, messages, message = fetch_chat_messages(base_url=backend_url, session_id=active_session_id)
            if success:
                active_session["messages"] = messages
                active_session["loaded"] = True
            else:
                st.error(message)
        current_messages = active_session.get("messages", [])

    for msg in current_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Type your question...")
    if not prompt:
        return

    with st.chat_message("user"):
        st.markdown(prompt)

    answer = ""
    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            for event in stream_events(
                base_url=backend_url,
                prompt=prompt,
                session_id=active_session_id,
            ):
                event_type = str(event.get("type", ""))
                if event_type == "token":
                    answer += str(event.get("token", ""))
                    placeholder.markdown(answer)
                elif event_type == "error":
                    answer = str(event.get("error", "Unknown backend error"))
                    placeholder.error(answer)
                    break
                elif event_type == "done":
                    break
        except PermissionError as exc:
            answer = str(exc)
            st.session_state.is_authenticated = False
            placeholder.error(answer)
        except requests.RequestException as exc:
            answer = f"Request failed: {exc}"
            placeholder.error(answer)

    resolved_session_id = _current_chat_session_id() or active_session_id
    if not resolved_session_id:
        st.error("Could not resolve active session.")
        return

    if resolved_session_id not in st.session_state.chat_sessions:
        st.session_state.chat_sessions[resolved_session_id] = {
            "title": prompt[:60],
            "messages": [],
            "loaded": True,
        }
        st.session_state.chat_order.insert(0, resolved_session_id)

    session_data = st.session_state.chat_sessions[resolved_session_id]
    if not session_data["messages"] or session_data["title"] in {"New Chat", "Chat session"}:
        session_data["title"] = prompt[:60]
    session_data["loaded"] = True

    session_data["messages"].append({"role": "user", "content": prompt})
    session_data["messages"].append({"role": "assistant", "content": answer})

    if resolved_session_id in st.session_state.chat_order:
        st.session_state.chat_order.remove(resolved_session_id)
    st.session_state.chat_order.insert(0, resolved_session_id)
    st.session_state.active_session_id = resolved_session_id


if __name__ == "__main__":
    main()
