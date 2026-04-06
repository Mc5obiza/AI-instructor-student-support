from __future__ import annotations

import json
from typing import Any, Iterator

import requests
import streamlit as st


DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"


def _api_base(url: str) -> str:
    return url.rstrip("/")


def _get_http_session() -> requests.Session:
    if "http_session" not in st.session_state:
        st.session_state.http_session = requests.Session()
    return st.session_state.http_session


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


def _current_chat_session_id() -> str | None:
    session = _get_http_session()
    cookie_value = session.cookies.get("chat_session_id")
    if cookie_value:
        return str(cookie_value)

    for cookie in session.cookies:
        if cookie.name == "chat_session_id" and cookie.value:
            return str(cookie.value)

    return None


def register_user(base_url: str, username: str, email: str, password: str) -> tuple[bool, str]:
    session = _get_http_session()
    response = session.post(
        f"{_api_base(base_url)}/auth/register",
        json={"username": username, "email": email, "password": password},
        timeout=20,
    )
    if response.status_code == 201:
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
        return True, "Login successful"
    return False, _response_detail(response)


def logout_user(base_url: str) -> tuple[bool, str]:
    session = _get_http_session()
    response = session.post(f"{_api_base(base_url)}/auth/logout", timeout=20)
    if response.ok:
        st.session_state.http_session = requests.Session()
        return True, "Logout successful"
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
