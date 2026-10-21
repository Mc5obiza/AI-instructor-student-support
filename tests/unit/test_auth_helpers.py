from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException
from jose import jwt

from backend import helper

pytestmark = pytest.mark.unit


def test_session_title_fallback_uses_first_words_and_handles_blank() -> None:
    assert helper._fallback_session_title("  Explain   linear regression with examples  ") == (
        "Explain linear regression with examples"
    )
    assert helper._fallback_session_title(" \n ") == "New Chat"


def test_session_title_sanitization_removes_prefix_and_bounds_length() -> None:
    cleaned = helper._sanitize_session_title('  Title: "A useful title"  ', "fallback")

    assert cleaned == "A useful title"
    assert len(helper._sanitize_session_title("X" * 200, "fallback")) <= helper.SESSION_TITLE_MAX_CHARS


def test_access_token_round_trip_resolves_user_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(helper, "_get_user_id_from_interface", lambda email: "user-42")
    token = helper._create_access_token("user-42", "Student@Example.test")

    user = helper.get_current_user(token)

    assert user.user_id == "user-42"
    assert user.email == "student@example.test"


@pytest.mark.parametrize("token", [None, "not-a-token"])
def test_current_user_rejects_missing_or_invalid_token(token: str | None) -> None:
    with pytest.raises(HTTPException) as error:
        helper.get_current_user(token)

    assert error.value.status_code == 401


def test_session_resolution_prefers_request_id_then_cookie(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        helper.SessionInterface,
        "create_session",
        lambda *_args, **_kwargs: pytest.fail("A new session should not be created."),
    )

    assert helper._resolve_or_create_session_id("question", " request-id ", "cookie-id", "user") == "request-id"
    assert helper._resolve_or_create_session_id("question", None, " cookie-id ", "user") == "cookie-id"


def test_register_user_returns_user_id_and_maps_duplicate_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class UserStore:
        def add_user(self, **_: str) -> str:
            return json.dumps({"status": "202"})

        def get_user_id(self, **_: str) -> str:
            return json.dumps({"status": "200", "user_id": "user-1"})

    monkeypatch.setattr(helper, "UserInterface", UserStore)
    assert helper._register_user_or_raise("student", "student@example.test", "password") == "user-1"

    class DuplicateStore(UserStore):
        def add_user(self, **_: str) -> str:
            return json.dumps({"status": "400"})

    monkeypatch.setattr(helper, "UserInterface", DuplicateStore)
    with pytest.raises(HTTPException) as error:
        helper._register_user_or_raise("student", "student@example.test", "password")
    assert error.value.status_code == 409


def test_login_maps_bad_credentials_and_resolves_authenticated_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UserStore:
        def verify_user(self, **_: str) -> str:
            return json.dumps({"status": "200"})

        def get_user_id(self, **_: str) -> str:
            return json.dumps({"status": "200", "user_id": "user-2"})

    monkeypatch.setattr(helper, "UserInterface", UserStore)
    assert helper._verify_login_or_raise("student@example.test", "password") == "user-2"

    class InvalidUserStore(UserStore):
        def verify_user(self, **_: str) -> str:
            return json.dumps({"status": "400"})

    monkeypatch.setattr(helper, "UserInterface", InvalidUserStore)
    with pytest.raises(HTTPException) as error:
        helper._verify_login_or_raise("student@example.test", "wrong")
    assert error.value.status_code == 401


def test_session_and_message_helpers_normalize_success_responses(monkeypatch: pytest.MonkeyPatch) -> None:
    class SessionStore:
        def get_session(self, **_: str) -> tuple[str, list[dict[str, str]]]:
            return (
                json.dumps({"status": "200"}),
                [{"session_id": " session-1 ", "title": "Review"}, {"title": "invalid"}],
            )

        def get_messages(self, **_: str) -> str:
            return json.dumps({"status": "200", "messages": [{"content": "hi"}, None]})

    monkeypatch.setattr(helper, "SessionInterface", SessionStore)

    assert helper._get_sessions_or_raise("user") == [
        {"session_id": "session-1", "title": "Review", "updated_at": None}
    ]
    assert helper._get_messages_or_raise("session-1") == [{"content": "hi"}]


def test_session_and_message_helpers_raise_on_storage_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingSessionStore:
        def get_session(self, **_: str) -> tuple[str, None]:
            return json.dumps({"status": "500", "message": "database unavailable"}), None

        def get_messages(self, **_: str) -> str:
            return json.dumps({"status": "500", "message": "database unavailable"})

    class FailingMessageStore:
        def send_message(self, **_: str) -> str:
            return json.dumps({"status": "500", "error": "write failed"})

    monkeypatch.setattr(helper, "SessionInterface", FailingSessionStore)
    monkeypatch.setattr(helper, "MessageInterface", FailingMessageStore)

    with pytest.raises(HTTPException) as sessions_error:
        helper._get_sessions_or_raise("user")
    with pytest.raises(HTTPException) as messages_error:
        helper._get_messages_or_raise("session")
    with pytest.raises(HTTPException) as store_error:
        helper._store_message_or_raise("session", "hello", "user")

    assert sessions_error.value.status_code == 500
    assert messages_error.value.status_code == 500
    assert store_error.value.detail == "write failed"


def test_message_count_and_title_update_handle_missing_storage_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenSessionStore:
        def get_messages(self, **_: str) -> str:
            return "{"

    monkeypatch.setattr(helper, "SessionInterface", BrokenSessionStore)
    assert helper._get_session_message_count("session") is None

    monkeypatch.setattr(helper, "_get_sessions_or_raise", lambda **_: [])
    assert helper._set_session_title_after_response("session", "user", "question", "answer", 0) is None


def test_user_id_lookup_maps_missing_users_to_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class UserStore:
        def get_user_id(self, **_: str) -> str:
            return json.dumps({"status": "200", "user_id": "user-3"})

    monkeypatch.setattr(helper, "UserInterface", UserStore)
    assert helper._get_user_id_from_interface("student@example.com") == "user-3"

    class MissingUserStore(UserStore):
        def get_user_id(self, **_: str) -> str:
            return json.dumps({"status": "404"})

    monkeypatch.setattr(helper, "UserInterface", MissingUserStore)
    with pytest.raises(HTTPException) as error:
        helper._get_user_id_from_interface("missing@example.com")
    assert error.value.status_code == 401


def test_session_creation_handles_success_and_invalid_storage_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SessionStore:
        def create_session(self, **_: str) -> str:
            return json.dumps({"status": "202", "session_id": "new-session"})

    monkeypatch.setattr(helper, "SessionInterface", SessionStore)
    assert helper._create_session_or_raise("user") == "new-session"
    assert helper._resolve_or_create_session_id("question", None, None, "user") == "new-session"

    class InvalidSessionStore(SessionStore):
        def create_session(self, **_: str) -> str:
            return json.dumps({"status": "500", "message": "database unavailable"})

    monkeypatch.setattr(helper, "SessionInterface", InvalidSessionStore)
    with pytest.raises(HTTPException) as error:
        helper._create_session_or_raise("user")
    assert error.value.status_code == 500


def test_missing_sessions_and_malformed_message_shapes_are_normalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SessionStore:
        def get_session(self, **_: str) -> tuple[str, None]:
            return json.dumps({"status": "404"}), None

        def get_messages(self, **_: str) -> str:
            return json.dumps({"status": "200", "messages": "not-a-list"})

    monkeypatch.setattr(helper, "SessionInterface", SessionStore)

    assert helper._get_sessions_or_raise("user") == []
    assert helper._get_messages_or_raise("session") == []


def test_message_store_success_and_access_token_with_missing_email(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MessageStore:
        def send_message(self, **_: str) -> str:
            return json.dumps({"status": "200"})

    monkeypatch.setattr(helper, "MessageInterface", MessageStore)
    helper._store_message_or_raise("session", "hello", "user")

    token = jwt.encode(
        {"sub": "", "uid": "user", "exp": datetime.now(timezone.utc) + timedelta(minutes=1)},
        helper.SECRET_KEY,
        algorithm=helper.ALGORITHM,
    )
    with pytest.raises(HTTPException) as error:
        helper.get_current_user(token)
    assert error.value.status_code == 401


def test_first_turn_title_update_uses_sanitized_suggestion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SessionStore:
        def set_title(self, **kwargs: str) -> str:
            assert kwargs["title"] == "A session title"
            return json.dumps({"status": "200"})

    monkeypatch.setattr(helper, "SessionInterface", SessionStore)
    monkeypatch.setattr(
        helper,
        "_get_session_title_for_user",
        lambda **_: "New Chat",
    )
    monkeypatch.setattr(helper, "_suggest_session_title", lambda **_: "A session title")

    assert helper._set_session_title_after_response(
        "session",
        "user",
        "question",
        "answer",
        message_count_before_user_message=0,
    ) == "A session title"


def test_title_model_fallback_and_logout_cookie_clearing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi import Response

    monkeypatch.setattr(helper, "ChatOllama", None)
    assert helper._suggest_session_title("Explain a mean", "Averages values.") == "Explain a mean"

    response = Response()
    helper._set_auth_cookie(response, "token")
    helper._set_chat_session_cookie(response, "session")
    helper._clear_logout_cookies(response)
    cookies = [
        value.decode("latin-1")
        for name, value in response.raw_headers
        if name.lower() == b"set-cookie"
    ]

    assert any(
        "access_token=" in cookie.lower() and "max-age=0" in cookie.lower()
        for cookie in cookies
    )
    assert any(
        "chat_session_id=" in cookie.lower() and "max-age=0" in cookie.lower()
        for cookie in cookies
    )
