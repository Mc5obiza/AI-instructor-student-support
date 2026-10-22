from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient
import pytest

from backend import main

pytestmark = pytest.mark.feature


def test_health_endpoint_is_public(api_client: TestClient) -> None:
    response = api_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_protected_chat_endpoint_rejects_anonymous_request(api_client: TestClient) -> None:
    response = api_client.get("/chat/sessions")

    assert response.status_code == 401


def test_registration_validates_email_before_database_access(api_client: TestClient) -> None:
    response = api_client.post(
        "/auth/register",
        json={"username": "student", "email": "not-an-email", "password": "valid-pass"},
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("username", "password"),
    [("s", "valid-pass"), ("student", "short")],
)
def test_registration_rejects_short_credentials_before_database_access(
    api_client: TestClient,
    monkeypatch: Any,
    username: str,
    password: str,
) -> None:
    monkeypatch.setattr(
        main,
        "_register_user_or_raise",
        lambda **_: pytest.fail("Invalid registration data must not access the database."),
    )

    response = api_client.post(
        "/auth/register",
        json={"username": username, "email": "student@example.test", "password": password},
    )

    assert response.status_code == 422


def test_user_cannot_read_another_session(
    authenticated_client: TestClient,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(main, "_get_sessions_or_raise", lambda **_: [{"session_id": "owned-session"}])
    monkeypatch.setattr(
        main,
        "_get_messages_or_raise",
        lambda **_: (_ for _ in ()).throw(AssertionError("Unowned session must not be queried.")),
    )

    response = authenticated_client.get("/chat/session/other-session/messages")

    assert response.status_code == 404


def test_user_cannot_delete_another_users_session(
    authenticated_client: TestClient,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(main, "_get_sessions_or_raise", lambda **_: [{"session_id": "owned-session"}])
    monkeypatch.setattr(
        main,
        "_delete_session_or_raise",
        lambda **_: pytest.fail("Unowned sessions must not be deleted."),
    )

    response = authenticated_client.delete("/chat/session/other-session")

    assert response.status_code == 404


def test_user_can_delete_owned_session(
    authenticated_client: TestClient,
    monkeypatch: Any,
) -> None:
    deleted_sessions: list[str] = []
    monkeypatch.setattr(main, "_get_sessions_or_raise", lambda **_: [{"session_id": "owned-session"}])
    monkeypatch.setattr(
        main,
        "_delete_session_or_raise",
        lambda *, session_id: deleted_sessions.append(session_id),
    )
    authenticated_client.cookies.set(main.CHAT_SESSION_COOKIE_NAME, "owned-session")

    response = authenticated_client.delete("/chat/session/owned-session")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Session deleted",
        "session_id": "owned-session",
    }
    assert deleted_sessions == ["owned-session"]
    assert f"{main.CHAT_SESSION_COOKIE_NAME}=" in response.headers["set-cookie"]
    assert "max-age=0" in response.headers["set-cookie"].lower()


def test_chat_endpoint_streams_and_persists_response(
    authenticated_client: TestClient,
    monkeypatch: Any,
) -> None:
    stored_messages: list[tuple[str, str]] = []
    observed_inputs: list[dict[str, str]] = []

    class GenerationChain:
        async def astream(self, payload: dict[str, str]):
            observed_inputs.append(payload)
            yield "Mean is "
            yield "the average. [1]"

    monkeypatch.setattr(main, "_resolve_or_create_session_id", lambda **_: "session-1")
    monkeypatch.setattr(main, "_get_session_message_count", lambda **_: 0)
    monkeypatch.setattr(
        main,
        "_store_message_or_raise",
        lambda *, content, role, **_: stored_messages.append((role, content)),
    )
    monkeypatch.setattr(main, "_set_session_title_after_response", lambda **_: "Mean")
    monkeypatch.setattr(
        main,
        "compile_prompt_with_orchestrator",
        lambda **_: {
            "status": "ok",
            "route": "rag",
            "chat_llm": object(),
            "memory_summary": "No prior conversation summary.",
            "context": "[1] source=statistics.pdf\nMean is the average.",
            "conversation_memory": None,
        },
    )
    monkeypatch.setattr(main, "build_generation_chain", lambda _: GenerationChain())

    response = authenticated_client.post("/ask/stream", json={"prompt": "What is the mean?"})
    events = [
        json.loads(event.removeprefix("data: "))
        for event in response.text.strip().split("\n\n")
        if event.startswith("data: ")
    ]

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert [event["token"] for event in events if event["type"] == "token"] == [
        "Mean is ",
        "the average. [1]",
    ]
    assert events[-1] == {"type": "done", "session_title": "Mean"}
    assert observed_inputs[0]["context"].startswith("[1] source=statistics.pdf")
    assert stored_messages == [
        ("user", "What is the mean?"),
        ("assistant", "Mean is the average. [1]"),
    ]


def test_authentication_endpoints_set_and_clear_authentication_cookie(
    api_client: TestClient,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(main, "_register_user_or_raise", lambda **_: "registered-user")
    monkeypatch.setattr(main, "_verify_login_or_raise", lambda **_: "registered-user")
    monkeypatch.setattr(main, "_create_access_token", lambda **_: "signed-test-token")

    registered = api_client.post(
        "/auth/register",
        json={
            "username": " Student ",
            "email": "STUDENT@example.com",
            "password": "password-123",
        },
    )
    logged_in = api_client.post(
        "/auth/login",
        data={"username": "student@example.com", "password": "password-123"},
    )
    logged_out = api_client.post("/auth/logout")

    assert registered.status_code == 201
    assert registered.json()["user_id"] == "registered-user"
    assert "httponly" in registered.headers["set-cookie"].lower()
    assert logged_in.status_code == 200
    assert "signed-test-token" in logged_in.headers["set-cookie"]
    assert logged_out.status_code == 200
    assert "max-age=0" in logged_out.headers["set-cookie"].lower()


def test_session_routes_return_only_the_current_users_sessions(
    authenticated_client: TestClient,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(main, "_create_session_or_raise", lambda **_: "new-session")
    monkeypatch.setattr(
        main,
        "_get_sessions_or_raise",
        lambda **_: [{"session_id": "owned-session", "title": "Statistics"}],
    )
    monkeypatch.setattr(
        main,
        "_get_messages_or_raise",
        lambda **_: [{"content": "hello", "role": "user"}],
    )

    created = authenticated_client.post("/chat/session/new")
    sessions = authenticated_client.get("/chat/sessions")
    messages = authenticated_client.get("/chat/session/owned-session/messages")

    assert created.status_code == 200
    assert created.json()["session_id"] == "new-session"
    assert sessions.json()["sessions"] == [{"session_id": "owned-session", "title": "Statistics"}]
    assert messages.json()["messages"] == [{"content": "hello", "role": "user"}]


def test_stream_endpoint_handles_guard_stop_without_calling_generation(
    authenticated_client: TestClient,
    monkeypatch: Any,
) -> None:
    saved_messages: list[tuple[str, str]] = []
    monkeypatch.setattr(main, "_resolve_or_create_session_id", lambda **_: "session-guarded")
    monkeypatch.setattr(main, "_get_session_message_count", lambda **_: 0)
    monkeypatch.setattr(
        main,
        "_store_message_or_raise",
        lambda *, content, role, **_: saved_messages.append((role, content)),
    )
    monkeypatch.setattr(main, "_set_session_title_after_response", lambda **_: None)
    monkeypatch.setattr(
        main,
        "compile_prompt_with_orchestrator",
        lambda **_: {"status": "stop", "message": "Please narrow the question."},
    )
    monkeypatch.setattr(
        main,
        "build_generation_chain",
        lambda *_: (_ for _ in ()).throw(AssertionError("Stopped requests must not generate.")),
    )

    response = authenticated_client.post("/ask/stream", json={"prompt": "everything"})
    events = [
        json.loads(event.removeprefix("data: "))
        for event in response.text.strip().split("\n\n")
        if event.startswith("data: ")
    ]

    assert response.status_code == 200
    assert events[0] == {"type": "token", "token": "Please narrow the question."}
    assert events[-1]["type"] == "done"
    assert saved_messages[-1] == ("assistant", "Please narrow the question.")
