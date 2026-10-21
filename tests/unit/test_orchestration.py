from __future__ import annotations

import asyncio
from typing import Any

import pytest

from backend.rag import orchestrator

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"route": "rag"}', "rag"),
        ('{"route": "code_reviewer"}', "code_reviewer"),
        ("code", "code_reviewer"),
        ("unexpected output", "rag"),
    ],
)
def test_router_response_normalizes_or_fails_safe(raw: str, expected: str) -> None:
    assert orchestrator._parse_router_response(raw) == expected


def test_code_followup_detection_requires_code_related_hint() -> None:
    assert orchestrator._looks_like_code_followup("Can you fix the function?")
    assert not orchestrator._looks_like_code_followup("What is a histogram?")


def test_route_after_orchestrator_selects_expected_node() -> None:
    assert orchestrator.route_after_orchestrator({"route": "code_reviewer"}) == "code_reviewer_tool"
    assert orchestrator.route_after_orchestrator({"route": "rag"}) == "rag_retriever_tool"


def test_compile_orchestrator_normalizes_stop_state(monkeypatch: pytest.MonkeyPatch) -> None:
    class Graph:
        def invoke(self, _: dict[str, Any]) -> dict[str, str]:
            return {"status": "stop", "route": "rag", "message": "Please clarify."}

    monkeypatch.setattr(orchestrator, "get_orchestrator_graph", lambda: Graph())

    assert orchestrator.compile_prompt_with_orchestrator("question", "session") == {
        "status": "stop",
        "route": "rag",
        "message": "Please clarify.",
    }


def test_choose_route_uses_model_decision_and_fails_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        orchestrator,
        "get_code_reviewer_router_context",
        lambda **_: {"memory_summary": "", "last_code_excerpt": ""},
    )

    class Router:
        def __init__(self, **_: Any) -> None:
            pass

        def invoke(self, _: str) -> Any:
            return type("Response", (), {"content": '{"route":"code_reviewer"}'})()

    monkeypatch.setattr(orchestrator, "create_chat_client", lambda **_: Router())
    assert orchestrator.choose_orchestration_route("Fix this function", session_id="route-test") == (
        "code_reviewer"
    )

    class BrokenRouter(Router):
        def invoke(self, _: str) -> Any:
            raise RuntimeError("model unavailable")

    monkeypatch.setattr(orchestrator, "create_chat_client", lambda **_: BrokenRouter())
    assert orchestrator.choose_orchestration_route("What is a mean?", session_id="route-error") == "rag"


def test_code_followup_keeps_previous_reviewer_route(monkeypatch: pytest.MonkeyPatch) -> None:
    session_id = "route-followup"
    orchestrator._SESSION_LAST_ROUTE[session_id] = "code_reviewer"
    monkeypatch.setattr(
        orchestrator,
        "get_code_reviewer_router_context",
        lambda **_: {"memory_summary": "", "last_code_excerpt": ""},
    )

    class Router:
        def __init__(self, **_: Any) -> None:
            pass

        def invoke(self, _: str) -> Any:
            return type("Response", (), {"content": '{"route":"rag"}'})()

    monkeypatch.setattr(orchestrator, "create_chat_client", lambda **_: Router())

    assert orchestrator.choose_orchestration_route(
        "Can you change the function?",
        session_id=session_id,
    ) == "code_reviewer"


def test_orchestrator_node_records_selected_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(orchestrator, "choose_orchestration_route", lambda **_: "code_reviewer")

    assert orchestrator.orchestrator_node({"question": "fix code"}) == {
        "status": "ok",
        "route": "code_reviewer",
    }


def test_code_reviewer_node_returns_tool_result_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        orchestrator,
        "run_code_reviewer_tool_from_query",
        lambda **_: {"status": "ok", "answer": "Reviewed code."},
    )
    assert orchestrator.code_reviewer_tool_node({"question": "review", "session_id": "s"}) == {
        "status": "ok",
        "route": "code_reviewer",
        "answer": "Reviewed code.",
    }

    monkeypatch.setattr(
        orchestrator,
        "run_code_reviewer_tool_from_query",
        lambda **_: {"status": "stop", "message": "No code found."},
    )
    assert orchestrator.code_reviewer_tool_node({"question": "review"})["message"] == "No code found."


def test_rag_node_stops_when_guard_rejects_question(monkeypatch: pytest.MonkeyPatch) -> None:
    class Chat:
        def __init__(self, **_: Any) -> None:
            pass

    monkeypatch.setattr(orchestrator, "create_chat_client", lambda **_: Chat())
    monkeypatch.setattr(
        orchestrator,
        "guard_question",
        lambda **_: {"status": "stop", "message": "Needs clarification."},
    )

    result = orchestrator.rag_retriever_tool_node(
        {"question": "broad question", "session_id": "guarded", "persist_dir": None}
    )

    assert result == {"status": "stop", "route": "rag", "message": "Needs clarification."}


def test_rag_node_passes_retrieved_context_to_graph_state(monkeypatch: pytest.MonkeyPatch) -> None:
    class Chat:
        def __init__(self, **_: Any) -> None:
            pass

    class RetrievalTool:
        pass

    monkeypatch.setattr(orchestrator, "create_chat_client", lambda **_: Chat())
    monkeypatch.setattr(orchestrator, "guard_question", lambda **_: {"status": "ok"})
    monkeypatch.setattr(
        orchestrator,
        "prepare_memory_summary",
        lambda **_: (object(), "Earlier session summary"),
    )
    monkeypatch.setattr(orchestrator, "CrossEncoder", lambda _: "reranker")
    monkeypatch.setattr(
        orchestrator,
        "get_collections",
        lambda **_: {
            "course_collection": "courses",
            "part_collection": "parts",
            "chunk_collection": "chunks",
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "build_retrieval_tool",
        lambda **kwargs: (kwargs, RetrievalTool())[1],
    )
    monkeypatch.setattr(
        orchestrator,
        "run_retrieval_tool_node",
        lambda **_: {"status": "ok", "context": "course context"},
    )

    result = orchestrator.rag_retriever_tool_node(
        {"question": "mean", "session_id": "session", "persist_dir": None}
    )

    assert result["status"] == "ok"
    assert result["route"] == "rag"
    assert result["context"] == "course context"
    assert result["memory_summary"] == "Earlier session summary"


def test_rag_node_stops_when_retrieval_has_no_relevant_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Chat:
        def __init__(self, **_: Any) -> None:
            pass

    monkeypatch.setattr(orchestrator, "create_chat_client", lambda **_: Chat())
    monkeypatch.setattr(orchestrator, "guard_question", lambda **_: {"status": "ok"})
    monkeypatch.setattr(orchestrator, "prepare_memory_summary", lambda **_: (object(), ""))
    monkeypatch.setattr(orchestrator, "CrossEncoder", lambda _: "reranker")
    monkeypatch.setattr(
        orchestrator,
        "get_collections",
        lambda **_: {
            "course_collection": "courses",
            "part_collection": "parts",
            "chunk_collection": "chunks",
        },
    )
    monkeypatch.setattr(orchestrator, "build_retrieval_tool", lambda **_: object())
    monkeypatch.setattr(
        orchestrator,
        "run_retrieval_tool_node",
        lambda **_: {"status": "stop", "message": "No supporting chunks."},
    )

    result = orchestrator.rag_retriever_tool_node(
        {"question": "mean", "session_id": "empty-results", "persist_dir": None}
    )

    assert result == {
        "status": "stop",
        "route": "rag",
        "message": "No supporting chunks.",
    }


def test_built_graph_routes_to_code_reviewer_node(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(orchestrator, "choose_orchestration_route", lambda **_: "code_reviewer")
    monkeypatch.setattr(
        orchestrator,
        "run_code_reviewer_tool_from_query",
        lambda **_: {"status": "ok", "answer": "Review complete."},
    )

    result = orchestrator.build_orchestrator_graph().invoke(
        {"question": "fix code", "session_id": "graph"}
    )

    assert result["route"] == "code_reviewer"
    assert result["answer"] == "Review complete."


def test_compile_orchestrator_normalizes_both_success_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    class Graph:
        def __init__(self, result: dict[str, Any]) -> None:
            self.result = result

        def invoke(self, _: dict[str, Any]) -> dict[str, Any]:
            return self.result

    monkeypatch.setattr(
        orchestrator,
        "get_orchestrator_graph",
        lambda: Graph({"status": "ok", "route": "code_reviewer", "answer": "Review."}),
    )
    assert orchestrator.compile_prompt_with_orchestrator("review", "code-session") == {
        "status": "ok",
        "route": "code_reviewer",
        "answer": "Review.",
    }

    memory = object()
    chat = object()
    monkeypatch.setattr(
        orchestrator,
        "get_orchestrator_graph",
        lambda: Graph(
            {
                "status": "ok",
                "route": "rag",
                "chat_llm": chat,
                "memory_summary": "summary",
                "context": "context",
                "conversation_memory": memory,
            }
        ),
    )
    compiled = orchestrator.compile_prompt_with_orchestrator("question", "rag-session")
    assert compiled == {
        "status": "ok",
        "route": "rag",
        "chat_llm": chat,
        "memory_summary": "summary",
        "context": "context",
        "conversation_memory": memory,
    }


def test_stream_orchestrated_response_handles_code_and_guard_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def collect() -> tuple[list[str], list[str]]:
        monkeypatch.setattr(
            orchestrator,
            "compile_prompt_with_orchestrator",
            lambda **_: {"status": "ok", "route": "code_reviewer", "answer": "Review complete."},
        )
        code_tokens = [
            token
            async for token in orchestrator.stream_orchestrated_response(
                "review", session_id="code-session"
            )
        ]
        monkeypatch.setattr(
            orchestrator,
            "compile_prompt_with_orchestrator",
            lambda **_: {"status": "stop", "message": "Please clarify."},
        )
        stop_tokens = [
            token
            async for token in orchestrator.stream_orchestrated_response(
                "unclear", session_id="stop-session"
            )
        ]
        return code_tokens, stop_tokens

    code_tokens, stop_tokens = asyncio.run(collect())

    assert code_tokens == ["Review complete."]
    assert stop_tokens == ["Please clarify."]


def test_stream_orchestrated_response_generates_and_saves_rag_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved_turns: list[dict[str, str]] = []

    class GenerationChain:
        async def astream(self, payload: dict[str, str]):
            assert payload["context"] == "retrieved context"
            yield "Grounded "
            yield "answer."

    monkeypatch.setattr(
        orchestrator,
        "compile_prompt_with_orchestrator",
        lambda **_: {
            "status": "ok",
            "route": "rag",
            "chat_llm": object(),
            "memory_summary": "summary",
            "context": "retrieved context",
            "conversation_memory": "memory",
        },
    )
    monkeypatch.setattr(orchestrator, "build_generation_chain", lambda _: GenerationChain())
    monkeypatch.setattr(
        orchestrator,
        "save_memory_turn",
        lambda **kwargs: saved_turns.append(kwargs),
    )

    async def collect() -> list[str]:
        return [
            token
            async for token in orchestrator.stream_orchestrated_response(
                "question",
                session_id="stream-session",
            )
        ]

    assert asyncio.run(collect()) == ["Grounded ", "answer."]
    assert saved_turns[0]["answer"] == "Grounded answer."
    assert saved_turns[0]["session_id"] == "stream-session"
