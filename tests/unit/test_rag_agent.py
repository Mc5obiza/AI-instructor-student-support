from __future__ import annotations

from typing import Any

import pytest

from backend.rag import rag_agent

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("response", "status"),
    [
        ('{"status":"ok"}', "ok"),
        ('{"status":"out_of_scope"}', "stop"),
        ('{"status":"needs_clarification","message":"Name a topic."}', "stop"),
        ("malformed model response", "stop"),
    ],
)
def test_guard_parser_fails_closed_on_invalid_or_rejected_output(
    response: str,
    status: str,
) -> None:
    assert rag_agent._parse_guard_response(response)["status"] == status


def test_guard_question_returns_original_question_for_accepted_response() -> None:
    class Guard:
        def invoke(self, _: str) -> Any:
            return type("Response", (), {"content": '{"status":"ok"}'})()

    assert rag_agent.guard_question("What is a mean?", Guard()) == {
        "status": "ok",
        "question": "What is a mean?",
    }


def test_guard_question_returns_rejection_message() -> None:
    class Guard:
        def invoke(self, _: str) -> Any:
            return type("Response", (), {"content": '{"status":"needs_clarification","message":"Which topic?"}'})()

    assert rag_agent.guard_question("Explain everything", Guard()) == {
        "status": "stop",
        "message": "Which topic?",
    }
