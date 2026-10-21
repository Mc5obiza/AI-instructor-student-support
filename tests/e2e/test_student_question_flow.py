from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient
from langchain_core.documents import Document
import pytest

from backend import main
from backend.rag.retriever import retrieve_context

pytestmark = pytest.mark.e2e


class InMemoryCourseCollection:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents

    def query(self, **kwargs: Any) -> dict[str, Any]:
        where = kwargs.get("where")
        matches = [
            document
            for document in self.documents
            if where is None or all(document.metadata.get(key) == value for key, value in where.items())
        ]
        return {
            "documents": [[document.page_content for document in matches[: kwargs["n_results"]]]],
            "metadatas": [[document.metadata for document in matches[: kwargs["n_results"]]]],
        }

    def get(self, **_: Any) -> dict[str, Any]:
        return {
            "documents": [document.page_content for document in self.documents],
            "metadatas": [document.metadata for document in self.documents],
        }


class DeterministicReranker:
    def predict(self, pairs: list[list[str]]) -> list[int]:
        return [1 if "standard deviation" in content.lower() else 0 for _, content in pairs]


def test_student_question_retrieves_course_source_and_streams_answer(
    authenticated_client: TestClient,
    monkeypatch: Any,
) -> None:
    course_chunks = InMemoryCourseCollection(
        [
            Document(
                page_content="Standard deviation measures the spread of data around the mean.",
                metadata={
                    "course": "statistics",
                    "section": "dispersion",
                    "source_file": "statistics.pdf",
                    "chunk_id": "std-1",
                },
            ),
            Document(
                page_content="A decision tree makes predictions through feature splits.",
                metadata={
                    "course": "statistics",
                    "section": "trees",
                    "source_file": "statistics.pdf",
                    "chunk_id": "tree-1",
                },
            ),
        ]
    )
    retrieval = retrieve_context(
        "What does standard deviation measure?",
        InMemoryCourseCollection([]),
        InMemoryCourseCollection(
            [Document(page_content="dispersion", metadata={"course": "statistics"})]
        ),
        course_chunks,
        DeterministicReranker(),
        chat_llm=object(),
        top_k_final=1,
    )
    assert retrieval["status"] == "ok"

    class GenerationChain:
        async def astream(self, payload: dict[str, str]):
            assert "source=statistics.pdf" in payload["context"]
            assert "Standard deviation measures the spread" in payload["context"]
            yield "It measures how spread out data is around the mean. [1]"

    messages: list[tuple[str, str]] = []
    monkeypatch.setattr(main, "_resolve_or_create_session_id", lambda **_: "e2e-session")
    monkeypatch.setattr(main, "_get_session_message_count", lambda **_: 0)
    monkeypatch.setattr(
        main,
        "_store_message_or_raise",
        lambda *, content, role, **_: messages.append((role, content)),
    )
    monkeypatch.setattr(main, "_set_session_title_after_response", lambda **_: None)
    monkeypatch.setattr(
        main,
        "compile_prompt_with_orchestrator",
        lambda **_: {
            "status": "ok",
            "route": "rag",
            "chat_llm": object(),
            "memory_summary": "No prior conversation summary.",
            "context": retrieval["context"],
            "conversation_memory": None,
        },
    )
    monkeypatch.setattr(main, "build_generation_chain", lambda _: GenerationChain())

    response = authenticated_client.post(
        "/ask/stream",
        json={"prompt": "What does standard deviation measure?"},
    )
    events = [
        json.loads(event.removeprefix("data: "))
        for event in response.text.strip().split("\n\n")
        if event.startswith("data: ")
    ]

    assert response.status_code == 200
    assert any(event["type"] == "token" and "[1]" in event["token"] for event in events)
    assert events[-1]["type"] == "done"
    assert messages == [
        ("user", "What does standard deviation measure?"),
        ("assistant", "It measures how spread out data is around the mean. [1]"),
    ]
