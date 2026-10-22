from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
import pytest

from backend.rag import rag_agent, retriever

pytestmark = pytest.mark.integration


class MemoryCollection:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents

    def query(self, **kwargs: Any) -> dict[str, Any]:
        where = kwargs.get("where")
        documents = [
            document
            for document in self.documents
            if where is None or all(document.metadata.get(key) == value for key, value in where.items())
        ]
        return {
            "documents": [[document.page_content for document in documents[: kwargs["n_results"]]]],
            "metadatas": [[document.metadata for document in documents[: kwargs["n_results"]]]],
        }

    def get(self, **_: Any) -> dict[str, Any]:
        return {
            "documents": [document.page_content for document in self.documents],
            "metadatas": [document.metadata for document in self.documents],
        }


class MeanFirstReranker:
    def predict(self, pairs: list[list[str]]) -> list[int]:
        return [2 if "mean" in text.lower() else 0 for _, text in pairs]


def test_retrieval_reranking_and_generation_keep_relevant_source_grounded() -> None:
    relevant = Document(
        page_content="The mean is the sum of values divided by the number of values.",
        metadata={
            "course": "statistics",
            "section": "descriptive statistics",
            "source_file": "statistics.pdf",
            "chunk_id": "mean-1",
        },
    )
    irrelevant = Document(
        page_content="A decision tree splits data using feature conditions.",
        metadata={
            "course": "statistics",
            "section": "descriptive statistics",
            "source_file": "statistics.pdf",
            "chunk_id": "tree-1",
        },
    )
    other_course = Document(
        page_content="The mean of this other course is unrelated.",
        metadata={
            "course": "python",
            "section": "basics",
            "source_file": "python.pdf",
            "chunk_id": "other-1",
        },
    )
    context = retriever.retrieve_context(
        "How is the mean calculated?",
        MemoryCollection([]),
        MemoryCollection(
            [
                Document(page_content="descriptive statistics", metadata={"course": "statistics"}),
                Document(page_content="basics", metadata={"course": "python"}),
            ]
        ),
        MemoryCollection([irrelevant, relevant, other_course]),
        MeanFirstReranker(),
        chat_llm=object(),
        top_k_final=1,
    )

    captured_prompt: dict[str, str] = {}

    def inspect_prompt(prompt: Any) -> str:
        captured_prompt["text"] = prompt.to_messages()[0].content
        return "The mean is the sum of values divided by their count. [1]"

    answer = rag_agent.build_generation_chain(RunnableLambda(inspect_prompt)).invoke(
        {
            "memory_summary": "No prior conversation summary.",
            "context": context["context"],
            "question": "How is the mean calculated?",
        }
    )

    assert context["status"] == "ok"
    assert context["selected_course"] == "statistics"
    assert [document.metadata["chunk_id"] for document in context["ranked_docs"]] == ["mean-1"]
    assert "source=statistics.pdf" in context["context"]
    assert "The mean is the sum of values" in captured_prompt["text"]
    assert "ONLY the provided course context" in captured_prompt["text"]
    assert "Format mathematical formulas and equations as LaTeX math" in captured_prompt["text"]
    assert "Put every code example in a fenced Markdown code block" in captured_prompt["text"]
    assert "Do not put programming code, API calls, or code arguments in math delimiters" in captured_prompt["text"]
    assert answer.endswith("[1]")
