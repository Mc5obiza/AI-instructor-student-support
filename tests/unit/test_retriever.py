from __future__ import annotations

from typing import Any

import pytest
from langchain_core.documents import Document

from backend.rag import retriever

pytestmark = pytest.mark.unit


def make_document(text: str, *, course: str = "statistics", section: str = "metrics", index: int = 1) -> Document:
    return Document(
        page_content=text,
        metadata={
            "course": course,
            "section": section,
            "source_file": f"lesson-{index}.pdf",
            "chunk_id": f"chunk-{index}",
        },
    )


class FakeCollection:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.query_calls: list[dict[str, Any]] = []

    def query(self, **kwargs: Any) -> dict[str, Any]:
        self.query_calls.append(kwargs)
        documents = self._filter(kwargs.get("where"))
        return {
            "documents": [[document.page_content for document in documents[: kwargs["n_results"]]]],
            "metadatas": [[document.metadata for document in documents[: kwargs["n_results"]]]],
        }

    def get(self, **_: Any) -> dict[str, Any]:
        return {
            "documents": [document.page_content for document in self.documents],
            "metadatas": [document.metadata for document in self.documents],
        }

    def _filter(self, where: dict[str, Any] | None) -> list[Document]:
        if where is None:
            return self.documents
        return [
            document
            for document in self.documents
            if all(document.metadata.get(key) == value for key, value in where.items())
        ]


class FakeReranker:
    def predict(self, pairs: list[list[str]]) -> list[int]:
        return [int("mean" in document.lower()) for _, document in pairs]


class UnusedLlm:
    def invoke(self, _: str) -> Any:
        raise AssertionError("The LLM should not be called when context fits the token limit.")


def test_query_collection_maps_documents_and_applies_filter() -> None:
    collection = FakeCollection(
        [make_document("Mean is the average.", course="statistics"), make_document("Tree depth.", course="trees")]
    )

    result = retriever.query_collection(
        collection,
        "What is mean?",
        n_results=5,
        where={"course": "statistics"},
    )

    assert len(result) == 1
    assert result[0].page_content == "Mean is the average."
    assert collection.query_calls[0]["where"] == {"course": "statistics"}


def test_query_collection_returns_empty_on_vector_store_error() -> None:
    class BrokenCollection:
        def query(self, **_: Any) -> None:
            raise RuntimeError("vector store unavailable")

    assert retriever.query_collection(BrokenCollection(), "mean", 3) == []


def test_query_normalization_and_tokenization_remove_noise() -> None:
    assert retriever.normalize_retrieval_query("  MEAN\r\n of values  ") == "mean of values"
    assert retriever.tokenize_text("The mean of data") == ["mean", "data"]
    assert retriever.tokenize_text("the mean", remove_stopwords=False) == ["the", "mean"]


def test_get_collection_documents_applies_exact_metadata_filters() -> None:
    collection = FakeCollection(
        [
            make_document("statistics chunk", course="statistics"),
            make_document("python chunk", course="python"),
        ]
    )

    results = retriever.get_collection_documents(collection, where={"course": "statistics"})

    assert [document.page_content for document in results] == ["statistics chunk"]


def test_bm25_retrieval_ranks_matching_document_first() -> None:
    collection = FakeCollection(
        [
            make_document("Decision trees split observations by feature.", index=1),
            make_document("The mean is calculated by averaging observations.", index=2),
        ]
    )

    results = retriever.bm25_retrieve_documents(collection, "how is the mean calculated", 2)

    assert results[0].metadata["chunk_id"] == "chunk-2"


def test_bm25_handles_empty_and_query_without_indexed_terms() -> None:
    assert retriever.bm25_retrieve_documents(FakeCollection([]), "mean", 3) == []
    documents = [make_document("mean"), make_document("median", index=2)]

    results = retriever.bm25_retrieve_documents(FakeCollection(documents), "the and of", 1)

    assert results == documents[:1]


def test_fuse_rankings_uses_repeated_hits_and_stable_ties() -> None:
    shared = make_document("shared result", index=1)
    first_only = make_document("first result", index=2)

    results = retriever.fuse_rankings_by_rrf([[shared, first_only], [shared]], top_k=2)

    assert [document.page_content for document in results] == ["shared result", "first result"]


def test_rerank_documents_sorts_scores_and_limits_results() -> None:
    documents = [make_document("unrelated", index=1), make_document("mean of values", index=2)]

    results = retriever.rerank_documents("mean", documents, FakeReranker(), top_k=1)

    assert [document.page_content for document in results] == ["mean of values"]


def test_compress_context_preserves_source_and_skips_llm_under_limit() -> None:
    document = make_document("Mean is the average.", index=4)

    context = retriever.compress_context("mean", [document], UnusedLlm(), token_limit=100)

    assert "source=lesson-4.pdf" in context
    assert "Mean is the average." in context


def test_compress_context_summarizes_chunks_over_limit() -> None:
    document = make_document("Mean is the average.", index=5)

    class SummaryLlm:
        def invoke(self, prompt: str) -> Any:
            assert "Mean is the average." in prompt
            return type("Response", (), {"content": "The mean is an average."})()

    context = retriever.compress_context("mean", [document], SummaryLlm(), token_limit=1)

    assert "The mean is an average." in context
    assert "source=lesson-5.pdf" in context


def test_summarize_chunk_preserves_formatted_source_attribution() -> None:
    document = make_document("Mean is an average.", index=6)

    class SummaryLlm:
        def invoke(self, _: str) -> Any:
            return type("Response", (), {"content": "Average of values."})()

    result = retriever.summarize_chunk_with_llm("mean", document, 3, SummaryLlm())

    assert result.startswith("[3] course=statistics | section=metrics | source=lesson-6.pdf")
    assert result.endswith("Average of values.")


def test_retrieve_context_stops_when_no_chunk_is_found() -> None:
    result = retriever.retrieve_context(
        "mean",
        FakeCollection([]),
        FakeCollection([]),
        FakeCollection([]),
        FakeReranker(),
        UnusedLlm(),
    )

    assert result["status"] == "stop"
    assert "retrieved context" in result["message"]


def test_retrieval_tool_binds_dependencies_and_returns_context() -> None:
    document = make_document("The mean is the average.", index=7)
    collection = FakeCollection([document])
    tool = retriever.build_retrieval_tool(
        FakeCollection([]),
        FakeCollection([Document(page_content="metrics", metadata={"course": "statistics"})]),
        collection,
        FakeReranker(),
        UnusedLlm(),
        top_k_final=1,
    )

    result = retriever.run_retrieval_tool_node("mean", tool)

    assert result["status"] == "ok"
    assert result["context"]
    assert result["retrieved"]["ranked_docs"][0].metadata["chunk_id"] == "chunk-7"


def test_retrieval_tool_node_rejects_blank_question_and_bad_tool_result() -> None:
    class Tool:
        def invoke(self, _: dict[str, str]) -> str:
            return "unexpected"

    assert retriever.run_retrieval_tool_node("  ", Tool())["status"] == "stop"
    assert retriever.run_retrieval_tool_node("mean", Tool())["message"] == (
        "Retrieval tool returned an unexpected response."
    )


def test_retrieval_tool_node_propagates_failure_message() -> None:
    class Tool:
        def invoke(self, _: dict[str, str]) -> dict[str, str]:
            return {"status": "stop", "message": "No relevant chunks."}

    assert retriever.run_retrieval_tool_node("mean", Tool()) == {
        "status": "stop",
        "message": "No relevant chunks.",
    }
