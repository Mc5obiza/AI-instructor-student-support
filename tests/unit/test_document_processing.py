from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents import Document

from backend.rag import notebook_processor, pdf_processor

pytestmark = pytest.mark.unit


def test_clean_course_name_normalizes_file_stem() -> None:
    assert pdf_processor.clean_course_name(Path("Intro_to-Data Science.PDF")) == "intro to data science"


def test_normalize_chunk_text_joins_hyphenated_lines_and_whitespace() -> None:
    assert pdf_processor.normalize_chunk_text("Data analy-\nsis\r\nuses\n\nstatistics.") == (
        "data analysis uses statistics."
    )


def test_detect_sections_uses_numbered_headings_and_preserves_text() -> None:
    text = "1.Introduction\nOpening text\n2. Methods\nDetails"

    assert pdf_processor.detect_sections(text) == [
        ("1.introduction", "1.Introduction\nOpening text"),
        ("2. methods", "2. Methods\nDetails"),
    ]


def test_detect_sections_falls_back_for_text_without_headings() -> None:
    assert pdf_processor.detect_sections("No numbered headings") == [
        ("0.general", "No numbered headings")
    ]


def test_chunker_applies_configured_overlap() -> None:
    chunks = pdf_processor.build_chunker(chunk_size_tokens=5, chunk_overlap_tokens=2).split_text(
        "one two three four five six seven eight"
    )

    assert len(chunks) > 1
    assert chunks[0].split()[-2:] == chunks[1].split()[:2]


def test_notebook_extraction_joins_nonempty_cells(tmp_path: Path) -> None:
    notebook_path = tmp_path / "lesson.ipynb"
    notebook_path.write_text(
        json.dumps(
            {
                "cells": [
                    {"cell_type": "markdown", "id": "heading", "metadata": {}, "source": ["# Regression"]},
                    {"cell_type": "code", "id": "code", "metadata": {}, "source": ["model.fit(X, y)"]},
                    {"cell_type": "markdown", "id": "blank", "metadata": {}, "source": ["  "]},
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )

    assert notebook_processor.extract_notebook_text(notebook_path) == "# Regression\n\nmodel.fit(X, y)"


def test_build_documents_indexes_notebook_chunks_and_source_metadata(tmp_path: Path) -> None:
    notebook_path = tmp_path / "Statistics_Intro.ipynb"
    notebook_path.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "markdown",
                        "id": "content",
                        "metadata": {},
                        "source": ["1. Descriptive statistics\nMean and median summarize a dataset."],
                    }
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )

    courses, parts, chunks = pdf_processor.build_documents(tmp_path)

    assert [document.page_content for document in courses] == ["statistics intro"]
    assert parts[0].metadata == {"course": "statistics intro", "section": "1. descriptive statistics"}
    assert chunks[0].page_content == "1. descriptive statistics mean and median summarize a dataset."
    assert chunks[0].metadata == {
        "course": "statistics intro",
        "section": "1. descriptive statistics",
        "source_file": "Statistics_Intro.ipynb",
        "chunk_id": "chunk::statistics intro::1. descriptive statistics::0",
    }


def test_build_documents_extracts_pdf_and_keeps_page_text_connected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "Regression.pdf").write_bytes(b"test pdf placeholder")
    monkeypatch.setattr(
        pdf_processor,
        "extract_pdf_text",
        lambda _: "1. Regression\nFirst page context\nSecond page context",
    )

    _, _, chunks = pdf_processor.build_documents(tmp_path)

    assert len(chunks) == 1
    assert "first page context second page context" in chunks[0].page_content


def test_upsert_documents_batches_documents_and_ignores_empty_input() -> None:
    class Collection:
        def __init__(self) -> None:
            self.batches: list[dict[str, Any]] = []

        def add(self, **kwargs: Any) -> None:
            self.batches.append(kwargs)

    collection = Collection()
    documents = [
        Document(page_content=f"chunk {index}", metadata={"chunk_id": str(index)})
        for index in range(3)
    ]

    pdf_processor.upsert_documents(collection, documents, "chunk", batch_size=2)
    pdf_processor.upsert_documents(collection, [], "chunk")

    assert [batch["ids"] for batch in collection.batches] == [
        ["chunk::0", "chunk::1"],
        ["chunk::2"],
    ]
    assert collection.batches[1]["documents"] == ["chunk 2"]


def test_reset_collections_deletes_named_collections_and_ignores_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deleted: list[str] = []

    class Client:
        def delete_collection(self, name: str) -> None:
            if name == "missing":
                raise ValueError("collection not found")
            deleted.append(name)

    monkeypatch.setattr(pdf_processor.SharedSystemClient, "clear_system_cache", lambda: None)
    monkeypatch.setattr(pdf_processor, "create_chroma_client", lambda _: Client())

    pdf_processor.reset_collections(Path("unused"), ["existing", "missing"])

    assert deleted == ["existing"]


def test_create_vector_databases_reports_document_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Collection:
        pass

    class Client:
        def get_or_create_collection(self, **_: Any) -> Collection:
            return Collection()

    collections = iter([Collection(), Collection(), Collection()])
    monkeypatch.setattr(pdf_processor, "reset_collections", lambda *_: None)
    monkeypatch.setattr(pdf_processor.embedding_functions, "OllamaEmbeddingFunction", lambda **_: object())
    monkeypatch.setattr(pdf_processor, "create_chroma_client", lambda _: Client())
    monkeypatch.setattr(
        pdf_processor,
        "build_documents",
        lambda _: (
            [Document(page_content="course")],
            [Document(page_content="part")],
            [Document(page_content="chunk")],
        ),
    )
    monkeypatch.setattr(pdf_processor, "upsert_documents", lambda *_args, **_kwargs: next(collections))

    result = pdf_processor.create_vector_databases(tmp_path / "raw", tmp_path / "vectors")

    assert result["course_count"] == 1
    assert result["part_count"] == 1
    assert result["chunk_count"] == 1
    assert Path(result["persist_dir"]) == tmp_path / "vectors"
