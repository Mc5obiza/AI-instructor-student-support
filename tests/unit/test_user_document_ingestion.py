from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from backend.rag import user_document_ingestion as ingest_user_document

pytestmark = pytest.mark.unit


def test_build_user_document_chunks_preserves_user_and_document_metadata(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	document_path = tmp_path / "Regression.pdf"
	document_path.write_bytes(b"placeholder")
	monkeypatch.setattr(
		ingest_user_document,
		"extract_pdf_text",
		lambda _: "1. Regression\nDecision trees split data into subsets.",
	)

	chunks = ingest_user_document.build_user_document_chunks(
		document_path,
		document_id="document-123",
		user_id="user-456",
	)

	assert len(chunks) == 1
	assert chunks[0].page_content == "1. regression decision trees split data into subsets."
	assert chunks[0].metadata == {
		"course": "regression",
		"section": "1. regression",
		"source_file": "Regression.pdf",
		"chunk_id": "chunk::document-123::1. regression::0",
		"id_document": "document-123",
		"id_user": "user-456",
	}


def test_build_user_document_chunks_rejects_unsupported_file(tmp_path: Path) -> None:
	document_path = tmp_path / "slides.docx"
	document_path.write_bytes(b"placeholder")

	with pytest.raises(ValueError, match="Unsupported document type"):
		ingest_user_document.build_user_document_chunks(
			document_path,
			document_id="document-123",
			user_id="user-456",
		)


def test_ingest_user_document_persists_owner_and_document_and_adds_chunks(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	document_path = tmp_path / "Regression.pdf"
	document_path.write_bytes(b"placeholder")
	monkeypatch.setattr(
		ingest_user_document,
		"extract_pdf_text",
		lambda _: "1. Regression\nDecision trees split data into subsets.",
	)

	class Cursor:
		def __init__(self) -> None:
			self.executions: list[tuple[str, tuple[str, ...]]] = []

		def execute(self, query: str, values: tuple[str, ...]) -> None:
			self.executions.append((query, values))

		def fetchone(self) -> tuple[str]:
			return ("user-456",)

		def close(self) -> None:
			pass

	class Connection:
		def __init__(self) -> None:
			self.test_cursor = Cursor()
			self.committed = False
			self.rolled_back = False

		def cursor(self) -> Cursor:
			return self.test_cursor

		def commit(self) -> None:
			self.committed = True

		def rollback(self) -> None:
			self.rolled_back = True

	class Collection:
		def __init__(self) -> None:
			self.added: dict[str, Any] | None = None

		def add(self, **kwargs: Any) -> None:
			self.added = kwargs

	connection = Connection()
	collection = Collection()

	result = ingest_user_document.ingest_user_document(
		file_path=document_path,
		user_id=" user-456 ",
		connection=connection,
		collection=collection,
	)

	assert result["id_user"] == "user-456"
	assert result["file_name"] == "Regression.pdf"
	assert result["chunk_count"] == 1
	assert connection.committed
	assert not connection.rolled_back
	assert [query for query, _ in connection.test_cursor.executions] == [
		"SELECT id FROM users WHERE id = %s",
		"INSERT INTO documents (id_document, file_name) VALUES (%s, %s)",
		"INSERT INTO user_documents (id_document, id_user) VALUES (%s, %s)",
	]
	assert collection.added is not None
	assert collection.added["metadatas"][0]["id_user"] == "user-456"


def test_ingest_user_document_removes_chunks_if_database_write_fails(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	document_path = tmp_path / "Regression.pdf"
	document_path.write_bytes(b"placeholder")
	monkeypatch.setattr(
		ingest_user_document,
		"extract_pdf_text",
		lambda _: "1. Regression\nDecision trees split data into subsets.",
	)

	class Cursor:
		def __init__(self) -> None:
			self.executions = 0

		def execute(self, _query: str, _values: tuple[str, ...]) -> None:
			self.executions += 1
			if self.executions == 2:
				raise RuntimeError("insert failed")

		def fetchone(self) -> tuple[str]:
			return ("user-456",)

		def close(self) -> None:
			pass

	class Connection:
		def __init__(self) -> None:
			self.test_cursor = Cursor()
			self.rolled_back = False

		def cursor(self) -> Cursor:
			return self.test_cursor

		def commit(self) -> None:
			raise AssertionError("failed insert must not commit")

		def rollback(self) -> None:
			self.rolled_back = True

	class Collection:
		def __init__(self) -> None:
			self.deleted_ids: list[str] | None = None

		def add(self, **_kwargs: Any) -> None:
			pass

		def delete(self, ids: list[str]) -> None:
			self.deleted_ids = ids

	connection = Connection()
	collection = Collection()

	with pytest.raises(RuntimeError, match="insert failed"):
		ingest_user_document.ingest_user_document(
			file_path=document_path,
			user_id="user-456",
			connection=connection,
			collection=collection,
		)

	assert connection.rolled_back
	assert collection.deleted_ids
