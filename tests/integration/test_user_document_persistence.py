from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from backend.rag import user_document_ingestion as ingest_user_document

pytestmark = pytest.mark.integration

RUN_MARIADB_INTEGRATION = os.getenv("RUN_MARIADB_INTEGRATION") == "1"


class MemoryCollection:
	def __init__(self) -> None:
		self.document_ids: list[str] = []

	def add(self, **kwargs: Any) -> None:
		assert kwargs["ids"]
		self.document_ids.extend(
			str(metadata["id_document"]) for metadata in kwargs["metadatas"]
		)

	def delete(self, **kwargs: Any) -> None:
		assert kwargs["ids"]


@pytest.mark.skipif(
	not RUN_MARIADB_INTEGRATION,
	reason="Set RUN_MARIADB_INTEGRATION=1 to connect to the configured MariaDB database.",
)
def test_ingestion_persists_document_and_owner_in_mariadb(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	lookup_connection = ingest_user_document.connect_database()
	lookup_cursor = None
	try:
		lookup_cursor = lookup_connection.cursor()
		lookup_cursor.execute("SELECT id FROM users LIMIT 1")
		user_row = lookup_cursor.fetchone()
	finally:
		if lookup_cursor is not None:
			lookup_cursor.close()
		lookup_connection.close()
	if user_row is None:
		pytest.skip("MariaDB integration test requires at least one existing user.")

	user_id = str(user_row[0])
	document_path = tmp_path / "integration-document.pdf"
	document_path.write_bytes(b"test PDF placeholder")
	monkeypatch.setattr(
		ingest_user_document,
		"extract_pdf_text",
		lambda _: "1. Integration\nThis text verifies database persistence.",
	)

	collection = MemoryCollection()
	document_id: str | None = None
	try:
		result = ingest_user_document.ingest_user_document(
			file_path=document_path,
			user_id=user_id,
			collection=collection,
		)
		document_id = str(result["id_document"])

		verification_connection = ingest_user_document.connect_database()
		verification_cursor = None
		try:
			verification_cursor = verification_connection.cursor()
			verification_cursor.execute(
				"""
				SELECT documents.file_name, user_documents.id_user
				FROM documents
				INNER JOIN user_documents
					ON user_documents.id_document = documents.id_document
				WHERE documents.id_document = %s
				""",
				(document_id,),
			)
			row = verification_cursor.fetchone()
		finally:
			if verification_cursor is not None:
				verification_cursor.close()
			verification_connection.close()

		assert row == (document_path.name, user_id)
	finally:
		cleanup_connection = ingest_user_document.connect_database()
		cleanup_cursor = None
		try:
			cleanup_cursor = cleanup_connection.cursor()
			temporary_document_id = document_id or (
				collection.document_ids[0] if collection.document_ids else None
			)
			if temporary_document_id is not None:
				cleanup_cursor.execute(
					"DELETE FROM documents WHERE id_document = %s",
					(temporary_document_id,),
				)
			cleanup_connection.commit()
		finally:
			if cleanup_cursor is not None:
				cleanup_cursor.close()
			cleanup_connection.close()
