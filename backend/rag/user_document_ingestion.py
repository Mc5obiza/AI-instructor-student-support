from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import pymysql
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

from config.DB_CONFIG import DB_CONFIG

from .notebook_processor import extract_notebook_text
from .pdf_processor import (
	DEFAULT_EMBEDDING_MODEL,
	DEFAULT_EMBEDDING_URL,
	DEFAULT_PERSIST_DIR,
	build_chunker,
	clean_course_name,
	detect_sections,
	extract_pdf_text,
	normalize_chunk_text,
)

USER_DOCUMENT_COLLECTION = "user_document_chunks"
SUPPORTED_EXTENSIONS = {".pdf", ".ipynb"}


def build_user_document_chunks(
	file_path: Path,
	document_id: str,
	user_id: str,
) -> list[Document]:
	"""Extract and chunk one supported document with ownership metadata."""
	file_path = file_path.resolve()
	if not file_path.is_file():
		raise FileNotFoundError(f"Document file was not found: {file_path}")
	if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
		supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
		raise ValueError(f"Unsupported document type {file_path.suffix!r}; use: {supported}")

	if file_path.suffix.lower() == ".pdf":
		text = extract_pdf_text(file_path)
	else:
		text = extract_notebook_text(file_path)

	chunker = build_chunker()
	course = clean_course_name(file_path)
	chunks: list[Document] = []
	for section, section_text in detect_sections(text):
		for chunk_index, chunk_text in enumerate(chunker.split_text(section_text)):
			normalized_text = normalize_chunk_text(chunk_text)
			if not normalized_text:
				continue

			chunks.append(
				Document(
					page_content=normalized_text,
					metadata={
						"course": course,
						"section": section,
						"source_file": file_path.name,
						"chunk_id": f"chunk::{document_id}::{section}::{chunk_index}",
						"id_document": document_id,
						"id_user": user_id,
					},
				)
			)

	if not chunks:
		raise ValueError(f"No text could be extracted from document: {file_path.name}")
	return chunks


def create_user_document_collection(
	persist_dir: Path,
	embedding_model: str = DEFAULT_EMBEDDING_MODEL,
	embedding_url: str = DEFAULT_EMBEDDING_URL,
) -> Any:
	"""Return the persistent Chroma collection for user-uploaded document chunks."""
	persist_dir.mkdir(parents=True, exist_ok=True)
	embedding_function: Any = embedding_functions.OllamaEmbeddingFunction(
		model_name=embedding_model,
		url=embedding_url,
	)
	client = PersistentClient(path=str(persist_dir))
	return client.get_or_create_collection(
		name=USER_DOCUMENT_COLLECTION,
		metadata={"hnsw:space": "cosine"},
		embedding_function=embedding_function,
	)


def connect_database() -> Any:
	"""Open a MariaDB/MySQL connection using the project's DB configuration."""
	return pymysql.connect(
		host=DB_CONFIG["HOST"],
		user=DB_CONFIG["USER"],
		password=DB_CONFIG["PASSWORD"],
		database=DB_CONFIG["DATABASE"],
		port=DB_CONFIG["PORT"],
	)


def ingest_user_document(
	file_path: Path,
	user_id: str,
	persist_dir: Path = DEFAULT_PERSIST_DIR,
	embedding_model: str = DEFAULT_EMBEDDING_MODEL,
	embedding_url: str = DEFAULT_EMBEDDING_URL,
	connection: Any | None = None,
	collection: Any | None = None,
) -> dict[str, Any]:
	"""Index one document's chunks and persist its MariaDB ownership records."""
	normalized_user_id = user_id.strip()
	if not normalized_user_id:
		raise ValueError("User ID is required.")

	file_path = file_path.resolve()
	document_id = str(uuid4())
	chunks = build_user_document_chunks(file_path, document_id, normalized_user_id)
	vector_collection = (
		collection
		if collection is not None
		else create_user_document_collection(
			persist_dir=persist_dir,
			embedding_model=embedding_model,
			embedding_url=embedding_url,
		)
	)
	db_connection = connection if connection is not None else connect_database()
	cursor = None
	chroma_ids = [f"user-document::{document_id}::{index}" for index in range(len(chunks))]
	vectors_may_exist = False

	try:
		cursor = db_connection.cursor()
		cursor.execute("SELECT id FROM users WHERE id = %s", (normalized_user_id,))
		if cursor.fetchone() is None:
			raise ValueError(f"User does not exist: {normalized_user_id}")

		vectors_may_exist = True
		vector_collection.add(
			ids=chroma_ids,
			documents=[chunk.page_content for chunk in chunks],
			metadatas=[chunk.metadata for chunk in chunks],
		)

		cursor.execute(
			"INSERT INTO documents (id_document, file_name) VALUES (%s, %s)",
			(document_id, file_path.name),
		)
		cursor.execute(
			"INSERT INTO user_documents (id_document, id_user) VALUES (%s, %s)",
			(document_id, normalized_user_id),
		)
		db_connection.commit()
	except Exception:
		db_connection.rollback()
		if vectors_may_exist:
			try:
				vector_collection.delete(ids=chroma_ids)
			except Exception as cleanup_error:
				raise RuntimeError(
					"Ingestion failed and the new Chroma chunks could not be removed."
				) from cleanup_error
		raise
	finally:
		if cursor is not None:
			cursor.close()
		if connection is None:
			db_connection.close()

	return {
		"id_document": document_id,
		"id_user": normalized_user_id,
		"file_name": file_path.name,
		"chunk_count": len(chunks),
		"collection": USER_DOCUMENT_COLLECTION,
	}
