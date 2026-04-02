from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from chromadb import PersistentClient
from chromadb.utils import embedding_functions

try:
	from .pdf_processor import PROJECT_ROOT, chunk_raw_pdfs
except ImportError:
	from pdf_processor import PROJECT_ROOT, chunk_raw_pdfs

DEFAULT_VECTOR_DB_PATH = PROJECT_ROOT / "backend" / "vectordb" / "chroma_db"
DEFAULT_COLLECTION_NAME = "course_chunks"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
DEFAULT_COURSE_PARTS_OUTPUT = PROJECT_ROOT / "backend" / "rag" / "course_parts.json"


def _sanitize_for_chroma_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
	"""Chroma metadata values must be primitive scalar types."""
	out: Dict[str, Any] = {}
	for key, value in metadata.items():
		if value is None:
			continue
		if isinstance(value, (str, int, float, bool)):
			out[key] = value
		else:
			out[key] = str(value)
	return out


def get_embedding_function(
	model_name: str = DEFAULT_EMBEDDING_MODEL,
	api_url: str = DEFAULT_OLLAMA_EMBED_URL,
) -> Any:
	return embedding_functions.OllamaEmbeddingFunction(
		model_name=model_name,
		url=api_url,
	)


def get_chroma_client(vector_db_path: Optional[str | Path] = None) -> Any:
	path = Path(vector_db_path) if vector_db_path is not None else DEFAULT_VECTOR_DB_PATH
	path.mkdir(parents=True, exist_ok=True)
	return PersistentClient(path=str(path))


def build_course_parts_payload(df: pd.DataFrame) -> List[Dict[str, Any]]:
	if df.empty:
		return []

	payload: List[Dict[str, Any]] = []
	grouped = df.groupby("course", dropna=False)

	for course, group in grouped:
		course_name = str(course).strip() if pd.notna(course) else ""
		if not course_name:
			continue

		parts = sorted(
			{
				str(value).strip()
				for value in group["part"].tolist()
				if isinstance(value, str) and value.strip()
			}
		)
		payload.append({"course": course_name, "parts": parts})

	payload.sort(key=lambda item: item["course"].lower())
	return payload


def write_course_parts_json(
	df: pd.DataFrame,
	output_path: Optional[str | Path] = None,
) -> Path:
	path = Path(output_path) if output_path is not None else DEFAULT_COURSE_PARTS_OUTPUT
	path.parent.mkdir(parents=True, exist_ok=True)

	payload = build_course_parts_payload(df)
	path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
	return path


def create_vector_database(
	raw_dir: Optional[str | Path] = None,
	vector_db_path: Optional[str | Path] = None,
	collection_name: str = DEFAULT_COLLECTION_NAME,
	course_parts_output_path: Optional[str | Path] = None,
	embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
	embedding_api_url: str = DEFAULT_OLLAMA_EMBED_URL,
	reset_collection: bool = True,
) -> Dict[str, Any]:
	chunks = chunk_raw_pdfs(raw_dir=raw_dir)

	if not chunks:
		return {
			"collection": collection_name,
			"chunk_count": 0,
			"courses": 0,
			"course_parts_path": None,
			"vector_db_path": str(vector_db_path or DEFAULT_VECTOR_DB_PATH),
		}

	rows = []
	for row in chunks:
		raw_metadata = row.get("metadata")
		metadata: Dict[str, Any] = raw_metadata if isinstance(raw_metadata, dict) else {}
		rows.append(
			{
				"id": str(row.get("id", "")),
				"text": str(row.get("text", "")),
				"course": str(metadata.get("course", "")).strip(),
				"part": str(metadata.get("part", "")).strip(),
				"metadata": metadata,
			}
		)

	df = pd.DataFrame(rows)
	df = df[(df["id"].astype(str).str.len() > 0) & (df["text"].astype(str).str.len() > 0)].copy()
	df["id"] = df["id"].astype(str)

	chunk_ids = df["id"].tolist()
	documents = df["text"].tolist()
	metadatas = [_sanitize_for_chroma_metadata(meta) for meta in df["metadata"].tolist()]

	client = get_chroma_client(vector_db_path=vector_db_path)
	if reset_collection:
		try:
			client.delete_collection(name=collection_name)
		except Exception:
			pass

	embedding_fn = get_embedding_function(model_name=embedding_model_name, api_url=embedding_api_url)
	collection = client.get_or_create_collection(
		name=collection_name,
		embedding_function=embedding_fn,
	)

	collection.upsert(ids=chunk_ids, documents=documents, metadatas=metadatas)
	parts_path = write_course_parts_json(df, output_path=course_parts_output_path)

	return {
		"collection": collection_name,
		"chunk_count": len(chunk_ids),
		"courses": int(df["course"].nunique(dropna=True)),
		"course_parts_path": str(parts_path),
		"vector_db_path": str(vector_db_path or DEFAULT_VECTOR_DB_PATH),
	}


def main() -> None:
	result = create_vector_database(reset_collection=True)
	print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
	main()

