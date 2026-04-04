from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from chromadb import PersistentClient
from chromadb.api.shared_system_client import SharedSystemClient
from chromadb.utils import embedding_functions
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

try:
	from .notebook_processor import extract_notebook_text
except ImportError:
	# Allows running this file directly: `python backend/rag/pdf_processor.py`.
	from notebook_processor import extract_notebook_text


CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 100
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_EMBEDDING_URL = "http://localhost:11434/api/embeddings"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PERSIST_DIR = PROJECT_ROOT / "backend" / "vectordb" / "chroma_db"


def clean_course_name(file_path: Path) -> str:
	"""Map file name to a normalized course name."""
	name = file_path.stem.lower().strip()
	name = re.sub(r"[_\-]+", " ", name)
	name = re.sub(r"\s+", " ", name)
	return name


def extract_pdf_text(file_path: Path) -> str:
	"""Extract text from PDF and keep pages connected for cross-page chunks."""
	reader = PdfReader(str(file_path))
	pages = [page.extract_text() or "" for page in reader.pages]
	return "\n".join(pages)


def normalize_chunk_text(text: str) -> str:
	"""Normalize chunk text for retrieval while preserving punctuation."""
	normalized = text.replace("\r\n", "\n").replace("\r", "\n")
	normalized = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", normalized)
	normalized = re.sub(r"\s*\n+\s*", " ", normalized)
	normalized = re.sub(r"\s+", " ", normalized).strip()
	return normalized.lower()


def detect_sections(text: str) -> list[tuple[str, str]]:
	"""Split by numbered headings like '1.intro' or fallback to one section."""
	pattern = re.compile(r"^\s*(\d+\.[^\n]+)", re.MULTILINE)
	matches = list(pattern.finditer(text))

	if not matches:
		return [("0.general", text)]

	sections: list[tuple[str, str]] = []
	for i, match in enumerate(matches):
		start = match.start()
		end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
		section_name = match.group(1).strip().lower()
		section_text = text[start:end].strip()
		sections.append((section_name, section_text))

	return sections


def build_chunker(
	chunk_size_tokens: int = CHUNK_SIZE_TOKENS,
	chunk_overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> RecursiveCharacterTextSplitter:
	"""Build LangChain recursive chunker with token-like word length function."""
	return RecursiveCharacterTextSplitter(
		chunk_size=chunk_size_tokens,
		chunk_overlap=chunk_overlap_tokens,
		length_function=lambda txt: len(txt.split()),
		separators=["\n\n", "\n", ". ", " ", ""],
	)


def build_documents(raw_data_dir: Path) -> tuple[list[Document], list[Document], list[Document]]:
	"""Create documents for course, part, and chunk collections."""
	chunker = build_chunker()

	course_docs: list[Document] = []
	part_docs: list[Document] = []
	chunk_docs: list[Document] = []

	seen_courses: set[str] = set()
	seen_parts: set[tuple[str, str]] = set()

	for file_path in raw_data_dir.rglob("*"):
		if not file_path.is_file() or file_path.suffix.lower() not in {".pdf", ".ipynb"}:
			continue

		course = clean_course_name(file_path)
		if course not in seen_courses:
			seen_courses.add(course)
			course_docs.append(Document(page_content=course, metadata={"course": course}))

		full_text = extract_pdf_text(file_path) if file_path.suffix.lower() == ".pdf" else extract_notebook_text(file_path)
		for section, section_text in detect_sections(full_text):
			part_key = (course, section)
			if part_key not in seen_parts:
				seen_parts.add(part_key)
				part_docs.append(Document(page_content=section, metadata={"course": course, "section": section}))

			for idx, chunk_text in enumerate(chunker.split_text(section_text)):
				normalized_chunk = normalize_chunk_text(chunk_text)
				if not normalized_chunk:
					continue

				chunk_docs.append(
					Document(
						page_content=normalized_chunk,
						metadata={
							"course": course,
							"section": section,
							"source_file": file_path.name,
							"chunk_id": f"chunk::{course}::{section}::{idx}",
						},
					)
				)

	return course_docs, part_docs, chunk_docs


def create_chroma_client(persist_dir: Path) -> Any:
	"""Create a Chroma persistent client."""
	return PersistentClient(path=str(persist_dir))


def reset_collections(persist_dir: Path, collection_names: list[str]) -> None:
	"""Drop prototype collections to avoid duplicate indexing."""
	SharedSystemClient.clear_system_cache()
	client = create_chroma_client(persist_dir)
	for name in collection_names:
		try:
			client.delete_collection(name)
		except ValueError:
			pass


def upsert_documents(
	collection: Any,
	documents: list[Document],
	id_prefix: str,
	batch_size: int = 64,
) -> None:
	"""Insert notebook-built documents into a Chroma collection in batches."""
	if not documents:
		return

	for start in range(0, len(documents), batch_size):
		batch = documents[start : start + batch_size]
		ids: list[str] = []
		texts: list[str] = []
		metadatas: list[dict[str, Any]] = []

		for offset, doc in enumerate(batch):
			ids.append(f"{id_prefix}::{start + offset}")
			texts.append(doc.page_content)
			metadatas.append(doc.metadata)

		collection.add(ids=ids, documents=texts, metadatas=metadatas)


def create_vector_databases(
	raw_data_dir: Path,
	persist_dir: Path,
	embedding_model: str = DEFAULT_EMBEDDING_MODEL,
	embedding_url: str = DEFAULT_EMBEDDING_URL,
) -> dict[str, Any]:
	"""Chunk documents and build course/part/chunk Chroma databases."""
	persist_dir.mkdir(parents=True, exist_ok=True)

	reset_collections(
		persist_dir,
		["prototype_courses", "prototype_parts", "prototype_chunks"],
	)

	embedding_fn = embedding_functions.OllamaEmbeddingFunction(
		model_name=embedding_model,
		url=embedding_url,
	)
	client = create_chroma_client(persist_dir)

	course_collection = client.get_or_create_collection(
		name="prototype_courses",
		embedding_function=embedding_fn,
	)
	part_collection = client.get_or_create_collection(
		name="prototype_parts",
		embedding_function=embedding_fn,
	)
	chunk_collection = client.get_or_create_collection(
		name="prototype_chunks",
		embedding_function=embedding_fn,
	)

	course_docs, part_docs, chunk_docs = build_documents(raw_data_dir)
	upsert_documents(course_collection, course_docs, "course")
	upsert_documents(part_collection, part_docs, "part")
	upsert_documents(chunk_collection, chunk_docs, "chunk")

	return {
		"persist_dir": str(persist_dir),
		"collections": {
			"course_collection": "prototype_courses",
			"part_collection": "prototype_parts",
			"chunk_collection": "prototype_chunks",
		},
		"course_count": len(course_docs),
		"part_count": len(part_docs),
		"chunk_count": len(chunk_docs),
	}


def parse_args() -> argparse.Namespace:
	"""Parse CLI arguments for vector database creation."""
	parser = argparse.ArgumentParser(description="Create Chroma vector databases from raw course files.")
	parser.add_argument(
		"--raw-data-dir",
		type=Path,
		default=DEFAULT_RAW_DATA_DIR,
		help="Directory containing source .pdf/.ipynb files.",
	)
	parser.add_argument(
		"--persist-dir",
		type=Path,
		default=DEFAULT_PERSIST_DIR,
		help="Directory where Chroma collections are stored.",
	)
	parser.add_argument(
		"--embedding-model",
		default=DEFAULT_EMBEDDING_MODEL,
		help="Embedding model name served by Ollama.",
	)
	parser.add_argument(
		"--embedding-url",
		default=DEFAULT_EMBEDDING_URL,
		help="Ollama embeddings endpoint URL.",
	)
	return parser.parse_args()


def main() -> None:
	"""CLI entrypoint to build vector DB used by the RAG backend modules."""
	args = parse_args()
	raw_data_dir = args.raw_data_dir.resolve()
	persist_dir = args.persist_dir.resolve()

	if not raw_data_dir.exists():
		raise FileNotFoundError(f"Raw data directory was not found: {raw_data_dir}")

	result = create_vector_databases(
		raw_data_dir=raw_data_dir,
		persist_dir=persist_dir,
		embedding_model=args.embedding_model,
		embedding_url=args.embedding_url,
	)
	print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
	main()
