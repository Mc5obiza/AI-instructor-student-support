from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from backend.rag.user_document_ingestion import (
	DEFAULT_EMBEDDING_MODEL,
	DEFAULT_EMBEDDING_URL,
	DEFAULT_PERSIST_DIR,
	ingest_user_document,
)


def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments for user document ingestion."""
	parser = argparse.ArgumentParser(
		description="Ingest one PDF or Jupyter notebook for a MariaDB user."
	)
	parser.add_argument("--file", type=Path, required=True, help="Path to a .pdf or .ipynb file.")
	parser.add_argument("--user-id", required=True, help="Existing users.id value.")
	parser.add_argument(
		"--persist-dir",
		type=Path,
		default=DEFAULT_PERSIST_DIR,
		help="Directory where the user document Chroma collection is stored.",
	)
	parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
	parser.add_argument("--embedding-url", default=DEFAULT_EMBEDDING_URL)
	return parser.parse_args()


def main() -> None:
	"""Run the user document ingestion CLI."""
	args = parse_args()
	result: dict[str, Any] = ingest_user_document(
		file_path=args.file,
		user_id=args.user_id,
		persist_dir=args.persist_dir,
		embedding_model=args.embedding_model,
		embedding_url=args.embedding_url,
	)
	print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
	main()
