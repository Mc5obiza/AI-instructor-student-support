from __future__ import annotations

from pathlib import Path

from nbformat import read as nb_read


def extract_notebook_text(file_path: Path) -> str:
	"""Extract all notebook cell content as plain text."""
	with file_path.open("r", encoding="utf-8") as file:
		notebook = nb_read(file, as_version=4)

	blocks: list[str] = []
	for cell in notebook.cells:
		source = cell.get("source", "")
		if isinstance(source, list):
			source = "".join(source)
		if source.strip():
			blocks.append(source)

	return "\n\n".join(blocks)
