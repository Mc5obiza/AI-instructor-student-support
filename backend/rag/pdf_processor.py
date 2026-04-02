from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"


def _dedupe_keep_order(items: List[Any]) -> List[Any]:
	seen = set()
	out: List[Any] = []
	for item in items:
		if item not in seen:
			seen.add(item)
			out.append(item)
	return out


def _build_document_text_and_page_spans(pages: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, int]]]:
	parts: List[str] = []
	spans: List[Dict[str, int]] = []
	cursor = 0

	for i, page in enumerate(pages):
		text = str(page.get("text", "")).strip()
		if not text:
			continue

		if parts:
			parts.append("\n\n")
			cursor += 2

		start = cursor
		parts.append(text)
		cursor += len(text)

		page_raw = page.get("page", i + 1)
		try:
			page_num = int(page_raw)
		except (TypeError, ValueError):
			page_num = i + 1

		spans.append({"start": start, "end": cursor, "page": page_num})

	return "".join(parts), spans


def split_numbered_sections_with_hierarchy(text: str) -> List[Dict[str, Any]]:
	"""Split by numbered headings and preserve parent-child header context."""
	heading_pattern = re.compile(r"(?m)^\s*(\d+(?:\.\d+)*)\.?\s+(.+?)\s*$")
	matches = list(heading_pattern.finditer(text))

	if not matches:
		cleaned = text.strip()
		return (
			[{"headers": [], "part": "Full Document", "body": cleaned, "start": 0, "end": len(cleaned)}]
			if cleaned
			else []
		)

	sections: List[Dict[str, Any]] = []
	stack: Dict[int, str] = {}

	for i, match in enumerate(matches):
		number = match.group(1).strip()
		title = match.group(2).strip()
		level = len(number.split("."))
		current_header = f"{number}. {title}"

		for key in list(stack.keys()):
			if key >= level:
				del stack[key]
		stack[level] = current_header

		headers = [stack[key] for key in sorted(stack.keys())]
		start = match.end()
		end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
		body = text[start:end].strip()

		if not body:
			continue

		sections.append(
			{
				"headers": headers,
				"part": current_header,
				"body": body,
				"start": start,
				"end": end,
			}
		)

	return sections


def merge_small_sections(
	sections: List[Dict[str, Any]],
	min_section_chars: int = 300,
	max_merge_headers: int = 3,
) -> List[Dict[str, Any]]:
	"""Merge tiny neighbor sections to improve context quality for retrieval."""
	if not sections:
		return []

	merged: List[Dict[str, Any]] = []
	i = 0
	while i < len(sections):
		current = {
			"headers": list(sections[i].get("headers", [])),
			"part": sections[i].get("part", "Full Document"),
			"body": sections[i].get("body", "").strip(),
			"start": int(sections[i].get("start", 0)),
			"end": int(sections[i].get("end", 0)),
		}

		while len(current["body"]) < min_section_chars and i + 1 < len(sections):
			nxt = sections[i + 1]
			nxt_body = str(nxt.get("body", "")).strip()
			if nxt_body:
				current["body"] = (current["body"] + "\n\n" + nxt_body).strip()

			merged_headers = _dedupe_keep_order(current.get("headers", []) + nxt.get("headers", []))
			current["headers"] = merged_headers[-max_merge_headers:] if merged_headers else []
			if current["headers"]:
				current["part"] = current["headers"][-1]

			current["end"] = int(nxt.get("end", current["end"]))
			i += 1

		if current["body"]:
			merged.append(current)
		i += 1

	return merged


def _pages_for_span(start: int, end: int, page_spans: List[Dict[str, int]]) -> List[int]:
	pages: List[int] = []
	for span in page_spans:
		overlaps = span["start"] < end and start < span["end"]
		if overlaps:
			pages.append(int(span["page"]))
	return _dedupe_keep_order(pages)


def _normalize_course_title(pdf_path: Path) -> str:
	return pdf_path.stem.replace("_", " ").replace("-", " ").strip()


def _to_source_string(path: Path) -> str:
	try:
		return str(path.resolve().relative_to(PROJECT_ROOT))
	except ValueError:
		return str(path.resolve())


def chunk_raw_pdfs(
	raw_dir: Optional[str | Path] = None,
	chunk_size: int = 1000,
	chunk_overlap: int = 150,
	min_section_chars: int = 300,
	max_header_depth_in_metadata: int = 3,
) -> List[Dict[str, Any]]:
	"""Chunk raw PDFs into metadata-rich fragments for vector indexing."""
	raw_path = Path(raw_dir) if raw_dir is not None else DEFAULT_RAW_DIR
	raw_path = raw_path.resolve()
	pdf_files = sorted(raw_path.glob("*.pdf"))

	splitter = RecursiveCharacterTextSplitter(
		chunk_size=chunk_size,
		chunk_overlap=chunk_overlap,
		separators=["\n\n", "\n", ". ", " ", ""],
	)

	chunks: List[Dict[str, Any]] = []

	for pdf_path in pdf_files:
		course_title = _normalize_course_title(pdf_path)
		docs = PyPDFLoader(str(pdf_path)).load()

		page_rows: List[Dict[str, Any]] = []
		for idx, doc in enumerate(docs, start=1):
			page_text = str(doc.page_content or "").strip()
			if not page_text:
				continue

			raw_page = (doc.metadata or {}).get("page", idx - 1)
			try:
				page_num = int(raw_page) + 1
			except (TypeError, ValueError):
				page_num = idx

			page_rows.append({"page": page_num, "text": page_text})

		if not page_rows:
			continue

		full_text, page_spans = _build_document_text_and_page_spans(page_rows)
		raw_sections = split_numbered_sections_with_hierarchy(full_text)
		sections = merge_small_sections(raw_sections, min_section_chars=min_section_chars)

		for section_idx, section in enumerate(sections, start=1):
			body = str(section.get("body", "")).strip()
			if not body:
				continue

			headers = section.get("headers", [])
			headers = headers[-max_header_depth_in_metadata:] if headers else []
			header_path = " > ".join(headers) if headers else ""
			part = headers[-1] if headers else str(section.get("part", "Full Document"))

			section_start = int(section.get("start", 0))
			section_end = int(section.get("end", section_start + len(body)))
			pages = _pages_for_span(section_start, section_end, page_spans)
			page_start = pages[0] if pages else None
			page_end = pages[-1] if pages else None

			context_prefix = header_path if header_path else part
			text_for_chunking = f"{context_prefix}\n{body}".strip()
			sub_chunks = splitter.split_text(text_for_chunking) or [text_for_chunking]

			for chunk_idx, sub_text in enumerate(sub_chunks, start=1):
				chunk_id = f"{course_title}::s{section_idx}::c{chunk_idx}"
				chunks.append(
					{
						"id": chunk_id,
						"text": sub_text,
						"metadata": {
							"course": course_title,
							"part": part,
							"header_path": header_path,
							"source": _to_source_string(pdf_path),
							"page_start": page_start,
							"page_end": page_end,
							"section_index": section_idx,
							"chunk_index": chunk_idx,
						},
					}
				)

	return chunks

