from __future__ import annotations

import hashlib
import json
import os
import re
from math import sqrt
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import ollama
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_VECTOR_DB_PATH = PROJECT_ROOT / "backend" / "vectordb" / "chroma_db"
MAIN_COLLECTION_NAME = "course_chunks"
COURSE_ROUTER_COLLECTION = "course_profiles_index"
PART_ROUTER_COLLECTION = "course_parts_profiles_index"

DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
DEFAULT_LLM_MODEL = "llama3.1"
DEFAULT_RERANKER_MODELS = [
	"BAAI/bge-reranker-large",
	"BAAI/bge-reranker-base",
	"cross-encoder/ms-marco-MiniLM-L-6-v2",
]

_client: Optional[Any] = None
_embedding_fn: Optional[Any] = None

_router_index_signature: Optional[str] = None
_router_index_ready = False
_all_course_values: List[str] = []

_cross_encoder_reranker: Optional[CrossEncoder] = None
_cross_encoder_disabled = False


def _load_hf_token_from_env() -> Optional[str]:
	token = (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or "").strip()
	return token or None


def llm_response(prompt: str, llm_model: str = DEFAULT_LLM_MODEL) -> str:
	try:
		response = ollama.chat(
			model=llm_model,
			messages=[{"role": "user", "content": prompt}],
		)
	except Exception:
		return ""

	if isinstance(response, dict):
		message = response.get("message")
	else:
		message = getattr(response, "message", None)

	if isinstance(message, dict):
		content = message.get("content")
	else:
		content = getattr(message, "content", "")

	if content is None:
		content = ""
	return str(content or "").strip()


def llm_response_stream(prompt: str, llm_model: str = DEFAULT_LLM_MODEL) -> Iterator[str]:
	"""Yield generated text chunks from Ollama as they arrive."""
	try:
		stream = ollama.chat(
			model=llm_model,
			messages=[{"role": "user", "content": prompt}],
			stream=True,
		)
	except Exception:
		return

	for chunk in stream:
		if isinstance(chunk, dict):
			message = chunk.get("message")
		else:
			message = getattr(chunk, "message", None)

		if isinstance(message, dict):
			content = message.get("content")
		else:
			content = getattr(message, "content", "")

		text = str(content or "")
		if text:
			yield text


def get_embedding_function(
	model_name: str = DEFAULT_EMBEDDING_MODEL,
	api_url: str = DEFAULT_OLLAMA_EMBED_URL,
) -> Any:
	global _embedding_fn
	if _embedding_fn is None:
		_embedding_fn = embedding_functions.OllamaEmbeddingFunction(model_name=model_name, url=api_url)
	return _embedding_fn


def get_chroma_client(vector_db_path: Optional[str | Path] = None) -> Any:
	global _client
	if _client is not None and vector_db_path is None:
		return _client

	path = Path(vector_db_path) if vector_db_path is not None else DEFAULT_VECTOR_DB_PATH
	path.mkdir(parents=True, exist_ok=True)
	client = PersistentClient(path=str(path))

	if vector_db_path is None:
		_client = client
	return client


def get_collection(
	collection_name: str,
	vector_db_path: Optional[str | Path] = None,
) -> Any:
	client = get_chroma_client(vector_db_path=vector_db_path)
	return client.get_or_create_collection(name=collection_name, embedding_function=get_embedding_function())


def _extract_field(value: Any) -> List[Any]:
	if not isinstance(value, list):
		return []
	if len(value) == 1 and isinstance(value[0], list):
		return value[0]
	return value


def _as_metadata_dict(value: Any) -> Dict[str, Any]:
	return value if isinstance(value, dict) else {}


def _cosine_similarity(a: List[float], b: List[float]) -> Optional[float]:
	if not a or not b or len(a) != len(b):
		return None

	dot = sum(x * y for x, y in zip(a, b))
	norm_a = sqrt(sum(x * x for x in a))
	norm_b = sqrt(sum(y * y for y in b))
	if norm_a == 0 or norm_b == 0:
		return None
	return dot / (norm_a * norm_b)


def _similarity_from_result(
	query_embedding: Optional[List[float]],
	result_embedding: Optional[List[float]],
	distance: Any,
	score: Any,
) -> float:
	if isinstance(score, (int, float)):
		return float(score)

	if isinstance(distance, (int, float)):
		return -float(distance)

	if query_embedding and result_embedding:
		cosine_sim = _cosine_similarity(query_embedding, result_embedding)
		if cosine_sim is not None:
			return float(cosine_sim)

	return 0.0


def _query_records(query_result: Dict[str, Any], query_embedding: Optional[List[float]] = None) -> List[Dict[str, Any]]:
	ids = _extract_field(query_result.get("ids", []))
	documents = _extract_field(query_result.get("documents", []))
	metadatas = _extract_field(query_result.get("metadatas", []))
	distances = _extract_field(query_result.get("distances", []))
	scores = _extract_field(query_result.get("scores", []))
	embeddings = _extract_field(query_result.get("embeddings", []))

	records: List[Dict[str, Any]] = []
	n = max(len(ids), len(documents), len(metadatas), len(distances), len(scores), len(embeddings))

	for idx in range(n):
		record_id = str(ids[idx]) if idx < len(ids) else None
		document = str(documents[idx]) if idx < len(documents) and documents[idx] is not None else ""
		metadata: Dict[str, Any] = _as_metadata_dict(metadatas[idx] if idx < len(metadatas) else None)
		distance = distances[idx] if idx < len(distances) else None
		score = scores[idx] if idx < len(scores) else None
		embedding = embeddings[idx] if idx < len(embeddings) and isinstance(embeddings[idx], list) else None

		similarity = _similarity_from_result(query_embedding, embedding, distance, score)

		records.append(
			{
				"id": record_id,
				"document": document,
				"metadata": metadata,
				"distance": distance,
				"score": score,
				"similarity": similarity,
			}
		)

	return records


def _normalize_text(value: Any) -> str:
	text = str(value or "").strip()
	return re.sub(r"\s+", " ", text)


def _is_generic_part(part: str) -> bool:
	lowered = _normalize_text(part).lower()
	if not lowered:
		return True

	generic = {
		"full document",
		"introduction",
		"overview",
		"summary",
		"conclusion",
		"appendix",
	}
	if lowered in generic:
		return True

	return lowered.startswith("chapter") and len(lowered.split()) <= 3


def _router_data_signature(rows: List[Dict[str, Any]]) -> str:
	normalized: List[Dict[str, str]] = []
	for row in rows:
		meta: Dict[str, Any] = _as_metadata_dict(row.get("metadata"))
		normalized.append(
			{
				"course": _normalize_text(meta.get("course", "")),
				"part": _normalize_text(meta.get("part", "")),
				"text": _normalize_text(row.get("document", ""))[:500],
			}
		)

	normalized.sort(key=lambda item: (item["course"], item["part"], item["text"]))
	payload = json.dumps(normalized, ensure_ascii=True, sort_keys=True)
	return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _fetch_main_collection_records(
	collection_name: str = MAIN_COLLECTION_NAME,
	vector_db_path: Optional[str | Path] = None,
) -> List[Dict[str, Any]]:
	collection = get_collection(collection_name=collection_name, vector_db_path=vector_db_path)
	fetched = collection.get(include=["documents", "metadatas"])

	ids = _extract_field(fetched.get("ids", []))
	docs = _extract_field(fetched.get("documents", []))
	metas = _extract_field(fetched.get("metadatas", []))

	n = max(len(ids), len(docs), len(metas))
	rows: List[Dict[str, Any]] = []

	for i in range(n):
		row_id = str(ids[i]) if i < len(ids) else None
		document = str(docs[i]) if i < len(docs) and docs[i] is not None else ""
		metadata: Dict[str, Any] = _as_metadata_dict(metas[i] if i < len(metas) else None)
		if not document:
			continue
		rows.append({"id": row_id, "document": document, "metadata": metadata})

	return rows


def build_vector_router_indexes(
	refresh: bool = False,
	vector_db_path: Optional[str | Path] = None,
	main_collection_name: str = MAIN_COLLECTION_NAME,
	course_collection_name: str = COURSE_ROUTER_COLLECTION,
	part_collection_name: str = PART_ROUTER_COLLECTION,
	course_samples_per_course: int = 30,
	part_samples_per_pair: int = 10,
	profile_char_limit: int = 4000,
) -> Dict[str, Any]:
	global _router_index_signature, _router_index_ready, _all_course_values

	rows = _fetch_main_collection_records(collection_name=main_collection_name, vector_db_path=vector_db_path)
	if not rows:
		_router_index_ready = False
		_all_course_values = []
		_router_index_signature = None
		return {"rebuilt": False, "records": 0, "courses": 0, "parts": 0}

	signature = _router_data_signature(rows)
	if _router_index_ready and not refresh and signature == _router_index_signature:
		return {
			"rebuilt": False,
			"records": len(rows),
			"courses": len(_all_course_values),
			"parts": -1,
		}

	client = get_chroma_client(vector_db_path=vector_db_path)
	embedding_fn = get_embedding_function()

	for router_name in (course_collection_name, part_collection_name):
		try:
			client.delete_collection(name=router_name)
		except Exception:
			pass

	course_collection = client.get_or_create_collection(name=course_collection_name, embedding_function=embedding_fn)
	part_collection = client.get_or_create_collection(name=part_collection_name, embedding_function=embedding_fn)

	by_course: Dict[str, List[Dict[str, str]]] = {}
	by_pair: Dict[Tuple[str, str], List[str]] = {}

	for row in rows:
		metadata: Dict[str, Any] = _as_metadata_dict(row.get("metadata"))
		course = _normalize_text(metadata.get("course", ""))
		part = _normalize_text(metadata.get("part", ""))
		text = _normalize_text(row.get("document", ""))

		if not course or not text:
			continue

		by_course.setdefault(course, []).append({"part": part, "text": text})
		if part:
			by_pair.setdefault((course, part), []).append(text)

	course_ids: List[str] = []
	course_docs: List[str] = []
	course_metas: List[Dict[str, Any]] = []

	for course in sorted(by_course.keys(), key=lambda value: value.lower()):
		samples = by_course[course]
		samples_sorted = sorted(samples, key=lambda row: (row["part"], row["text"]))[:course_samples_per_course]

		parts = sorted({row["part"] for row in samples if row["part"]})
		snippets = [f"{row['part']}: {row['text'][:300]}" for row in samples_sorted]

		profile = (
			f"Course: {course}\n"
			f"Parts: {', '.join(parts)}\n\n"
			f"{chr(10).join(snippets)}"
		).strip()[:profile_char_limit]

		course_id = f"course::{hashlib.sha1(course.encode('utf-8')).hexdigest()[:20]}"
		course_ids.append(course_id)
		course_docs.append(profile)
		course_metas.append({"course": course, "parts_count": len(parts)})

	if course_ids:
		course_collection.upsert(ids=course_ids, documents=course_docs, metadatas=course_metas)

	part_ids: List[str] = []
	part_docs: List[str] = []
	part_metas: List[Dict[str, Any]] = []

	for course, part in sorted(by_pair.keys(), key=lambda pair: (pair[0].lower(), pair[1].lower())):
		texts = sorted(by_pair[(course, part)])[:part_samples_per_pair]
		snippets = [text[:350] for text in texts]

		profile = (
			f"Course: {course}\n"
			f"Part: {part}\n\n"
			f"{chr(10).join(snippets)}"
		).strip()[:profile_char_limit]

		key = f"{course}::{part}"
		part_id = f"part::{hashlib.sha1(key.encode('utf-8')).hexdigest()[:20]}"
		part_ids.append(part_id)
		part_docs.append(profile)
		part_metas.append({"course": course, "part": part})

	if part_ids:
		part_collection.upsert(ids=part_ids, documents=part_docs, metadatas=part_metas)

	_router_index_signature = signature
	_router_index_ready = True
	_all_course_values = sorted(by_course.keys(), key=lambda value: value.lower())

	return {
		"rebuilt": True,
		"records": len(rows),
		"courses": len(course_ids),
		"parts": len(part_ids),
	}


def retrieve_courses(
	query: str,
	desired_courses: int = 2,
	max_candidates: int = 8,
	vector_db_path: Optional[str | Path] = None,
) -> List[str]:
	build_vector_router_indexes(refresh=False, vector_db_path=vector_db_path)

	if not _router_index_ready:
		return []

	course_collection = get_collection(collection_name=COURSE_ROUTER_COLLECTION, vector_db_path=vector_db_path)
	query_result = course_collection.query(
		query_texts=[query],
		n_results=max(desired_courses, max_candidates),
		include=["metadatas", "documents", "distances"],
	)

	records = _query_records(query_result)
	best_by_course: Dict[str, float] = {}

	for record in records:
		metadata: Dict[str, Any] = _as_metadata_dict(record.get("metadata"))
		course = _normalize_text(metadata.get("course", ""))
		if not course:
			continue

		similarity = float(record.get("similarity", 0.0))
		previous = best_by_course.get(course)
		if previous is None or similarity > previous:
			best_by_course[course] = similarity

	if not best_by_course:
		return []

	ranked = sorted(best_by_course.items(), key=lambda item: (-item[1], item[0].lower()))
	return [course for course, _ in ranked[:desired_courses]]


def retrieve_parts_for_courses(
	query: str,
	routed_courses: List[str],
	max_parts_per_course: int = 3,
	max_candidates_per_course: int = 8,
	vector_db_path: Optional[str | Path] = None,
) -> Dict[str, List[str]]:
	if not routed_courses:
		return {}

	build_vector_router_indexes(refresh=False, vector_db_path=vector_db_path)
	part_collection = get_collection(collection_name=PART_ROUTER_COLLECTION, vector_db_path=vector_db_path)

	result: Dict[str, List[str]] = {}

	for course in routed_courses:
		try:
			query_result = part_collection.query(
				query_texts=[query],
				n_results=max_parts_per_course + max_candidates_per_course,
				where={"course": course},
				include=["metadatas", "documents", "distances"],
			)
		except Exception:
			query_result = {"ids": [], "documents": [], "metadatas": [], "distances": []}

		records = _query_records(query_result)
		best_by_part: Dict[str, float] = {}

		for record in records:
			metadata: Dict[str, Any] = _as_metadata_dict(record.get("metadata"))
			part = _normalize_text(metadata.get("part", ""))
			if not part:
				continue
			similarity = float(record.get("similarity", 0.0))
			previous = best_by_part.get(part)
			if previous is None or similarity > previous:
				best_by_part[part] = similarity

		ranked_parts = sorted(best_by_part.items(), key=lambda item: (-item[1], item[0].lower()))
		selected = [part for part, _ in ranked_parts[:max_parts_per_course]]
		if selected:
			result[course] = selected

	return result


def route_query_to_course_part(
	query: str,
	n_courses: int = 2,
	max_parts_per_course: int = 3,
	vector_db_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
	courses = retrieve_courses(
		query=query,
		desired_courses=n_courses,
		vector_db_path=vector_db_path,
	)
	parts = retrieve_parts_for_courses(
		query=query,
		routed_courses=courses,
		max_parts_per_course=max_parts_per_course,
		vector_db_path=vector_db_path,
	)
	return {"courses": courses, "parts": parts}


def vector_search(
	query: str,
	n_results: int = 120,
	vector_db_path: Optional[str | Path] = None,
	collection_name: str = MAIN_COLLECTION_NAME,
) -> List[Dict[str, Any]]:
	collection = get_collection(collection_name=collection_name, vector_db_path=vector_db_path)
	query_result = collection.query(
		query_texts=[query],
		n_results=n_results,
		include=["documents", "metadatas", "distances"],
	)

	records = _query_records(query_result)
	candidates: List[Dict[str, Any]] = []

	for record in records:
		metadata: Dict[str, Any] = _as_metadata_dict(record.get("metadata"))
		text = record.get("document", "")
		if not text:
			continue

		candidates.append(
			{
				"id": record.get("id"),
				"text": str(text),
				"metadata": metadata,
				"similarity": float(record.get("similarity", 0.0)),
				"distance": record.get("distance"),
			}
		)

	return candidates


def hierarchical_filter(
	candidates: List[Dict[str, Any]],
	routing: Dict[str, Any],
	max_per_course_part: int = 5,
	default_top_k: int = 40,
) -> List[Dict[str, Any]]:
	if not candidates:
		return []

	sorted_candidates = sorted(candidates, key=lambda item: item.get("similarity", 0.0), reverse=True)

	routed_courses = set(routing.get("courses", []))
	routed_parts = routing.get("parts", {}) if isinstance(routing.get("parts", {}), dict) else {}

	if not routed_courses:
		return sorted_candidates[:default_top_k]

	filtered: List[Dict[str, Any]] = []
	pair_counts: Dict[Tuple[str, str], int] = {}

	for item in sorted_candidates:
		metadata: Dict[str, Any] = _as_metadata_dict(item.get("metadata"))
		course = _normalize_text(metadata.get("course", ""))
		part = _normalize_text(metadata.get("part", ""))

		if course not in routed_courses:
			continue

		allowed_parts = routed_parts.get(course, [])
		if allowed_parts and part and part not in allowed_parts and not _is_generic_part(part):
			continue

		key = (course, part)
		pair_counts[key] = pair_counts.get(key, 0)
		if pair_counts[key] >= max_per_course_part:
			continue

		pair_counts[key] += 1
		filtered.append(item)

		if len(filtered) >= default_top_k:
			break

	return filtered if filtered else sorted_candidates[:default_top_k]


def get_cross_encoder(reranker_models: Optional[List[str]] = None) -> Optional[CrossEncoder]:
	global _cross_encoder_reranker, _cross_encoder_disabled

	if _cross_encoder_disabled:
		return None
	if _cross_encoder_reranker is not None:
		return _cross_encoder_reranker

	models = reranker_models or DEFAULT_RERANKER_MODELS
	hf_token = _load_hf_token_from_env()
	last_error: Optional[Exception] = None

	for model_name in models:
		try:
			kwargs: Dict[str, Any] = {}
			if hf_token:
				kwargs["token"] = hf_token
			_cross_encoder_reranker = CrossEncoder(model_name, **kwargs)
			return _cross_encoder_reranker
		except Exception as exc:
			last_error = exc

	print(f"[WARN] CrossEncoder disabled; could not load models: {models}. Last error: {last_error}")
	_cross_encoder_disabled = True
	return None


def neural_rerank(
	query: str,
	candidates: List[Dict[str, Any]],
	top_k: int = 20,
	pool_size: int = 80,
	max_doc_chars: int = 1200,
) -> List[Dict[str, Any]]:
	if not candidates:
		return []

	pool = sorted(candidates, key=lambda item: item.get("similarity", 0.0), reverse=True)[:pool_size]
	reranker = get_cross_encoder()

	if reranker is None:
		return pool[:top_k]

	pairs: List[List[str]] = []
	for item in pool:
		doc = _normalize_text(item.get("text", ""))[:max_doc_chars]
		pairs.append([query, doc])

	try:
		scores = reranker.predict(pairs, show_progress_bar=False)
	except TypeError:
		scores = reranker.predict(pairs)

	reranked: List[Dict[str, Any]] = []
	for item, score in zip(pool, scores):
		enriched = dict(item)
		enriched["rerank_score"] = float(score)
		reranked.append(enriched)

	reranked.sort(
		key=lambda item: (
			item.get("rerank_score", 0.0),
			item.get("similarity", 0.0),
		),
		reverse=True,
	)
	return reranked[:top_k]


def _estimate_tokens(text: str) -> int:
	return max(1, len(text) // 4)


def _safe_json_array(payload: str) -> List[str]:
	try:
		parsed = json.loads(payload)
	except Exception:
		return []
	if not isinstance(parsed, list):
		return []
	return [str(item).strip() for item in parsed if str(item).strip()]


def _llm_compress_context_items(
	question: str,
	contexts: List[Dict[str, Any]],
	llm_model: str,
	per_item_char_limit: int,
) -> List[Dict[str, Any]]:
	compressed: List[Dict[str, Any]] = []

	for item in contexts:
		source_text = _normalize_text(item.get("text", ""))
		if not source_text:
			continue

		prompt = (
			"Compress this context chunk while keeping only facts useful to answer the question. "
			"Return only the compressed text.\n\n"
			f"Question: {question}\n\n"
			f"Context:\n{source_text[:5000]}"
		)
		summary = llm_response(prompt, llm_model=llm_model)
		summary = summary.strip()

		if not summary:
			summary = source_text[:per_item_char_limit]
		if len(summary) > per_item_char_limit:
			summary = summary[:per_item_char_limit]

		new_item = dict(item)
		new_item["text"] = summary
		new_item["compressed"] = True
		compressed.append(new_item)

	return compressed


def context_compression(
	question: str,
	contexts: List[Dict[str, Any]],
	token_threshold: int = 2200,
	llm_model: str = DEFAULT_LLM_MODEL,
) -> List[Dict[str, Any]]:
	if not contexts:
		return []

	total_tokens = sum(_estimate_tokens(str(item.get("text", ""))) for item in contexts)
	if total_tokens <= token_threshold:
		return contexts

	target_tokens_per_item = max(80, token_threshold // max(1, len(contexts)))
	per_item_char_limit = target_tokens_per_item * 4
	compressed = _llm_compress_context_items(
		question=question,
		contexts=contexts,
		llm_model=llm_model,
		per_item_char_limit=per_item_char_limit,
	)

	return compressed if compressed else contexts


def _log_context_snapshot(label: str, items: List[Dict[str, Any]], max_items: int = 5) -> None:
	print(f"[DEBUG] {label}: total={len(items)}")
	for idx, item in enumerate(items[:max_items], start=1):
		metadata: Dict[str, Any] = _as_metadata_dict(item.get("metadata"))
		course = metadata.get("course", "")
		part = metadata.get("part", "")
		score = item.get("rerank_score")
		if score is None:
			score = item.get("similarity")
		snippet = _normalize_text(item.get("text", ""))[:180]
		print(
			f"  - #{idx} course={course} | part={part} | score={score} | "
			f"chars={len(str(item.get('text', '')))} | text={snippet}"
		)


def _format_context_for_prompt(contexts: List[Dict[str, Any]]) -> str:
	lines: List[str] = []
	for idx, item in enumerate(contexts, start=1):
		metadata: Dict[str, Any] = _as_metadata_dict(item.get("metadata"))
		course = metadata.get("course", "")
		part = metadata.get("part", "")
		source = metadata.get("source", "")
		page_start = metadata.get("page_start")
		page_end = metadata.get("page_end")

		header = (
			f"[{idx}] course={course}; part={part}; source={source}; "
			f"pages={page_start}-{page_end}"
		)
		content = str(item.get("text", "")).strip()
		lines.append(f"{header}\n{content}")

	return "\n\n".join(lines)


def _build_answer_prompt(
	question: str,
	contexts: List[Dict[str, Any]],
	user_id: Optional[str],
	request_date: Optional[str],
) -> str:
	context_block = _format_context_for_prompt(contexts)
	metadata_line = f"UserId: {user_id or ''} | Date: {request_date or ''}".strip()

	return (
		"You are a helpful learning assistant. Use only the provided context to answer. "
		"If context is insufficient, say what is missing.\n\n"
		f"Request metadata: {metadata_line}\n\n"
		f"Question:\n{question}\n\n"
		f"Context:\n{context_block}\n\n"
		"Answer clearly and include practical steps if relevant."
	)


def _build_sources(contexts: List[Dict[str, Any]], max_sources: int = 8) -> List[Dict[str, Any]]:
	sources: List[Dict[str, Any]] = []
	seen = set()

	for item in contexts:
		metadata: Dict[str, Any] = _as_metadata_dict(item.get("metadata"))
		key = (
			_normalize_text(metadata.get("source", "")),
			_normalize_text(metadata.get("course", "")),
			_normalize_text(metadata.get("part", "")),
			metadata.get("page_start"),
			metadata.get("page_end"),
		)

		if key in seen:
			continue
		seen.add(key)

		score = item.get("rerank_score")
		if score is None:
			score = item.get("similarity")

		sources.append(
			{
				"source": metadata.get("source"),
				"course": metadata.get("course"),
				"part": metadata.get("part"),
				"page_start": metadata.get("page_start"),
				"page_end": metadata.get("page_end"),
				"score": score,
			}
		)

		if len(sources) >= max_sources:
			break

	return sources


def _prepare_answer_payload(
	question: str,
	user_id: Optional[str],
	request_date: Optional[str],
	n_courses: int,
	vector_k: int,
	filtered_k: int,
	rerank_pool_size: int,
	rerank_top_k: int,
	compression_token_threshold: int,
	llm_model: str,
	vector_db_path: Optional[str | Path],
	debug_log_context: bool,
	debug_log_max_items: int,
) -> Dict[str, Any]:
	query = _normalize_text(question)
	if not query:
		return {
			"error": "Question is empty.",
			"answer": "",
			"sources": [],
			"routing": {"courses": [], "parts": {}},
			"retrieval_counts": {
				"vector_candidates": 0,
				"after_hierarchical_filter": 0,
				"after_neural_rerank": 0,
				"after_context_compression": 0,
			},
		}

	build_vector_router_indexes(refresh=False, vector_db_path=vector_db_path)

	routing = route_query_to_course_part(
		query=query,
		n_courses=n_courses,
		vector_db_path=vector_db_path,
	)

	candidates = vector_search(
		query=query,
		n_results=vector_k,
		vector_db_path=vector_db_path,
	)

	filtered = hierarchical_filter(
		candidates=candidates,
		routing=routing,
		default_top_k=filtered_k,
	)

	if debug_log_context:
		_log_context_snapshot("Context before reranking", filtered, max_items=debug_log_max_items)

	reranked = neural_rerank(
		query=query,
		candidates=filtered,
		top_k=rerank_top_k,
		pool_size=rerank_pool_size,
	)

	if debug_log_context:
		_log_context_snapshot("Context after reranking", reranked, max_items=debug_log_max_items)

	compressed = context_compression(
		question=query,
		contexts=reranked,
		token_threshold=compression_token_threshold,
		llm_model=llm_model,
	)

	prompt = _build_answer_prompt(
		question=query,
		contexts=compressed,
		user_id=user_id,
		request_date=request_date,
	)
	sources = _build_sources(compressed)

	return {
		"error": None,
		"query": query,
		"routing": routing,
		"sources": sources,
		"prompt": prompt,
		"user_id": user_id,
		"date": request_date,
		"retrieval_counts": {
			"vector_candidates": len(candidates),
			"after_hierarchical_filter": len(filtered),
			"after_neural_rerank": len(reranked),
			"after_context_compression": len(compressed),
		},
	}


def generate_answer_with_sources(
	question: str,
	user_id: Optional[str] = None,
	request_date: Optional[str] = None,
	n_courses: int = 2,
	vector_k: int = 120,
	filtered_k: int = 40,
	rerank_pool_size: int = 80,
	rerank_top_k: int = 20,
	compression_token_threshold: int = 2200,
	llm_model: str = DEFAULT_LLM_MODEL,
	vector_db_path: Optional[str | Path] = None,
	debug_log_context: bool = False,
	debug_log_max_items: int = 5,
) -> Dict[str, Any]:
	payload = _prepare_answer_payload(
		question=question,
		user_id=user_id,
		request_date=request_date,
		n_courses=n_courses,
		vector_k=vector_k,
		filtered_k=filtered_k,
		rerank_pool_size=rerank_pool_size,
		rerank_top_k=rerank_top_k,
		compression_token_threshold=compression_token_threshold,
		llm_model=llm_model,
		vector_db_path=vector_db_path,
		debug_log_context=debug_log_context,
		debug_log_max_items=debug_log_max_items,
	)

	if payload.get("error"):
		return {
			"answer": "",
			"sources": payload.get("sources", []),
			"routing": payload.get("routing", {"courses": [], "parts": {}}),
			"error": payload.get("error"),
		}

	answer = llm_response(str(payload.get("prompt", "")), llm_model=llm_model)

	return {
		"answer": answer,
		"sources": payload.get("sources", []),
		"routing": payload.get("routing", {}),
		"user_id": user_id,
		"date": request_date,
		"retrieval_counts": payload.get("retrieval_counts", {}),
	}


def generate_answer_with_sources_stream(
	question: str,
	user_id: Optional[str] = None,
	request_date: Optional[str] = None,
	n_courses: int = 2,
	vector_k: int = 120,
	filtered_k: int = 40,
	rerank_pool_size: int = 80,
	rerank_top_k: int = 20,
	compression_token_threshold: int = 2200,
	llm_model: str = DEFAULT_LLM_MODEL,
	vector_db_path: Optional[str | Path] = None,
	debug_log_context: bool = False,
	debug_log_max_items: int = 5,
) -> Iterator[Dict[str, Any]]:
	payload = _prepare_answer_payload(
		question=question,
		user_id=user_id,
		request_date=request_date,
		n_courses=n_courses,
		vector_k=vector_k,
		filtered_k=filtered_k,
		rerank_pool_size=rerank_pool_size,
		rerank_top_k=rerank_top_k,
		compression_token_threshold=compression_token_threshold,
		llm_model=llm_model,
		vector_db_path=vector_db_path,
		debug_log_context=debug_log_context,
		debug_log_max_items=debug_log_max_items,
	)

	if payload.get("error"):
		yield {
			"type": "error",
			"error": payload.get("error"),
		}
		return

	yield {
		"type": "meta",
		"user_id": user_id,
		"date": request_date,
		"routing": payload.get("routing", {}),
		"sources": payload.get("sources", []),
		"retrieval_counts": payload.get("retrieval_counts", {}),
	}

	stream_had_output = False
	for token in llm_response_stream(str(payload.get("prompt", "")), llm_model=llm_model):
		stream_had_output = True
		yield {
			"type": "token",
			"token": token,
		}

	if not stream_had_output:
		yield {
			"type": "token",
			"token": "",
		}

	yield {
		"type": "done",
	}

