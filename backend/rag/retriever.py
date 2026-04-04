from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document


def query_collection(
	collection: Any,
	question: str,
	n_results: int,
	where: dict[str, Any] | None = None,
) -> list[Document]:
	"""Query a Chroma collection and map results to LangChain Documents."""
	query_kwargs: dict[str, Any] = {
		"query_texts": [question],
		"n_results": n_results,
	}
	if where is not None:
		query_kwargs["where"] = where

	try:
		result = collection.query(**query_kwargs)
	except Exception:
		return []
	documents = result.get("documents", [[]])[0]
	metadatas = result.get("metadatas", [[]])[0]
	return [
		Document(page_content=doc_text, metadata=metadata or {})
		for doc_text, metadata in zip(documents, metadatas)
	]


def normalize_retrieval_query(question: str) -> str:
	"""Normalize retrieval query without stripping punctuation."""
	normalized = question.lower().replace("\r\n", "\n").replace("\r", "\n")
	normalized = re.sub(r"\s*\n+\s*", " ", normalized)
	normalized = re.sub(r"\s+", " ", normalized).strip()
	return normalized or question.strip()


def rerank_documents(
	question: str,
	documents: list[Document],
	reranker: Any,
	top_k: int = 5,
) -> list[Document]:
	"""Rerank with bge-reranker-base and keep top-k."""
	if not documents:
		return []

	pairs = [[question, doc.page_content] for doc in documents]
	scores = reranker.predict(pairs)
	ordered_indices = sorted(
		range(len(documents)),
		key=lambda idx: float(scores[idx]),
		reverse=True,
	)
	return [documents[idx] for idx in ordered_indices[:top_k]]


def approximate_tokens(text: str) -> int:
	"""Token approximation using whitespace-separated units."""
	return len(text.split())


def format_context_block(doc: Document, index: int, body: str) -> str:
	"""Format one context block with source metadata."""
	meta = doc.metadata
	return (
		f"[{index}] course={meta['course']} | section={meta['section']} | "
		f"source={meta['source_file']} | chunk_id={meta['chunk_id']}\n{body}"
	)


def summarize_chunk_with_llm(question: str, doc: Document, index: int, chat_llm: Any) -> str:
	"""Summarize contextual ideas only while preserving formulas and algorithms."""
	prompt = f"""
You are compressing a retrieved context chunk for a RAG system.

Rules:
1. Preserve every formula, equation, and mathematical expression exactly as written.
2. Preserve every algorithm step, pseudocode, and code instruction exactly as written.
3. Summarize only explanatory/contextual ideas around formulas and algorithms.
4. Do not invent facts and do not add external knowledge.
5. Keep output concise and faithful to the chunk.

Question:
{question}

Chunk metadata:
course={doc.metadata['course']}, section={doc.metadata['section']}, source={doc.metadata['source_file']}, chunk_id={doc.metadata['chunk_id']}

Chunk text:
{doc.page_content}
""".strip()

	response = chat_llm.invoke(prompt)
	summary = getattr(response, "content", str(response)).strip()
	return format_context_block(doc, index, summary)


def compress_context(
	question: str,
	documents: list[Document],
	chat_llm: Any,
	token_limit: int = 5000,
) -> str:
	"""Summarize chunks with LLM when context exceeds token limit."""
	full_blocks = [
		format_context_block(doc, i, doc.page_content)
		for i, doc in enumerate(documents, start=1)
	]
	full_context = "\n\n".join(full_blocks)

	if approximate_tokens(full_context) <= token_limit:
		return full_context

	summarized_blocks = [
		summarize_chunk_with_llm(question, doc, i, chat_llm)
		for i, doc in enumerate(documents, start=1)
	]
	return "\n\n".join(summarized_blocks)


def retrieve_context(
	question: str,
	course_collection: Any,
	part_collection: Any,
	chunk_collection: Any,
	reranker: Any,
	chat_llm: Any,
	top_k_courses: int = 3,
	top_k_parts: int = 8,
	top_k_chunks: int = 20,
	top_k_final: int = 5,
	context_token_limit: int = 5000,
) -> dict[str, Any]:
	"""Course -> parts -> chunks retrieval, then rerank and compress."""
	retrieval_question = normalize_retrieval_query(question)

	# Anchor routing on chunk-level semantic hits to avoid course-name misrouting.
	global_chunk_hits = query_collection(
		chunk_collection,
		retrieval_question,
		top_k_chunks,
	)
	if not global_chunk_hits:
		return {"status": "stop", "message": "I do not have enough retrieved context to answer this reliably."}

	top_global_course = str(global_chunk_hits[0].metadata.get("course", "")).strip()
	selected_course = top_global_course
	if not selected_course:
		course_hits = query_collection(course_collection, retrieval_question, top_k_courses)
		selected_course = course_hits[0].page_content if course_hits else ""
	if not selected_course:
		return {"status": "stop", "message": "No indexed course matched this question."}

	part_hits = query_collection(
		part_collection,
		retrieval_question,
		top_k_parts,
		where={"course": selected_course},
	)
	selected_parts = [doc.page_content for doc in part_hits]

	chunk_hits = [doc for doc in global_chunk_hits if doc.metadata.get("course") == selected_course]
	if not chunk_hits:
		chunk_hits = query_collection(
			chunk_collection,
			retrieval_question,
			top_k_chunks,
			where={"course": selected_course},
		)

	if selected_parts:
		part_set = set(selected_parts)
		part_filtered = [doc for doc in chunk_hits if doc.metadata.get("section") in part_set]
		if part_filtered:
			chunk_hits = part_filtered

	ranked_docs = rerank_documents(question, chunk_hits, reranker=reranker, top_k=top_k_final)
	if not ranked_docs:
		return {"status": "stop", "message": "I do not have enough retrieved context to answer this reliably."}

	context = compress_context(question, ranked_docs, chat_llm=chat_llm, token_limit=context_token_limit)
	return {
		"status": "ok",
		"question": question,
		"selected_course": selected_course,
		"selected_parts": selected_parts,
		"ranked_docs": ranked_docs,
		"context": context,
	}
