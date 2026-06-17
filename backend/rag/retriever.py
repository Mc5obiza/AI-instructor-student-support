from __future__ import annotations

import re
from collections import Counter, defaultdict
from math import log
from typing import Any

from langchain_core.documents import Document
from langchain_core.tools import tool


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


ENGLISH_STOPWORDS = {
	"a",
	"an",
	"and",
	"are",
	"as",
	"at",
	"be",
	"by",
	"for",
	"from",
	"has",
	"he",
	"in",
	"is",
	"it",
	"its",
	"of",
	"on",
	"that",
	"the",
	"to",
	"was",
	"were",
	"will",
	"with",
}


def normalize_retrieval_query(question: str) -> str:
	"""Normalize retrieval query without stripping punctuation."""
	normalized = question.lower().replace("\r\n", "\n").replace("\r", "\n")
	normalized = re.sub(r"\s*\n+\s*", " ", normalized)
	normalized = re.sub(r"\s+", " ", normalized).strip()
	return normalized or question.strip()


def tokenize_text(text: str, remove_stopwords: bool = True) -> list[str]:
	"""Tokenize text for BM25 scoring and optionally remove English stopwords."""
	tokens = re.findall(r"\b\w+\b", text.lower())
	if remove_stopwords:
		return [token for token in tokens if token not in ENGLISH_STOPWORDS]
	return tokens


def document_key(doc: Document) -> str:
	"""Build a stable key for a retrieved document across rankings."""
	meta = doc.metadata or {}
	return "|".join(
		[
			str(meta.get("course", "")),
			str(meta.get("section", "")),
			str(meta.get("source_file", "")),
			str(meta.get("chunk_id", "")),
			str(doc.page_content[:128]),
		]
	)


def get_collection_documents(collection: Any, where: dict[str, Any] | None = None) -> list[Document]:
	"""Load all documents for a collection, optionally filtering by metadata."""
	try:
		result = collection.get(include=["documents", "metadatas"])
		documents = result.get("documents", [])
		metadatas = result.get("metadatas", [])
		if documents and isinstance(documents[0], list):
			documents = documents[0]
		if metadatas and isinstance(metadatas[0], list):
			metadatas = metadatas[0]
		loaded = [
			Document(page_content=str(doc_text or ""), metadata=metadata or {})
			for doc_text, metadata in zip(documents, metadatas)
		]
		if where is None:
			return loaded
		return [doc for doc in loaded if all(doc.metadata.get(k) == v for k, v in where.items())]
	except Exception:
		return []


def compute_bm25_idf(documents: list[list[str]]) -> dict[str, float]:
	"""Compute BM25 inverse document frequency values."""
	n_documents = len(documents)
	df: dict[str, int] = {}
	for doc_tokens in documents:
		for token in set(doc_tokens):
			df[token] = df.get(token, 0) + 1
	return {
		token: log((n_documents - freq + 0.5) / (freq + 0.5) + 1)
		for token, freq in df.items()
	}


def bm25_score(
	query_terms: list[str],
	doc_terms: list[str],
	idf: dict[str, float],
	avgdl: float,
	k1: float = 1.5,
	b: float = 0.75,
) -> float:
	"""Score a document using BM25 for the given query terms."""
	doc_freq = Counter(doc_terms)
	dl = len(doc_terms)
	if dl == 0:
		return 0.0
	score = 0.0
	for term in query_terms:
		if term not in idf:
			continue
		freq = doc_freq.get(term, 0)
		if freq <= 0:
			continue
		score += idf[term] * ((freq * (k1 + 1)) / (freq + k1 * (1 - b + b * dl / avgdl)))
	return score


def bm25_retrieve_documents(
	collection: Any,
	question: str,
	n_results: int,
	where: dict[str, Any] | None = None,
) -> list[Document]:
	"""Retrieve documents using BM25 ranking from a Chroma collection."""
	candidates = get_collection_documents(collection, where=where)
	if not candidates:
		return []
	query_terms = tokenize_text(question)
	if not query_terms:
		return candidates[:n_results]
	doc_tokens = [tokenize_text(doc.page_content) for doc in candidates]
	idf = compute_bm25_idf(doc_tokens)
	avgdl = sum(len(tokens) for tokens in doc_tokens) / max(1, len(doc_tokens))
	scores = [bm25_score(query_terms, tokens, idf, avgdl) for tokens in doc_tokens]
	indexed = sorted(
		zip(scores, candidates),
		key=lambda pair: pair[0],
		reverse=True,
	)
	return [doc for score, doc in indexed][:n_results]


def fuse_rankings_by_rrf(
	rankings: list[list[Document]],
	top_k: int,
	constant: int = 60,
) -> list[Document]:
	"""Fuse multiple ranked lists using Reciprocal Rank Fusion."""
	scores: dict[str, float] = defaultdict(float)
	documents: dict[str, Document] = {}
	for ranking in rankings:
		for rank, doc in enumerate(ranking, start=1):
			key = document_key(doc)
			documents.setdefault(key, doc)
			scores[key] += 1.0 / (constant + rank)
	ordered_keys = sorted(scores, key=lambda key: (-scores[key], key))
	return [documents[key] for key in ordered_keys[:top_k]]


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
		course_hits = bm25_retrieve_documents(course_collection, retrieval_question, top_k_courses)
		if not course_hits:
			course_hits = query_collection(course_collection, retrieval_question, top_k_courses)
		selected_course = course_hits[0].page_content if course_hits else ""
	if not selected_course:
		return {"status": "stop", "message": "No indexed course matched this question."}

	part_hits = bm25_retrieve_documents(
		part_collection,
		retrieval_question,
		top_k_parts,
		where={"course": selected_course},
	)
	if not part_hits:
		part_hits = query_collection(
			part_collection,
			retrieval_question,
			top_k_parts,
			where={"course": selected_course},
		)
	selected_parts = [doc.page_content for doc in part_hits]

	chunk_semantic_hits = query_collection(
		chunk_collection,
		retrieval_question,
		top_k_chunks,
		where={"course": selected_course},
	)
	chunk_bm25_hits = bm25_retrieve_documents(
		chunk_collection,
		retrieval_question,
		top_k_chunks,
		where={"course": selected_course},
	)
	chunk_hits = fuse_rankings_by_rrf([chunk_semantic_hits, chunk_bm25_hits], top_k=top_k_chunks)

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


def build_retrieval_tool(
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
) -> Any:
	"""Create a retrieval tool with bound dependencies for graph/node usage."""

	@tool("rag_retrieval_tool")
	def rag_retrieval_tool(question: str) -> dict[str, Any]:
		"""Retrieve, rerank, and compress context for one user question."""
		return retrieve_context(
			question=question,
			course_collection=course_collection,
			part_collection=part_collection,
			chunk_collection=chunk_collection,
			reranker=reranker,
			chat_llm=chat_llm,
			top_k_courses=top_k_courses,
			top_k_parts=top_k_parts,
			top_k_chunks=top_k_chunks,
			top_k_final=top_k_final,
			context_token_limit=context_token_limit,
		)

	return rag_retrieval_tool


def run_retrieval_tool_node(question: str, retrieval_tool: Any) -> dict[str, Any]:
	"""Execute a bound retrieval tool and return node-friendly state updates."""
	if not question.strip():
		return {"status": "stop", "message": "Question is required for retrieval."}

	tool_result = retrieval_tool.invoke({"question": question})
	if not isinstance(tool_result, dict):
		return {"status": "stop", "message": "Retrieval tool returned an unexpected response."}

	if tool_result.get("status") != "ok":
		return {
			"status": "stop",
			"message": tool_result.get("message", "I do not have enough retrieved context to answer this reliably."),
		}

	return {
		"status": "ok",
		"retrieved": tool_result,
		"context": str(tool_result.get("context", "")),
	}
