from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, AsyncIterator

from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from sentence_transformers import CrossEncoder

from .memory import prepare_memory_summary, save_memory_turn
from .retriever import retrieve_context


LLM_MODEL = "llama3.1"
EMBEDDING_MODEL = "nomic-embed-text"
RERANKER_MODEL = "BAAI/bge-reranker-base"
EMBEDDING_URL = "http://localhost:11434/api/embeddings"

TOP_K_COURSES = 3
TOP_K_PARTS = 8
TOP_K_CHUNKS = 20
TOP_K_FINAL = 5
CONTEXT_TOKEN_LIMIT = 5000

MEMORY_MAX_TOKEN_LIMIT = 1200
MEMORY_SUMMARY_TOKEN_LIMIT = 180


def _ollama_token_ids(text: str) -> list[int]:
	"""Return stable pseudo token ids for local token budgeting.

	ChatOllama doesn't expose a model tokenizer through LangChain, so we provide
	a lightweight tokenizer approximation to avoid GPT-2 fallback warnings.
	"""
	pieces = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
	return list(range(len(pieces)))


def create_collections_client(persist_dir: Path) -> Any:
	"""Create Chroma client for persisted vector DB."""
	return PersistentClient(path=str(persist_dir))


def get_collections(
	persist_dir: Path,
	embedding_model: str = EMBEDDING_MODEL,
	embedding_url: str = EMBEDDING_URL,
) -> dict[str, Any]:
	"""Get or create the three retrieval collections."""
	embedding_fn = embedding_functions.OllamaEmbeddingFunction(
		model_name=embedding_model,
		url=embedding_url,
	)
	client = create_collections_client(persist_dir)

	return {
		"course_collection": client.get_or_create_collection(
			name="prototype_courses",
			embedding_function=embedding_fn,
		),
		"part_collection": client.get_or_create_collection(
			name="prototype_parts",
			embedding_function=embedding_fn,
		),
		"chunk_collection": client.get_or_create_collection(
			name="prototype_chunks",
			embedding_function=embedding_fn,
		),
	}


def build_generation_chain(chat_llm: ChatOllama):
	"""Create the final generation chain."""
	prompt_template = ChatPromptTemplate.from_template(
		"""
You are a Data Science course assistant. You answer student questions using ONLY the provided course context.

INSTRUCTIONS:
- Give detailed, structured, educational answers.
- Use only information from the context.
- Do not hallucinate or add external knowledge.
- If the answer is not in the context, say:
  "The answer is not available in the provided course material."
- If the question is not related to the course, say:
  "This question is outside the scope of the course material."
- If the question is unclear, ask for clarification.
- Do not repeat the context.
- Keep formulas exactly as written in the context.
- Cite sources like [1], [2].
- Use the conversation summary for continuity.
- Always end your answer with a related follow-up question to help the student continue learning.

INPUTS:
Conversation Summary:
{memory_summary}

Context:
{context}

Question:
{question}
""".strip()
	)

	return prompt_template | chat_llm | StrOutputParser()


def _parse_guard_response(raw_output: str) -> dict[str, str]:
	"""Parse guard LLM output and return a normalized guard decision."""
	default_message = "Your question is broad or ambiguous. Please narrow it to a specific topic, section, or example."
	cleaned = raw_output.strip()
	cleaned = re.sub(r"^```(?:json)?\\s*", "", cleaned, flags=re.IGNORECASE)
	cleaned = re.sub(r"\\s*```$", "", cleaned)

	try:
		payload = json.loads(cleaned)
		status = str(payload.get("status", "")).strip().lower()
		if status == "ok":
			return {"status": "ok"}
		if status in {"needs_clarification", "clarify", "ambiguous", "stop"}:
			message = str(payload.get("message", "")).strip()
			return {"status": "stop", "message": message or default_message}
	except json.JSONDecodeError:
		pass

	lowered = cleaned.lower()
	if "needs_clarification" in lowered or "clarify" in lowered or "ambiguous" in lowered:
		return {"status": "stop", "message": default_message}
	if '"status"' in lowered and '"ok"' in lowered:
		return {"status": "ok"}

	# Default open: avoid rejecting valid short prompts when formatting drifts.
	return {"status": "ok"}


def guard_question(question: str, guard_llm: ChatOllama) -> dict[str, str]:
	"""Use the model to decide if the question needs clarification."""
	guard_prompt = f"""
You classify whether a user question is clear enough for retrieval in a data-science learning assistant.

Return JSON only with one of these exact formats:
{{"status":"ok"}}
{{"status":"needs_clarification","message":"<one short clarifying request>"}}

Decision rule:
- Choose "ok" if the question has a concrete topic and can reasonably be answered, even when short.
- Choose "needs_clarification" only when user intent is genuinely unclear or too broad to answer usefully.

Question:
{question}
""".strip()

	response = guard_llm.invoke(guard_prompt)
	decision = _parse_guard_response(str(response.content))
	if decision.get("status") == "ok":
		return {"status": "ok", "question": question}

	return {
		"status": "stop",
		"message": decision.get(
			"message",
			"Your question is broad or ambiguous. Please narrow it to a specific topic, section, or example.",
		),
	}


async def generate_stream(
	question: str,
	session_id: str = "default",
	persist_dir: Path | None = None,
	llm_model: str = LLM_MODEL,
	reranker_model: str = RERANKER_MODEL,
) -> AsyncIterator[str]:
	"""Stream generated answer tokens for the backend API."""
	guard_llm = ChatOllama(
		model=llm_model,
		temperature=0.0,
		verbose=True,
		custom_get_token_ids=_ollama_token_ids,
	)
	guarded = guard_question(question=question, guard_llm=guard_llm)
	if guarded.get("status") != "ok":
		yield guarded.get(
			"message",
			"Your question is broad or ambiguous. Please narrow it to a specific topic, section, or example.",
		)
		return

	if persist_dir is None:
		backend_dir = Path(__file__).resolve().parent.parent
		persist_dir = backend_dir / "vectordb" / "chroma_db"

	chat_llm = ChatOllama(
		model=llm_model,
		temperature=0.0,
		verbose=True,
		custom_get_token_ids=_ollama_token_ids,
	)
	memory_llm = ChatOllama(
		model=llm_model,
		temperature=0.0,
		verbose=True,
		custom_get_token_ids=_ollama_token_ids,
	)
	reranker = CrossEncoder(reranker_model)

	conversation_memory, memory_summary = prepare_memory_summary(
		session_id=session_id,
		memory_llm=memory_llm,
		summary_llm=memory_llm,
		max_token_limit=MEMORY_MAX_TOKEN_LIMIT,
		summary_token_limit=MEMORY_SUMMARY_TOKEN_LIMIT,
	)

	collections = get_collections(persist_dir=persist_dir)
	retrieved = retrieve_context(
		question=question,
		course_collection=collections["course_collection"],
		part_collection=collections["part_collection"],
		chunk_collection=collections["chunk_collection"],
		reranker=reranker,
		chat_llm=chat_llm,
		top_k_courses=TOP_K_COURSES,
		top_k_parts=TOP_K_PARTS,
		top_k_chunks=TOP_K_CHUNKS,
		top_k_final=TOP_K_FINAL,
		context_token_limit=CONTEXT_TOKEN_LIMIT,
	)

	if retrieved.get("status") != "ok":
		stop_answer = retrieved.get("message", "I do not have enough information.")
		yield stop_answer
		return

	generation_chain = build_generation_chain(chat_llm)
	chain_input = {
		"memory_summary": memory_summary,
		"question": question,
		"context": retrieved["context"],
	}

	full_answer = ""
	async for chunk in generation_chain.astream(chain_input):
		token = str(chunk)
		full_answer += token
		yield token

	save_memory_turn(conversation_memory=conversation_memory, question=question, answer=full_answer)
