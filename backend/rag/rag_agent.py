from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


LLM_MODEL = "llama3.1"
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_URL = "http://localhost:11434/api/embeddings"


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


def build_generation_chain(chat_llm: ChatOllama) -> Any:
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
