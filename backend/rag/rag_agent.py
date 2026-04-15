from __future__ import annotations

import json
import os
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
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
EMBEDDING_URL = os.getenv("OLLAMA_EMBEDDING_URL", f"{OLLAMA_BASE_URL}/api/embeddings")


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
- Whenever your answer includes equations, always format them as LaTeX math (not plain-text equations).
- For standalone equations, use display math blocks with $$ ... $$.
- For multiple related equations/steps, write each equation in its own separate display block.
- Do not put two equations in the same line.
- Do not combine multiple equations inside one $$ ... $$ block.
- Do not output formula lines like "m_t = ..." as plain text when LaTeX can be used.
- Cite sources like [1], [2].
- Use the conversation summary for continuity.
- First infer user intent from the current Question:
	- follow_up_answer_intent: the user is replying to the previous follow-up prompt (for example: yes/no, "go ahead", "tell me more", or a short continuation request).
	- new_question_intent: the user is asking a new standalone question.
- If intent is follow_up_answer_intent, answer the previous follow-up topic using Conversation Summary + Context, even if the current Question text is short.
- If intent is new_question_intent, answer the current Question normally.
- Do not use this prefix: "Follow-up question:".
- Always end your answer with exactly one line in this format:
	"Do u want to know about <related follow-up question>?"
- Replace <related follow-up question> with a real specific question topic; never output angle brackets.

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
	default_clarification_message = "Your question is broad or ambiguous. Please narrow it to a specific topic, section, or example."
	default_scope_message = "I am designed to assist with Data Science and course-related questions only. Please ask a question related to the course material."
	cleaned = raw_output.strip()
	cleaned = re.sub(r"^```(?:json)?\\s*", "", cleaned, flags=re.IGNORECASE)
	cleaned = re.sub(r"\\s*```$", "", cleaned)

	try:
		payload = json.loads(cleaned)
		status = str(payload.get("status", "")).strip().lower()
		if status == "ok":
			return {"status": "ok"}
		if status in {"out_of_scope", "out-of-scope", "not_related", "off_topic", "off-topic"}:
			message = str(payload.get("message", "")).strip()
			return {"status": "stop", "message": message or default_scope_message}
		if status in {"needs_clarification", "clarify", "ambiguous", "stop"}:
			message = str(payload.get("message", "")).strip()
			return {"status": "stop", "message": message or default_clarification_message}
	except json.JSONDecodeError:
		pass

	lowered = cleaned.lower()
	if "out_of_scope" in lowered or "outside the scope" in lowered or "off topic" in lowered:
		return {"status": "stop", "message": default_scope_message}
	if "needs_clarification" in lowered or "clarify" in lowered or "ambiguous" in lowered:
		return {"status": "stop", "message": default_clarification_message}
	if '"status"' in lowered and '"ok"' in lowered:
		return {"status": "ok"}

	# Default closed: fail safe to safeguard domain and avoid policy drift.
	return {"status": "stop", "message": default_scope_message}


def guard_question(question: str, guard_llm: ChatOllama) -> dict[str, str]:
	"""Use the model to enforce domain scope and question clarity."""
	guard_prompt = f"""
You classify whether a user question is both:
1) in scope for a Data Science learning assistant, and
2) clear enough for retrieval.

Return JSON only with one of these exact formats:
{{"status":"ok"}}
{{"status":"needs_clarification","message":"<one short clarifying request>"}}
{{"status":"out_of_scope","message":"I am designed to assist with Data Science and course-related questions only. Please ask a question related to the course material."}}

Decision rule:
- Choose "ok" only if the question is about Data Science, Machine Learning, AI, Python-for-data, statistics, data analysis, or your indexed course material.
- Choose "needs_clarification" only when the question is in-scope but genuinely unclear or too broad to answer usefully.
- Choose "out_of_scope" for unrelated requests (for example: personal reminders, arbitrary secret phrases, general chit-chat, non-course tasks).

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
