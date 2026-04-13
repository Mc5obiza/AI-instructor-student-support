from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path
from typing import Any, cast

from langchain_classic.memory import ConversationSummaryBufferMemory


MEMORY_KEY = "conversation_summary"
MEMORY_MAX_TOKEN_LIMIT = 32000
MEMORY_SUMMARY_TOKEN_LIMIT = 32000
PERSISTED_SUMMARY_MAX_CHARS = 8000
PERSISTED_SUMMARY_MIN_CHARS = 400
PERSISTED_SUMMARY_RECENT_TURNS = 8

_SESSION_MEMORIES: dict[str, ConversationSummaryBufferMemory] = {}
_HYDRATED_SESSION_IDS: set[str] = set()
_SESSION_INTERFACE_CLASS: Any | None = None

_MANAGING_SESSIONS_PATH = Path(__file__).resolve().parents[1] / "managing_sessions"


def _get_session_interface_class() -> Any | None:
	"""Load session interface lazily without hard package coupling."""
	global _SESSION_INTERFACE_CLASS
	if _SESSION_INTERFACE_CLASS is not None:
		return _SESSION_INTERFACE_CLASS

	try:
		if str(_MANAGING_SESSIONS_PATH) not in sys.path:
			sys.path.insert(0, str(_MANAGING_SESSIONS_PATH))
		_SESSION_INTERFACE_CLASS = importlib.import_module("sessionInterface").SessionInterface
	except Exception:
		_SESSION_INTERFACE_CLASS = None

	return _SESSION_INTERFACE_CLASS


def _extract_memory_text(conversation_memory: ConversationSummaryBufferMemory) -> str:
	"""Read current summary/history text from conversation memory."""
	if hasattr(conversation_memory, "moving_summary_buffer"):
		buffer_text = str(getattr(conversation_memory, "moving_summary_buffer", "")).strip()
		if buffer_text:
			return buffer_text

	memory_vars = conversation_memory.load_memory_variables({})
	return str(memory_vars.get(MEMORY_KEY, "")).strip()


def _clip_text(value: str, max_chars: int = PERSISTED_SUMMARY_MAX_CHARS) -> str:
	"""Bound persisted summary length for DB storage safety."""
	text = str(value).strip()
	if len(text) <= max_chars:
		return text
	return f"{text[:max_chars].rstrip()}..."


def _extract_recent_turns_text(
	conversation_memory: ConversationSummaryBufferMemory,
	max_turns: int = PERSISTED_SUMMARY_RECENT_TURNS,
) -> str:
	"""Build a readable snapshot of recent turns from memory chat history."""
	chat_memory = getattr(conversation_memory, "chat_memory", None)
	messages = getattr(chat_memory, "messages", None)
	if not isinstance(messages, list) or not messages:
		return ""

	tail_messages = messages[-(max_turns * 2) :]
	lines: list[str] = []
	for message in tail_messages:
		content = str(getattr(message, "content", "")).strip()
		if not content:
			continue

		message_type = str(getattr(message, "type", "")).strip().lower()
		if "human" in message_type:
			role = "Human"
		elif "ai" in message_type:
			role = "AI"
		else:
			role = "Message"

		lines.append(f"{role}: {content}")

	return "\n".join(lines).strip()


def _extract_follow_up_question(answer: str) -> str:
	"""Extract the assistant follow-up question when present."""
	match = re.search(
		r"(Do\s*(?:u|you)\s+want\s+to\s+know\s+about[^\n\r?]*\?)",
		answer,
		flags=re.IGNORECASE,
	)
	if match:
		return match.group(1).strip()

	for line in reversed(str(answer).splitlines()):
		cleaned = line.strip()
		if not cleaned:
			continue
		if "follow-up question" in cleaned.lower():
			candidate = cleaned.split(":", 1)[-1].strip()
			if candidate and not candidate.endswith("?"):
				candidate = f"{candidate}?"
			return candidate

	return ""


def _generate_persisted_summary(
	conversation_memory: ConversationSummaryBufferMemory,
	question: str,
	answer: str,
) -> str:
	"""Create a rich, structured summary string suitable for DB persistence."""
	existing_summary = _extract_memory_text(conversation_memory)
	recent_turns = _extract_recent_turns_text(conversation_memory)
	follow_up_question = _extract_follow_up_question(answer)
	llm = getattr(conversation_memory, "llm", None)

	candidate = ""
	if llm is not None and hasattr(llm, "invoke"):
		summary_prompt = f"""
You are updating long-term memory for one tutoring session.

Write a detailed but concise plain-text summary with these exact sections:
Session Objective:
Key Concepts Covered:
Important Formulas/Definitions:
Learner Progress And Gaps:
Current Answer Snapshot:
Follow-up Focus:

Rules:
- Do not output dialogue transcripts unless needed for clarity.
- Preserve important technical terms and formulas exactly.
- Include concrete details, not generic phrases.
- In Follow-up Focus, include one explicit next question.
- Keep total length under {PERSISTED_SUMMARY_MAX_CHARS} characters.

Existing Persisted Summary:
{existing_summary or 'None'}

Recent Turns Snapshot:
{recent_turns or 'None'}

Latest User Question:
{question}

Latest Assistant Answer:
{answer}

Extracted Follow-up Question:
{follow_up_question or 'None'}
""".strip()

		try:
			response = llm.invoke(summary_prompt)
			candidate = str(getattr(response, "content", response)).strip()
		except Exception:
			candidate = ""

	if not candidate:
		candidate = (
			"Session Objective:\n"
			f"- {question}\n\n"
			"Current Answer Snapshot:\n"
			f"- {answer}\n\n"
			"Follow-up Focus:\n"
			f"- {follow_up_question or 'Do u want to know about a deeper part of this topic?'}"
		)

	if follow_up_question and follow_up_question.lower() not in candidate.lower():
		candidate = f"{candidate}\n\nFollow-up Focus:\n- {follow_up_question}"

	if len(candidate.strip()) < PERSISTED_SUMMARY_MIN_CHARS and recent_turns:
		candidate = f"{candidate}\n\nRecent Turns Snapshot:\n{recent_turns}"

	return _clip_text(candidate, max_chars=PERSISTED_SUMMARY_MAX_CHARS)


def _fetch_session_summary_text(session_id: str) -> str:
	"""Fetch persisted session summary text from the session table."""
	session_interface_class = _get_session_interface_class()
	if session_interface_class is None:
		return ""

	try:
		response = session_interface_class().get_session_summary(session_id=session_id)
		payload = json.loads(response)
		if str(payload.get("status", "")) != "200":
			return ""
		return str(payload.get("summary", "")).strip()
	except Exception:
		return ""


def _hydrate_memory_from_session_summary(
	session_id: str,
	conversation_memory: ConversationSummaryBufferMemory,
) -> None:
	"""Seed in-memory summary with persisted session summary once per session."""
	if session_id in _HYDRATED_SESSION_IDS:
		return

	_HYDRATED_SESSION_IDS.add(session_id)
	persisted_summary = _fetch_session_summary_text(session_id=session_id)
	if not persisted_summary:
		return

	if hasattr(conversation_memory, "moving_summary_buffer"):
		conversation_memory.moving_summary_buffer = persisted_summary


def persist_session_summary(
	session_id: str,
	conversation_memory: ConversationSummaryBufferMemory,
	question: str,
	answer: str,
) -> None:
	"""Persist current conversation summary text into session.summary."""
	summary_text = _generate_persisted_summary(
		conversation_memory=conversation_memory,
		question=question,
		answer=answer,
	)
	if not summary_text:
		return

	session_interface_class = _get_session_interface_class()
	if session_interface_class is None:
		return

	try:
		session_interface_class().set_session_summary(session_id=session_id, summary=summary_text)
	except Exception:
		# Memory persistence should never block response generation.
		return


def get_or_create_memory(
	session_id: str,
	llm: Any,
	max_token_limit: int = MEMORY_MAX_TOKEN_LIMIT,
) -> ConversationSummaryBufferMemory:
	"""Get a session memory object or create it once."""
	if session_id not in _SESSION_MEMORIES:
		memory_cls = cast(Any, ConversationSummaryBufferMemory)
		_SESSION_MEMORIES[session_id] = memory_cls(
			llm=llm,
			memory_key=MEMORY_KEY,
			max_token_limit=max_token_limit,
			return_messages=False,
			input_key="question",
			output_key="answer",
		)
	return _SESSION_MEMORIES[session_id]


def load_memory_summary(
	conversation_memory: ConversationSummaryBufferMemory,
	llm: Any,
	summary_token_limit: int = MEMORY_SUMMARY_TOKEN_LIMIT,
) -> str:
	"""Load memory and return a short summary text for prompting."""
	history_text = _extract_memory_text(conversation_memory)

	if not history_text:
		return "No prior conversation summary."

	summary_prompt = f"""
Summarize this conversation history in at most {summary_token_limit} tokens.
Preserve formulas, equations, and algorithm steps exactly as written.
Return summary text only.

History:
{history_text}
""".strip()

	summary_response = llm.invoke(summary_prompt)
	return getattr(summary_response, "content", str(summary_response)).strip()


def save_memory_turn(
	conversation_memory: ConversationSummaryBufferMemory,
	question: str,
	answer: str,
	session_id: str | None = None,
) -> None:
	"""Save one user/assistant turn in memory."""
	conversation_memory.save_context(
		{"question": question},
		{"answer": answer},
	)
	if session_id:
		persist_session_summary(
			session_id=session_id,
			conversation_memory=conversation_memory,
			question=question,
			answer=answer,
		)


def prepare_memory_summary(
	session_id: str,
	memory_llm: Any,
	summary_llm: Any,
	max_token_limit: int = MEMORY_MAX_TOKEN_LIMIT,
	summary_token_limit: int = MEMORY_SUMMARY_TOKEN_LIMIT,
) -> tuple[ConversationSummaryBufferMemory, str]:
	"""Return session memory and the summary that will be injected in the prompt."""
	conversation_memory = get_or_create_memory(
		session_id=session_id,
		llm=memory_llm,
		max_token_limit=max_token_limit,
	)
	_hydrate_memory_from_session_summary(
		session_id=session_id,
		conversation_memory=conversation_memory,
	)
	memory_summary = load_memory_summary(
		conversation_memory=conversation_memory,
		llm=summary_llm,
		summary_token_limit=summary_token_limit,
	)
	return conversation_memory, memory_summary
