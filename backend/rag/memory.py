from __future__ import annotations

from typing import Any, cast

from langchain_classic.memory import ConversationSummaryBufferMemory


MEMORY_KEY = "conversation_summary"
MEMORY_MAX_TOKEN_LIMIT = 1200
MEMORY_SUMMARY_TOKEN_LIMIT = 180

_SESSION_MEMORIES: dict[str, ConversationSummaryBufferMemory] = {}


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
	memory_vars = conversation_memory.load_memory_variables({})
	history_text = str(memory_vars.get(MEMORY_KEY, "")).strip()

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
) -> None:
	"""Save one user/assistant turn in memory."""
	conversation_memory.save_context(
		{"question": question},
		{"answer": answer},
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
	memory_summary = load_memory_summary(
		conversation_memory=conversation_memory,
		llm=summary_llm,
		summary_token_limit=summary_token_limit,
	)
	return conversation_memory, memory_summary
