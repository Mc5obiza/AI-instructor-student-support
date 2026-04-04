from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, AsyncIterator, TypedDict, cast

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from sentence_transformers import CrossEncoder

try:
	from backend.rag.code_reviewer import (
		get_code_reviewer_router_context,
		run_code_reviewer_tool_from_query,
	)
	from backend.rag.memory import prepare_memory_summary, save_memory_turn
	from backend.rag.rag_agent import build_generation_chain, get_collections, guard_question
	from backend.rag.retriever import build_retrieval_tool, run_retrieval_tool_node
except ImportError:
	from rag.code_reviewer import get_code_reviewer_router_context, run_code_reviewer_tool_from_query
	from rag.memory import prepare_memory_summary, save_memory_turn
	from rag.rag_agent import build_generation_chain, get_collections, guard_question
	from rag.retriever import build_retrieval_tool, run_retrieval_tool_node


LLM_MODEL = "llama3.1"
RERANKER_MODEL = "BAAI/bge-reranker-base"

TOP_K_COURSES = 3
TOP_K_PARTS = 8
TOP_K_CHUNKS = 20
TOP_K_FINAL = 5
CONTEXT_TOKEN_LIMIT = 5000

MEMORY_MAX_TOKEN_LIMIT = 32000
MEMORY_SUMMARY_TOKEN_LIMIT = 32000

_SESSION_LAST_ROUTE: dict[str, str] = {}


class OrchestratorState(TypedDict, total=False):
	question: str
	session_id: str
	persist_dir: Path | None
	llm_model: str
	reranker_model: str
	route: str
	status: str
	message: str
	answer: str
	chat_llm: Any
	memory_summary: str
	context: str
	conversation_memory: Any


def _ollama_token_ids(text: str) -> list[int]:
	"""Return stable pseudo token ids for local token budgeting."""
	pieces = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
	return list(range(len(pieces)))


def _parse_router_response(raw_output: str) -> str:
	"""Parse router LLM output and normalize route key."""
	cleaned = raw_output.strip()
	cleaned = re.sub(r"^```(?:json)?\\s*", "", cleaned, flags=re.IGNORECASE)
	cleaned = re.sub(r"\\s*```$", "", cleaned)

	try:
		payload = json.loads(cleaned)
		route = str(payload.get("route", "")).strip().lower()
		if route in {"rag", "rag_pipeline", "rag_agent"}:
			return "rag"
		if route in {"code", "code_reviewer", "code-review", "review"}:
			return "code_reviewer"
	except json.JSONDecodeError:
		pass

	normalized = cleaned.strip().strip('"').strip("'").lower()
	if normalized in {"code_reviewer", "code reviewer", "code-review", "code"}:
		return "code_reviewer"
	if normalized in {"rag", "rag_agent", "rag_pipeline"}:
		return "rag"

	# Fail safe to RAG if router output is malformed.
	return "rag"


def _get_previous_session_route(session_id: str) -> str:
	"""Return previous route for this session if known."""
	return _SESSION_LAST_ROUTE.get(session_id, "none")


def _set_previous_session_route(session_id: str, route: str) -> None:
	"""Persist last chosen route per session."""
	if route in {"rag", "code_reviewer"}:
		_SESSION_LAST_ROUTE[session_id] = route


def _looks_like_code_followup(question: str) -> bool:
	"""Detect follow-up prompts that refer to prior code without re-posting it."""
	lowered = question.lower()
	hints = [
		"code",
		"function",
		"argument",
		"parameter",
		"type",
		"string",
		"int",
		"float",
		"bool",
		"validate",
		"check",
		"change",
		"modify",
		"fix",
		"how can i",
	]
	return any(hint in lowered for hint in hints)


def choose_orchestration_route(
	question: str,
	llm_model: str = LLM_MODEL,
	session_id: str = "default",
) -> str:
	"""Choose route using LLM-only decision."""
	session_id = session_id.strip() or "default"
	previous_route = _get_previous_session_route(session_id)
	router_context = get_code_reviewer_router_context(session_id=session_id, llm_model=llm_model)

	router_llm = ChatOllama(
		model=llm_model,
		temperature=0.0,
		verbose=False,
		custom_get_token_ids=_ollama_token_ids,
	)
	router_prompt = f"""
You are a strict router for a learning assistant with exactly two routes:
- rag
- code_reviewer

Return JSON only in this exact schema:
{{"route":"rag"}}
or
{{"route":"code_reviewer"}}

Routing rules:
- Choose "code_reviewer" when the user asks to review, debug, fix, explain, or run code.
- Choose "code_reviewer" when the message contains code snippets in any language (Python, C, C++, Java, JavaScript, pseudocode).
- If previous route is "code_reviewer" and user asks a follow-up about modifying/checking/validating "the code" without restating it, keep route as "code_reviewer".
- Choose "rag" for normal conceptual/course questions that do not need code review.

Do not add extra keys or explanations.

Session context:
- session_id: {session_id}
- previous_route: {previous_route}
- code_reviewer_memory_summary: {router_context.get("memory_summary", "")}
- last_reviewed_code_excerpt: {router_context.get("last_code_excerpt", "")}

Question:
{question}
""".strip()

	try:
		response = router_llm.invoke(router_prompt)
		route = _parse_router_response(str(response.content))
		if previous_route == "code_reviewer" and route != "code_reviewer" and _looks_like_code_followup(question):
			return "code_reviewer"
		return route
	except Exception:
		# Keep backend available even if routing model call fails.
		return "rag"


def orchestrator_node(state: OrchestratorState) -> OrchestratorState:
	"""Choose which tool-node should handle the request."""
	question = state.get("question", "")
	session_id = state.get("session_id", "default")
	llm_model = state.get("llm_model", LLM_MODEL)
	route = choose_orchestration_route(
		question=question,
		llm_model=llm_model,
		session_id=session_id,
	)
	return {
		"status": "ok",
		"route": route,
	}


def route_after_orchestrator(state: OrchestratorState) -> str:
	"""Route key for graph conditional edges."""
	return "code_reviewer_tool" if state.get("route") == "code_reviewer" else "rag_retriever_tool"


def code_reviewer_tool_node(state: OrchestratorState) -> OrchestratorState:
	"""Call the code-review tool path from orchestrator."""
	question = state.get("question", "")
	session_id = state.get("session_id", "default")
	llm_model = state.get("llm_model", LLM_MODEL)
	code_result = run_code_reviewer_tool_from_query(
		query=question,
		llm_model=llm_model,
		session_id=session_id,
	)

	if code_result.get("status") != "ok":
		message = str(code_result.get("answer", code_result.get("message", "Code review failed."))).strip()
		return {
			"status": "stop",
			"route": "code_reviewer",
			"message": message,
		}

	return {
		"status": "ok",
		"route": "code_reviewer",
		"answer": str(code_result.get("answer", "")).strip(),
	}


def rag_retriever_tool_node(state: OrchestratorState) -> OrchestratorState:
	"""Call guard + retrieval tool path from orchestrator."""
	question = state.get("question", "")
	session_id = state.get("session_id", "default")
	llm_model = state.get("llm_model", LLM_MODEL)
	reranker_model = state.get("reranker_model", RERANKER_MODEL)
	persist_dir = state.get("persist_dir")

	if persist_dir is None:
		backend_dir = Path(__file__).resolve().parent.parent
		persist_dir = backend_dir / "vectordb" / "chroma_db"

	chat_llm = ChatOllama(
		model=llm_model,
		temperature=0.0,
		verbose=True,
		custom_get_token_ids=_ollama_token_ids,
	)
	guard_llm = ChatOllama(
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

	guarded = guard_question(question=question, guard_llm=guard_llm)
	if guarded.get("status") != "ok":
		return {
			"status": "stop",
			"route": "rag",
			"message": str(
				guarded.get(
					"message",
					"Your question is broad or ambiguous. Please narrow it to a specific topic, section, or example.",
				)
			),
		}

	conversation_memory, memory_summary = prepare_memory_summary(
		session_id=session_id,
		memory_llm=memory_llm,
		summary_llm=memory_llm,
		max_token_limit=MEMORY_MAX_TOKEN_LIMIT,
		summary_token_limit=MEMORY_SUMMARY_TOKEN_LIMIT,
	)

	reranker = CrossEncoder(reranker_model)
	collections = get_collections(persist_dir=persist_dir)
	retrieval_tool = build_retrieval_tool(
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
	retrieval_update = run_retrieval_tool_node(question=question, retrieval_tool=retrieval_tool)
	if retrieval_update.get("status") != "ok":
		return {
			"status": "stop",
			"route": "rag",
			"message": str(
				retrieval_update.get(
					"message",
					"I do not have enough retrieved context to answer this reliably.",
				)
			),
		}

	return {
		"status": "ok",
		"route": "rag",
		"chat_llm": chat_llm,
		"memory_summary": memory_summary,
		"context": str(retrieval_update.get("context", "")),
		"conversation_memory": conversation_memory,
	}


def build_orchestrator_graph() -> Any:
	"""Build unified graph: START -> orchestrator -> conditional tool node."""
	builder = StateGraph(OrchestratorState)
	builder.add_node("orchestrator", orchestrator_node)
	builder.add_node("code_reviewer_tool", code_reviewer_tool_node)
	builder.add_node("rag_retriever_tool", rag_retriever_tool_node)

	builder.add_edge(START, "orchestrator")
	builder.add_conditional_edges(
		"orchestrator",
		route_after_orchestrator,
		{
			"code_reviewer_tool": "code_reviewer_tool",
			"rag_retriever_tool": "rag_retriever_tool",
		},
	)
	builder.add_edge("code_reviewer_tool", END)
	builder.add_edge("rag_retriever_tool", END)
	return builder.compile()


_ORCHESTRATOR_GRAPH: Any = None


def get_orchestrator_graph() -> Any:
	"""Return cached unified orchestrator graph."""
	global _ORCHESTRATOR_GRAPH
	if _ORCHESTRATOR_GRAPH is None:
		_ORCHESTRATOR_GRAPH = build_orchestrator_graph()
	return _ORCHESTRATOR_GRAPH


def compile_prompt_with_orchestrator(
	question: str,
	session_id: str = "default",
	persist_dir: Path | None = None,
	llm_model: str = LLM_MODEL,
	reranker_model: str = RERANKER_MODEL,
) -> dict[str, Any]:
	"""Run unified graph and normalize output for stream rendering."""
	graph = get_orchestrator_graph()
	initial_state: OrchestratorState = {
		"question": question,
		"session_id": session_id,
		"persist_dir": persist_dir,
		"llm_model": llm_model,
		"reranker_model": reranker_model,
	}
	final_state = cast(OrchestratorState, graph.invoke(initial_state))

	if final_state.get("status") != "ok":
		return {
			"status": "stop",
			"route": final_state.get("route", "rag"),
			"message": str(final_state.get("message", "I do not have enough information.")),
		}

	_set_previous_session_route(
		session_id=session_id,
		route=str(final_state.get("route", "rag")),
	)

	if final_state.get("route") == "code_reviewer":
		return {
			"status": "ok",
			"route": "code_reviewer",
			"answer": str(final_state.get("answer", "")).strip(),
		}

	return {
		"status": "ok",
		"route": "rag",
		"chat_llm": final_state.get("chat_llm"),
		"memory_summary": str(final_state.get("memory_summary", "")),
		"context": str(final_state.get("context", "")),
		"conversation_memory": final_state.get("conversation_memory"),
	}


async def stream_orchestrated_response(
	question: str,
	session_id: str = "default",
	persist_dir: Path | None = None,
	llm_model: str = LLM_MODEL,
	reranker_model: str = RERANKER_MODEL,
) -> AsyncIterator[str]:
	"""Stream one unified orchestrated response over either route."""
	compiled = compile_prompt_with_orchestrator(
		question=question,
		session_id=session_id,
		persist_dir=persist_dir,
		llm_model=llm_model,
		reranker_model=reranker_model,
	)
	if compiled.get("status") != "ok":
		yield str(compiled.get("message", "I do not have enough information."))
		return

	if compiled.get("route") == "code_reviewer":
		yield str(compiled.get("answer", ""))
		return

	chat_llm = compiled.get("chat_llm")
	if chat_llm is None:
		yield "I do not have enough information to answer right now."
		return

	generation_chain = build_generation_chain(chat_llm)
	chain_input = {
		"memory_summary": compiled["memory_summary"],
		"question": question,
		"context": compiled["context"],
	}

	full_answer = ""
	async for chunk in generation_chain.astream(chain_input):
		token = str(chunk)
		full_answer += token
		yield token

	conversation_memory = compiled.get("conversation_memory")
	if conversation_memory is not None:
		save_memory_turn(
			conversation_memory=conversation_memory,
			question=question,
			answer=full_answer,
		)
