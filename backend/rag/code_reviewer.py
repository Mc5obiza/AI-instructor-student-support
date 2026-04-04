from __future__ import annotations

import ast
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Literal, cast

from langchain_classic.memory import ConversationSummaryBufferMemory
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


LLM_MODEL = "llama3.1"
REVIEW_TIMEOUT_SECONDS = 5
PYLINT_TIMEOUT_SECONDS = 20
COMPLEXITY_THRESHOLD = 10
MAX_LINT_MESSAGES = 25
CODE_REVIEW_MEMORY_MAX_TOKEN_LIMIT = 32000
CODE_REVIEW_MEMORY_SUMMARY_TOKEN_LIMIT = 32000
CODE_REVIEW_HISTORY_TURNS = 5
CODE_REVIEW_HISTORY_CHARS = 8000

_CODE_REVIEW_MEMORIES: dict[str, ConversationSummaryBufferMemory] = {}
_CODE_REVIEW_TURNS: dict[str, list[dict[str, str]]] = {}
REVIEW_ACTION_VALUES = {"correction", "explanation", "generation"}
DEFAULT_REVIEW_ACTIONS = ["correction", "explanation"]


class CodeReviewExtraction(BaseModel):
	"""Structured request extracted by model function-calling."""

	user_task: str = Field(
		...,
		description="Clear user request for this code review step.",
	)
	user_code: str = Field(
		default="",
		description="Code to review. Leave empty only if user is referring to previously reviewed code.",
	)
	use_previous_code: bool = Field(
		default=False,
		description="True when user follow-up should reuse previous code from session history.",
	)
	requested_actions: list[Literal["correction", "explanation", "generation"]] = Field(
		default_factory=lambda: ["correction", "explanation"],
		description=(
			"Requested outputs. correction=fixed code, explanation=why/how, "
			"generation=generate new/improved code variant."
		),
	)

try:
	import resource
except ImportError:
	resource = None


def _to_text(value: str | bytes | None) -> str:
	if value is None:
		return ""
	if isinstance(value, bytes):
		return value.decode("utf-8", errors="ignore")
	if isinstance(value, (bytearray, memoryview)):
		return bytes(value).decode("utf-8", errors="ignore")
	return str(value)


def extract_traceback(stderr: str) -> str:
	"""Extract traceback block from stderr when present."""
	marker = "Traceback (most recent call last):"
	if marker not in stderr:
		return ""
	return stderr[stderr.find(marker) :].strip()


def _sandbox_preexec_fn() -> None:
	"""Apply lightweight process limits on POSIX systems."""
	if resource is None:
		return
	setrlimit = getattr(resource, "setrlimit", None)
	rlimit_as = getattr(resource, "RLIMIT_AS", None)
	rlimit_cpu = getattr(resource, "RLIMIT_CPU", None)
	if not callable(setrlimit) or rlimit_as is None or rlimit_cpu is None:
		return

	memory_limit_bytes = 512 * 1024 * 1024
	cpu_limit_seconds = REVIEW_TIMEOUT_SECONDS
	setrlimit(rlimit_as, (memory_limit_bytes, memory_limit_bytes))
	setrlimit(rlimit_cpu, (cpu_limit_seconds, cpu_limit_seconds))
	if hasattr(signal, "SIGXCPU"):
		signal.signal(getattr(signal, "SIGXCPU"), signal.SIG_DFL)


def run_pylint_report(user_code: str) -> dict[str, Any]:
	"""Run pylint via subprocess and return parsed JSON messages."""
	temp_path: str | None = None
	try:
		with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as temp_file:
			temp_file.write(user_code)
			temp_path = temp_file.name

		process = subprocess.run(
			[
				sys.executable,
				"-m",
				"pylint",
				temp_path,
				"--output-format=json",
				"--score=n",
				"--reports=n",
			],
			capture_output=True,
			text=True,
			timeout=PYLINT_TIMEOUT_SECONDS,
		)

		stdout = (process.stdout or "").strip()
		stderr = (process.stderr or "").strip()

		messages: list[dict[str, Any]] = []
		if stdout:
			try:
				payload = json.loads(stdout)
				if isinstance(payload, list):
					for item in payload[:MAX_LINT_MESSAGES]:
						messages.append(
							{
								"type": item.get("type", ""),
								"symbol": item.get("symbol", ""),
								"line": item.get("line", None),
								"message": item.get("message", ""),
							}
						)
			except json.JSONDecodeError:
				return {
					"messages": [],
					"error": f"Unable to parse pylint JSON output: {stdout[:400]}",
				}

		return {
			"messages": messages,
			"error": stderr[:800] if stderr else "",
		}
	except subprocess.TimeoutExpired:
		return {
			"messages": [],
			"error": f"pylint timed out after {PYLINT_TIMEOUT_SECONDS} seconds.",
		}
	except Exception as exc:
		return {
			"messages": [],
			"error": f"pylint execution failed: {exc}",
		}
	finally:
		if temp_path:
			Path(temp_path).unlink(missing_ok=True)


class CodeStructureAnalyzer(ast.NodeVisitor):
	"""AST-based checks for unreachable code and deep loop nesting."""

	def __init__(self) -> None:
		self.unreachable_code: list[dict[str, Any]] = []
		self.nested_loops: list[dict[str, Any]] = []
		self._loop_depth = 0

	def _scan_block(self, body: list[ast.stmt]) -> None:
		terminated = False
		for stmt in body:
			if terminated:
				self.unreachable_code.append(
					{
						"line": getattr(stmt, "lineno", None),
						"node": type(stmt).__name__,
					}
				)
			self.visit(stmt)
			if isinstance(stmt, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
				terminated = True

	def visit_Module(self, node: ast.Module) -> None:
		self._scan_block(node.body)

	def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
		self._scan_block(node.body)

	def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
		self._scan_block(node.body)

	def visit_For(self, node: ast.For) -> None:
		self._loop_depth += 1
		if self._loop_depth >= 2:
			self.nested_loops.append(
				{
					"line": node.lineno,
					"depth": self._loop_depth,
					"node": "For",
				}
			)
		self.visit(node.target)
		self.visit(node.iter)
		self._scan_block(node.body)
		self._scan_block(node.orelse)
		self._loop_depth -= 1

	def visit_While(self, node: ast.While) -> None:
		self._loop_depth += 1
		if self._loop_depth >= 2:
			self.nested_loops.append(
				{
					"line": node.lineno,
					"depth": self._loop_depth,
					"node": "While",
				}
			)
		self.visit(node.test)
		self._scan_block(node.body)
		self._scan_block(node.orelse)
		self._loop_depth -= 1


def run_static_analysis(user_code: str) -> dict[str, Any]:
	"""Run syntax, AST checks, cyclomatic complexity, and pylint."""
	report: dict[str, Any] = {
		"syntax_error": None,
		"unreachable_code": [],
		"nested_loops": [],
		"complexity": [],
		"pylint": [],
	}

	try:
		syntax_tree = ast.parse(user_code)
	except SyntaxError as exc:
		report["syntax_error"] = {
			"line": exc.lineno,
			"offset": exc.offset,
			"message": exc.msg,
		}
		return report

	analyzer = CodeStructureAnalyzer()
	analyzer.visit(syntax_tree)
	report["unreachable_code"] = analyzer.unreachable_code
	report["nested_loops"] = analyzer.nested_loops

	try:
		from radon.complexity import cc_rank, cc_visit

		complexity_items = cc_visit(user_code)
		report["complexity"] = [
			{
				"name": item.name,
				"line": item.lineno,
				"complexity": item.complexity,
				"rank": cc_rank(item.complexity),
			}
			for item in complexity_items
			if item.complexity >= COMPLEXITY_THRESHOLD
		]
	except Exception as exc:
		report["complexity_error"] = f"radon analysis unavailable: {exc}"

	pylint_report = run_pylint_report(user_code)
	report["pylint"] = pylint_report.get("messages", [])
	if pylint_report.get("error"):
		report["pylint_error"] = pylint_report["error"]

	return report


def execute_code_sandbox(user_code: str, timeout_seconds: int = REVIEW_TIMEOUT_SECONDS) -> dict[str, Any]:
	"""Run code in a temp-file subprocess and capture stdout/stderr/traceback."""
	temp_path: str | None = None
	try:
		with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as temp_file:
			temp_file.write(user_code)
			temp_path = temp_file.name

		run_kwargs: dict[str, Any] = {
			"capture_output": True,
			"text": True,
			"timeout": timeout_seconds,
		}
		if os.name != "nt" and resource is not None:
			run_kwargs["preexec_fn"] = _sandbox_preexec_fn

		completed = subprocess.run([sys.executable, temp_path], **run_kwargs)
		stderr = completed.stderr or ""
		return {
			"stdout": completed.stdout or "",
			"stderr": stderr,
			"traceback": extract_traceback(stderr),
			"returncode": completed.returncode,
			"timed_out": False,
		}
	except subprocess.TimeoutExpired as exc:
		stdout = _to_text(exc.stdout)
		stderr = _to_text(exc.stderr)
		timeout_message = f"Execution timed out after {timeout_seconds} seconds."
		stderr_with_timeout = f"{stderr}\n{timeout_message}".strip()
		return {
			"stdout": stdout,
			"stderr": stderr_with_timeout,
			"traceback": extract_traceback(stderr),
			"returncode": None,
			"timed_out": True,
		}
	except Exception as exc:
		message = f"Sandbox execution failed: {exc}"
		return {
			"stdout": "",
			"stderr": message,
			"traceback": "",
			"returncode": None,
			"timed_out": False,
		}
	finally:
		if temp_path:
			Path(temp_path).unlink(missing_ok=True)


def build_code_review_prompt(
	user_task: str,
	user_code: str,
	static_report: dict[str, Any],
	execution_report: dict[str, Any],
	requested_actions: list[str],
) -> str:
	"""Build action-aware plain-text code-review prompt."""
	requested_actions = _normalize_review_actions(requested_actions)
	explanation_only = requested_actions == ["explanation"]
	action_bullets: list[str] = []
	if "correction" in requested_actions:
		action_bullets.append("- correction: include corrected Python code.")
	if "explanation" in requested_actions:
		action_bullets.append("- explanation: explain key issues and why the fix works.")
	if "generation" in requested_actions:
		action_bullets.append("- generation: generate an improved or alternative Python implementation.")

	action_requirements = "\n".join(action_bullets)
	action_labels = [action.upper() for action in requested_actions]
	section_headers = "\n".join([f"### {label}" for label in action_labels])
	actions_text = ", ".join(requested_actions)
	extra_rule = ""
	if explanation_only:
		extra_rule = (
			"4. Because this request is explanation-only, do not include corrected code, "
			"code snippets, diffs, or replacement implementations."
		)

	return f"""
You are a Python code reviewer.

Mandatory rules:
1. Always treat USER CODE as Python code, even if it looks like machine code, pseudocode, or another language.
2. Respond with plain text only.
3. Do not return JSON.
{extra_rule}

Requested actions for this turn: {actions_text}

Action requirements:
{action_requirements}

Your response must include only requested sections, using these exact headers when present:
{section_headers}

If a section is not requested, omit it entirely.

USER TASK:
{user_task}

USER CODE:
{user_code}

STATIC ANALYSIS:
{json.dumps(static_report, indent=2, ensure_ascii=False)}

EXECUTION OUTPUT (stdout):
{execution_report.get("stdout", "")}

EXECUTION ERRORS (stderr):
{execution_report.get("stderr", "")}

TRACEBACK (if exists):
{execution_report.get("traceback", "")}
""".strip()


def parse_code_review_response(raw_output: str) -> str:
	"""Normalize model output as plain text."""
	cleaned = raw_output.strip()
	cleaned = re.sub(r"^```(?:json|python|text)?\\s*", "", cleaned, flags=re.IGNORECASE)
	cleaned = re.sub(r"\\s*```$", "", cleaned)
	return cleaned.strip()


def _filter_response_to_requested_actions(review_response: str, requested_actions: list[str]) -> str:
	"""Keep only requested sections and enforce explanation-only behavior."""
	normalized_actions = _normalize_review_actions(requested_actions)
	if not review_response.strip():
		return review_response.strip()

	header_pattern = re.compile(r"^###\s+(CORRECTION|EXPLANATION|GENERATION)\s*$", re.MULTILINE)
	matches = list(header_pattern.finditer(review_response))
	if matches:
		kept_sections: list[str] = []
		for idx, match in enumerate(matches):
			header = match.group(1).lower()
			start = match.start()
			end = matches[idx + 1].start() if idx + 1 < len(matches) else len(review_response)
			section_text = review_response[start:end].strip()
			if header in normalized_actions:
				kept_sections.append(section_text)
		review_response = "\n\n".join(kept_sections).strip() if kept_sections else review_response.strip()

	if normalized_actions == ["explanation"]:
		# Ensure explanation-only responses cannot leak corrected/generated code blocks.
		review_response = re.sub(r"```[\s\S]*?```", "", review_response).strip()

	return review_response


def _normalize_review_actions(actions: Any) -> list[str]:
	"""Normalize and validate requested reviewer actions."""
	if not isinstance(actions, list):
		return list(DEFAULT_REVIEW_ACTIONS)

	normalized: list[str] = []
	for action in actions:
		value = str(action).strip().lower()
		if value in REVIEW_ACTION_VALUES and value not in normalized:
			normalized.append(value)

	if not normalized:
		return list(DEFAULT_REVIEW_ACTIONS)
	return normalized


def _looks_like_code_followup_request(query: str) -> bool:
	"""Detect natural-language follow-ups that refer to previously shared code."""
	lowered = query.lower()
	hints = [
		"the code",
		"change the code",
		"modify the code",
		"check if",
		"validate",
		"type",
		"string",
		"int",
		"float",
		"bool",
		"add",
		"improve",
		"how can i change",
	]
	return any(hint in lowered for hint in hints)


def infer_explicit_actions_from_prompt(query: str) -> list[str]:
	"""Infer explicit action requests from natural language for deterministic overrides."""
	lowered = query.lower()
	actions: list[str] = []

	explain_markers = ["explain", "why", "how does", "what does"]
	correction_markers = ["fix", "correct", "debug", "change", "modify", "update", "check"]
	generation_markers = ["generate", "create", "write", "produce", "build"]

	if any(marker in lowered for marker in explain_markers):
		actions.append("explanation")
	if any(marker in lowered for marker in correction_markers):
		actions.append("correction")
	if any(marker in lowered for marker in generation_markers):
		actions.append("generation")

	if "only" in lowered:
		if "explain" in lowered and "explanation" in actions:
			return ["explanation"]
		if any(marker in lowered for marker in ["fix", "correct", "debug"]) and "correction" in actions:
			return ["correction"]
		if any(marker in lowered for marker in ["generate", "create", "write"]) and "generation" in actions:
			return ["generation"]

	return _normalize_review_actions(actions) if actions else []


def extract_code_review_request_with_function_call(
	query: str,
	llm_model: str = LLM_MODEL,
	session_id: str = "default",
) -> dict[str, Any]:
	"""Use function-calling to extract user_task, code payload, and requested actions."""
	session_id = session_id.strip() or "default"
	previous_code = get_last_reviewed_code(session_id=session_id)
	previous_code_excerpt = _compact_text(previous_code, 1200) if previous_code else ""

	planner_llm = ChatOllama(model=llm_model, temperature=0.0, verbose=False)
	bound_planner = planner_llm.bind_tools([CodeReviewExtraction])
	planner_prompt = f"""
You are extracting a structured Python code-review request.

You must call exactly one tool: CodeReviewExtraction.

Rules:
- Put explicit code from the prompt in user_code.
- If no explicit code is present but prompt is a follow-up about previous code, set use_previous_code=true.
- requested_actions can include one or more of: correction, explanation, generation.
- correction: produce fixed code.
- explanation: explain issues/fixes.
- generation: generate an improved/new code variant.

Session context:
- session_id: {session_id}
- previous_code_excerpt:
{previous_code_excerpt or '<none>'}

User prompt:
{query}
""".strip()

	tool_calls: list[dict[str, Any]] = []
	try:
		planner_response = bound_planner.invoke(planner_prompt)
		tool_calls = list(getattr(planner_response, "tool_calls", []) or [])
	except Exception:
		tool_calls = []

	if tool_calls:
		raw_args: Any = tool_calls[0].get("args", {})
		if isinstance(raw_args, str):
			try:
				raw_args = json.loads(raw_args)
			except json.JSONDecodeError:
				raw_args = {}
		if not isinstance(raw_args, dict):
			raw_args = {}

		user_task = str(raw_args.get("user_task", "")).strip() or query.strip() or "Review and fix this as Python code."
		user_code = str(raw_args.get("user_code", "")).strip()
		use_previous_code = bool(raw_args.get("use_previous_code", False))
		actions = _normalize_review_actions(raw_args.get("requested_actions", DEFAULT_REVIEW_ACTIONS))
	else:
		user_task = query.strip() or "Review and fix this as Python code."
		user_code = ""
		use_previous_code = _looks_like_code_followup_request(query)
		actions = list(DEFAULT_REVIEW_ACTIONS)

	if not user_code and (use_previous_code or _looks_like_code_followup_request(query)) and previous_code:
		user_code = previous_code

	if not user_code and ("\n" in query or "def " in query or "class " in query):
		# Last-resort fallback when planner misses embedded code.
		user_code = query.strip()

	return {
		"user_task": user_task,
		"user_code": user_code,
		"requested_actions": actions,
	}


def get_or_create_code_review_memory(
	session_id: str,
	llm: Any,
	max_token_limit: int = CODE_REVIEW_MEMORY_MAX_TOKEN_LIMIT,
) -> ConversationSummaryBufferMemory:
	"""Get or create a code-review session memory."""
	if session_id not in _CODE_REVIEW_MEMORIES:
		memory_cls = cast(Any, ConversationSummaryBufferMemory)
		_CODE_REVIEW_MEMORIES[session_id] = memory_cls(
			llm=llm,
			memory_key="conversation_summary",
			max_token_limit=max_token_limit,
			return_messages=False,
			input_key="question",
			output_key="answer",
		)
	if session_id not in _CODE_REVIEW_TURNS:
		_CODE_REVIEW_TURNS[session_id] = []
	return _CODE_REVIEW_MEMORIES[session_id]


def load_code_review_summary(
	conversation_memory: ConversationSummaryBufferMemory,
	llm: Any,
	summary_token_limit: int = CODE_REVIEW_MEMORY_SUMMARY_TOKEN_LIMIT,
) -> str:
	"""Summarize prior code-review conversation for prompt injection."""
	memory_vars = conversation_memory.load_memory_variables({})
	history_text = str(memory_vars.get("conversation_summary", "")).strip()
	if not history_text:
		return "No prior code review summary."

	summary_prompt = f"""
Summarize this Python code review conversation in at most {summary_token_limit} tokens.
Preserve key bug patterns, fixes, and constraints.
Return summary text only.

History:
{history_text}
""".strip()

	summary_response = llm.invoke(summary_prompt)
	return getattr(summary_response, "content", str(summary_response)).strip()


def _compact_text(text: str, max_chars: int) -> str:
	if len(text) <= max_chars:
		return text
	return text[: max_chars - 3] + "..."


def format_code_review_history(
	session_id: str,
	max_turns: int = CODE_REVIEW_HISTORY_TURNS,
	max_chars: int = CODE_REVIEW_HISTORY_CHARS,
) -> str:
	"""Format recent code input/output memory for the reviewer prompt."""
	turns = _CODE_REVIEW_TURNS.get(session_id, [])
	if not turns:
		return "No prior code input/output history."

	selected = turns[-max_turns:]
	blocks = []
	for idx, turn in enumerate(selected, start=1):
		block = (
			f"[Turn {idx}]\n"
			f"Task:\n{turn.get('task', '')}\n\n"
			f"Code Input:\n{turn.get('code_input', '')}\n\n"
			f"Execution Output:\n{turn.get('execution_output', '')}\n\n"
			f"Review Output:\n{turn.get('review_output', '')}"
		)
		blocks.append(block)

	joined = "\n\n".join(blocks)
	return _compact_text(joined, max_chars)


def save_code_review_turn(
	session_id: str,
	conversation_memory: ConversationSummaryBufferMemory,
	user_task: str,
	user_code: str,
	requested_actions: list[str],
	execution_report: dict[str, Any],
	review_response: str,
) -> None:
	"""Persist one code-review turn in summary memory and structured I/O history."""
	stdout = str(execution_report.get("stdout", "")).strip()
	stderr = str(execution_report.get("stderr", "")).strip()
	traceback_text = str(execution_report.get("traceback", "")).strip()

	execution_output = (
		f"STDOUT:\n{stdout or '<empty>'}\n\n"
		f"STDERR:\n{stderr or '<empty>'}\n\n"
		f"TRACEBACK:\n{traceback_text or '<empty>'}"
	)

	conversation_memory.save_context(
		{
			"question": (
				f"Requested Actions:\n{', '.join(requested_actions)}\n\n"
				f"Task:\n{user_task}\n\n"
				f"Code Input:\n{user_code}\n\n"
				f"Execution Output:\n{execution_output}"
			)
		},
		{"answer": f"Review Output:\n{review_response}"},
	)

	turns = _CODE_REVIEW_TURNS.setdefault(session_id, [])
	turns.append(
		{
			"requested_actions": ", ".join(requested_actions),
			"task": user_task,
			"code_input": user_code,
			"execution_output": execution_output,
			"review_output": review_response,
		}
	)


def get_last_reviewed_code(session_id: str) -> str:
	"""Return last code input stored for this session, if available."""
	turns = _CODE_REVIEW_TURNS.get(session_id, [])
	if not turns:
		return ""
	return str(turns[-1].get("code_input", "")).strip()


def get_code_reviewer_router_context(
	session_id: str,
	llm_model: str = LLM_MODEL,
) -> dict[str, str]:
	"""Return compact session context to help route follow-up code-review prompts."""
	session_id = session_id.strip() or "default"
	turns = _CODE_REVIEW_TURNS.get(session_id, [])
	if not turns:
		return {
			"memory_summary": "No prior code review summary.",
			"last_code_excerpt": "",
		}

	chat_llm = ChatOllama(model=llm_model, temperature=0.0, verbose=False)
	conversation_memory = get_or_create_code_review_memory(session_id=session_id, llm=chat_llm)
	memory_summary = load_code_review_summary(conversation_memory=conversation_memory, llm=chat_llm)
	last_code_excerpt = _compact_text(get_last_reviewed_code(session_id=session_id), 1200)
	return {
		"memory_summary": memory_summary,
		"last_code_excerpt": last_code_excerpt,
	}


@tool("code_reviewer_tool")
def code_reviewer_tool(
	user_task: str,
	user_code: str,
	llm_model: str = LLM_MODEL,
	session_id: str = "default",
	requested_actions: list[str] | None = None,
) -> dict[str, Any]:
	"""Run static analysis, sandbox execution, and plain-text LLM code review."""
	if not user_code.strip():
		return {
			"status": "stop",
			"message": "Please provide code to review.",
		}
	session_id = session_id.strip() or "default"
	selected_actions = _normalize_review_actions(requested_actions or DEFAULT_REVIEW_ACTIONS)

	static_report = run_static_analysis(user_code)
	execution_report = execute_code_sandbox(user_code)

	chat_llm = ChatOllama(model=llm_model, temperature=0.0, verbose=False)
	conversation_memory = get_or_create_code_review_memory(session_id=session_id, llm=chat_llm)
	memory_summary = load_code_review_summary(conversation_memory=conversation_memory, llm=chat_llm)
	code_history = format_code_review_history(session_id=session_id)

	review_prompt = f"""
Code Review Memory Summary:
{memory_summary}

Recent Code Input/Output History:
{code_history}

{build_code_review_prompt(user_task, user_code, static_report, execution_report, selected_actions)}
""".strip()

	llm_response = chat_llm.invoke(review_prompt)
	review_response = parse_code_review_response(str(llm_response.content))
	review_response = _filter_response_to_requested_actions(review_response, selected_actions)
	save_code_review_turn(
		session_id=session_id,
		conversation_memory=conversation_memory,
		user_task=user_task,
		user_code=user_code,
		requested_actions=selected_actions,
		execution_report=execution_report,
		review_response=review_response,
	)

	return {
		"status": "ok",
		"user_task": user_task,
		"requested_actions": selected_actions,
		"static_analysis": static_report,
		"execution": execution_report,
		"memory_summary": memory_summary,
		"code_history": code_history,
		"review_response": review_response,
	}


def run_code_reviewer_tool_pipeline(
	user_task: str,
	user_code: str,
	llm_model: str = LLM_MODEL,
	session_id: str = "default",
	requested_actions: list[str] | None = None,
) -> dict[str, Any]:
	"""Run the code-review tool directly and return normalized payload."""
	if not user_code.strip():
		message = "Please provide code text so I can review and fix it as Python."
		return {
			"status": "stop",
			"route": "code_reviewer",
			"message": message,
			"answer": message,
		}

	tool_output = code_reviewer_tool.invoke(
		{
			"user_task": user_task,
			"user_code": user_code,
			"llm_model": llm_model,
			"session_id": session_id,
			"requested_actions": _normalize_review_actions(requested_actions or DEFAULT_REVIEW_ACTIONS),
		}
	)

	if tool_output.get("status") != "ok":
		message = str(tool_output.get("message", "Code review could not be completed.")).strip()
		return {
			"status": "stop",
			"route": "code_reviewer",
			"message": message,
			"answer": message,
		}

	answer = str(tool_output.get("review_response", "")).strip()
	return {
		"status": "ok",
		"route": "code_reviewer",
		"answer": answer,
		"requested_actions": tool_output.get("requested_actions", []),
		"memory_summary": str(tool_output.get("memory_summary", "")).strip(),
		"code_history": str(tool_output.get("code_history", "")).strip(),
		"static_analysis": tool_output.get("static_analysis", {}),
		"execution": tool_output.get("execution", {}),
	}


def run_code_reviewer_tool_from_query(
	query: str,
	llm_model: str = LLM_MODEL,
	session_id: str = "default",
) -> dict[str, Any]:
	"""Run the code-review tool directly from a raw query."""
	session_id = session_id.strip() or "default"

	request = extract_code_review_request_with_function_call(
		query=query,
		llm_model=llm_model,
		session_id=session_id,
	)
	user_task = str(request.get("user_task", "")).strip() or "Review and fix this as Python code."
	user_code = str(request.get("user_code", "")).strip()
	requested_actions = _normalize_review_actions(request.get("requested_actions", DEFAULT_REVIEW_ACTIONS))
	explicit_actions = infer_explicit_actions_from_prompt(query)
	if explicit_actions:
		requested_actions = explicit_actions

	if not user_code:
		message = "No code found in this request and no previous code is available in this session."
		return {
			"status": "stop",
			"route": "code_reviewer",
			"message": message,
			"answer": message,
		}

	return run_code_reviewer_tool_pipeline(
		user_task=user_task,
		user_code=user_code,
		llm_model=llm_model,
		session_id=session_id,
		requested_actions=requested_actions,
	)
