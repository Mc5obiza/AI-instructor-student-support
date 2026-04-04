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
from typing import Any

from langchain_core.tools import tool
from langchain_ollama import ChatOllama


LLM_MODEL = "llama3.1"
REVIEW_TIMEOUT_SECONDS = 5
PYLINT_TIMEOUT_SECONDS = 20
COMPLEXITY_THRESHOLD = 10
MAX_LINT_MESSAGES = 25

try:
	import resource
except ImportError:
	resource = None


def extract_markdown_code(text: str) -> str | None:
	"""Extract first fenced code block, preferring python fences."""
	pattern = r"```(?:python)?\s*(.*?)```"
	match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
	if not match:
		return None

	code = match.group(1).strip()
	return code or None


def _earliest_python_keyword_start(text: str) -> int:
	"""Find earliest index where Python-looking code begins."""
	python_keywords = ["def ", "class ", "import ", "for ", "while ", "if ", "print("]
	lowered = text.lower()

	positions = [lowered.find(keyword) for keyword in python_keywords if lowered.find(keyword) != -1]
	return min(positions) if positions else -1


def looks_like_python(text: str) -> bool:
	"""Heuristic check for Python-like content."""
	python_keywords = ["def ", "class ", "import ", "for ", "while ", "if ", "print("]
	lowered = text.lower()
	return any(keyword in lowered for keyword in python_keywords)


def extract_task_and_code(query: str) -> tuple[str, str]:
	"""Extract user task and code from markdown or plain text prompts."""
	markdown_code = extract_markdown_code(query)
	if markdown_code:
		task = re.sub(r"```(?:python)?\s*.*?```", "", query, flags=re.DOTALL | re.IGNORECASE).strip()
		return (task or "Review and fix this as Python code.", markdown_code)

	start_idx = _earliest_python_keyword_start(query)
	if start_idx != -1:
		code_candidate = query[start_idx:].strip()
		if looks_like_python(code_candidate):
			task = query[:start_idx].strip()
			return (task or "Review and fix this as Python code.", code_candidate)

	return (query.strip() or "Review and fix this as Python code.", "")


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
) -> str:
	"""Build plain-text code-review prompt that always treats input as Python."""
	return f"""
You are a Python code reviewer.

Mandatory rules:
1. Always treat USER CODE as Python code, even if it looks like machine code, pseudocode, or another language.
2. Always provide an appropriate Python fix.
3. Respond with plain text only.
4. Do not return JSON.

Your response must include:
- A short diagnosis of what is wrong.
- A corrected Python version of the code.
- A short explanation of why the fix works.

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


@tool("code_reviewer_tool")
def code_reviewer_tool(user_task: str, user_code: str, llm_model: str = LLM_MODEL) -> dict[str, Any]:
	"""Run static analysis, sandbox execution, and plain-text LLM code review."""
	if not user_code.strip():
		return {
			"status": "stop",
			"message": "Please provide code to review.",
		}

	static_report = run_static_analysis(user_code)
	execution_report = execute_code_sandbox(user_code)
	review_prompt = build_code_review_prompt(user_task, user_code, static_report, execution_report)

	chat_llm = ChatOllama(model=llm_model, temperature=0.0, verbose=False)
	llm_response = chat_llm.invoke(review_prompt)
	review_response = parse_code_review_response(str(llm_response.content))

	return {
		"status": "ok",
		"user_task": user_task,
		"static_analysis": static_report,
		"execution": execution_report,
		"review_response": review_response,
	}


def resolve_code_review_input(
	query: str = "",
	user_task: str = "",
	user_code: str = "",
) -> tuple[str, str]:
	"""Resolve task/code pair using extraction plus safe fallback behavior."""
	existing_task = user_task.strip()
	existing_code = user_code.strip()

	extracted_task, extracted_code = extract_task_and_code(query)
	resolved_task = existing_task or extracted_task or "Review and fix this as Python code."
	resolved_code = existing_code or extracted_code

	if not resolved_code:
		fallback_code = query.strip()
		if fallback_code:
			resolved_code = fallback_code

	return resolved_task, resolved_code


def run_code_reviewer_tool_pipeline(
	user_task: str,
	user_code: str,
	llm_model: str = LLM_MODEL,
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
		"static_analysis": tool_output.get("static_analysis", {}),
		"execution": tool_output.get("execution", {}),
	}


def run_code_reviewer_tool_from_query(query: str, llm_model: str = LLM_MODEL) -> dict[str, Any]:
	"""Run the code-review tool directly from a raw query."""
	user_task, user_code = resolve_code_review_input(query=query)
	return run_code_reviewer_tool_pipeline(user_task=user_task, user_code=user_code, llm_model=llm_model)
