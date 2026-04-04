from __future__ import annotations

from pathlib import Path
from typing import AsyncIterator

try:
	from backend.rag.orchestrator import stream_orchestrated_response
except ImportError:
	from rag.orchestrator import stream_orchestrated_response


LLM_MODEL = "llama3.1"
RERANKER_MODEL = "BAAI/bge-reranker-base"


async def generate_stream(
	question: str,
	session_id: str = "default",
	persist_dir: Path | None = None,
	llm_model: str = LLM_MODEL,
	reranker_model: str = RERANKER_MODEL,
) -> AsyncIterator[str]:
	"""Stream generated answer tokens through one orchestrator router."""
	async for token in stream_orchestrated_response(
		question=question,
		session_id=session_id,
		persist_dir=persist_dir,
		llm_model=llm_model,
		reranker_model=reranker_model,
	):
		yield token
