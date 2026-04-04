from __future__ import annotations

import json
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

try:
	from backend.rag.generator import generate_stream
except ImportError:
	from rag.generator import generate_stream


class PromptRequest(BaseModel):
	"""Request model for frontend prompt streaming."""
	prompt: str = Field(..., min_length=1)
	session_id: str | None = Field(default=None, min_length=1)
	user_id: str | None = Field(default=None, min_length=1)


def _resolve_prompt_session(request: PromptRequest) -> tuple[str, str]:
	"""Normalize request payload to (prompt, session_id)."""
	prompt = request.prompt.strip()
	session_id = (request.session_id or request.user_id or "default").strip() or "default"

	if not prompt:
		raise HTTPException(status_code=400, detail="prompt is required")

	return prompt, session_id


def create_app() -> FastAPI:
	"""Create and configure FastAPI application."""
	app = FastAPI(title="AI Learning Assistant API", version="1.0.0")

	@app.get("/health")
	async def health() -> dict[str, str]:
		return {"status": "ok"}

	@app.post("/ask/stream")
	async def ask_prompt_stream(request: PromptRequest) -> StreamingResponse:
		"""Stream SSE events with token chunks and final status."""
		prompt, session_id = _resolve_prompt_session(request)

		async def event_generator() -> AsyncIterator[str]:
			try:
				async for token in generate_stream(question=prompt, session_id=session_id):
					event = {"type": "token", "token": token}
					yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
				yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
			except Exception as exc:
				error_event = {"type": "error", "error": f"Failed to process prompt: {exc}"}
				yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"

		return StreamingResponse(
			event_generator(),
			media_type="text/event-stream",
			headers={
				"Cache-Control": "no-cache",
				"Connection": "keep-alive",
				"X-Accel-Buffering": "no",
			},
		)

	return app


app = create_app()


def main() -> None:
	"""Run the backend with Uvicorn when executed directly."""
	import uvicorn

	uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
	main()


