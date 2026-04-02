from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

try:
	from .rag.retriever import generate_answer_with_sources, generate_answer_with_sources_stream
except ImportError:
	from rag.retriever import generate_answer_with_sources, generate_answer_with_sources_stream


class PromptRequest(BaseModel):
	user_id: str = Field(..., min_length=1)
	prompt: str = Field(..., min_length=1)
	date: str = Field(..., min_length=1)


class SourceItem(BaseModel):
	source: Optional[str] = None
	course: Optional[str] = None
	part: Optional[str] = None
	page_start: Optional[int] = None
	page_end: Optional[int] = None
	score: Optional[float] = None


class PromptResponse(BaseModel):
	user_id: str
	date: str
	prompt: str
	answer: str
	sources: List[SourceItem]
	routing: Dict[str, Any]


def create_app() -> FastAPI:
	app = FastAPI(title="AI_LilMose3da API", version="1.0.0")

	@app.get("/health")
	def health() -> Dict[str, str]:
		return {"status": "ok"}

	@app.post("/ask", response_model=PromptResponse)
	def ask_prompt(request: PromptRequest) -> PromptResponse:
		try:
			result = generate_answer_with_sources(
				question=request.prompt,
				user_id=request.user_id,
				request_date=request.date,
			)
		except Exception as exc:
			raise HTTPException(status_code=500, detail=f"Failed to process prompt: {exc}") from exc

		return PromptResponse(
			user_id=request.user_id,
			date=request.date,
			prompt=request.prompt,
			answer=str(result.get("answer", "")),
			sources=result.get("sources", []),
			routing=result.get("routing", {}),
		)

	@app.post("/ask/stream")
	def ask_prompt_stream(request: PromptRequest) -> StreamingResponse:
		def event_generator():
			try:
				for event in generate_answer_with_sources_stream(
					question=request.prompt,
					user_id=request.user_id,
					request_date=request.date,
				):
					yield f"data: {json.dumps(event, ensure_ascii=False)}\\n\\n"
			except Exception as exc:
				error_event = {
					"type": "error",
					"error": f"Failed to process prompt: {exc}",
				}
				yield f"data: {json.dumps(error_event, ensure_ascii=False)}\\n\\n"

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

