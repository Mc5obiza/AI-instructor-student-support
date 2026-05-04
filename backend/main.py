from __future__ import annotations

import json
from typing import Any, AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field

try:
	from backend.helper import (
		CHAT_SESSION_COOKIE_NAME,
		FRONTEND_ORIGINS,
		CurrentUser,
		_clear_logout_cookies,
		_create_access_token,
		_create_session_or_raise,
		_get_messages_or_raise,
		_get_session_message_count,
		_get_sessions_or_raise,
		_register_user_or_raise,
		_resolve_or_create_session_id,
		_set_auth_cookie,
		_set_chat_session_cookie,
		_set_session_title_after_response,
		_store_message_or_raise,
		_verify_login_or_raise,
		get_current_user,
	)
except ImportError:
	from helper import (
		CHAT_SESSION_COOKIE_NAME,
		FRONTEND_ORIGINS,
		CurrentUser,
		_clear_logout_cookies,
		_create_access_token,
		_create_session_or_raise,
		_get_messages_or_raise,
		_get_session_message_count,
		_get_sessions_or_raise,
		_register_user_or_raise,
		_resolve_or_create_session_id,
		_set_auth_cookie,
		_set_chat_session_cookie,
		_set_session_title_after_response,
		_store_message_or_raise,
		_verify_login_or_raise,
		get_current_user,
	)

try:
	from backend.rag.memory import save_memory_turn
	from backend.rag.orchestrator import compile_prompt_with_orchestrator
	from backend.rag.rag_agent import build_generation_chain
except ImportError:
	from rag.memory import save_memory_turn
	from rag.orchestrator import compile_prompt_with_orchestrator
	from rag.rag_agent import build_generation_chain


class PromptRequest(BaseModel):
	"""Request model for frontend prompt streaming."""
	prompt: str = Field(..., min_length=1)
	session_id: str | None = Field(default=None, min_length=1)
	user_id: str | None = Field(default=None, min_length=1)


class RegisterRequest(BaseModel):
	username: str = Field(..., min_length=2)
	email: EmailStr
	password: str = Field(..., min_length=8)


def create_app() -> FastAPI:
	"""Create and configure FastAPI application."""
	app = FastAPI(title="AI Learning Assistant API", version="1.1.0")

	app.add_middleware(
		CORSMiddleware,
		allow_origins=FRONTEND_ORIGINS,
		allow_credentials=True,
		allow_methods=["*"],
		allow_headers=["*"],
	)

	@app.get("/health")
	async def health() -> dict[str, str]:
		return {"status": "ok"}

	@app.post("/auth/register", status_code=status.HTTP_201_CREATED)
	async def register(request: RegisterRequest, response: Response) -> dict[str, str]:
		username = request.username.strip()
		email = str(request.email).strip().lower()
		if not username:
			raise HTTPException(status_code=400, detail="username is required")

		user_id = _register_user_or_raise(username=username, email=email, password=request.password)
		token = _create_access_token(user_id=user_id, email=email)
		_set_auth_cookie(response, token)
		return {"message": "Account created", "user_id": user_id}

	@app.post("/auth/login")
	async def login(
		response: Response,
		form_data: OAuth2PasswordRequestForm = Depends(),
	) -> dict[str, str]:
		email = form_data.username.strip().lower()
		password = form_data.password
		if not email or not password:
			raise HTTPException(status_code=400, detail="email and password are required")

		user_id = _verify_login_or_raise(email=email, password=password)
		token = _create_access_token(user_id=user_id, email=email)
		_set_auth_cookie(response, token)
		return {"message": "Login successful"}

	@app.post("/auth/logout")
	async def logout(response: Response) -> dict[str, str]:
		_clear_logout_cookies(response)
		return {"message": "Successfully logged out"}

	@app.get("/logout")
	async def logout_simple(response: Response) -> dict[str, str]:
		_clear_logout_cookies(response)
		return {"message": "Successfully logged out"}

	@app.post("/chat/session/new")
	async def create_new_chat_session(
		response: Response,
		current_user: CurrentUser = Depends(get_current_user),
	) -> dict[str, str]:
		session_id = _create_session_or_raise(user_id=current_user.user_id)
		_set_chat_session_cookie(response=response, session_id=session_id)
		return {"message": "New session created", "session_id": session_id}

	@app.get("/chat/sessions")
	async def list_chat_sessions(
		current_user: CurrentUser = Depends(get_current_user),
	) -> dict[str, Any]:
		sessions = _get_sessions_or_raise(user_id=current_user.user_id)
		return {"sessions": sessions}

	@app.get("/chat/session/{session_id}/messages")
	async def get_chat_session_messages(
		session_id: str,
		current_user: CurrentUser = Depends(get_current_user),
	) -> dict[str, Any]:
		sessions = _get_sessions_or_raise(user_id=current_user.user_id)
		allowed_session_ids = {str(item.get("session_id", "")).strip() for item in sessions}
		if session_id not in allowed_session_ids:
			raise HTTPException(status_code=404, detail="Session not found")

		messages = _get_messages_or_raise(session_id=session_id)
		return {"session_id": session_id, "messages": messages}

	@app.post("/ask/stream")
	async def ask_prompt_stream(
		request: PromptRequest,
		http_request: Request,
		current_user: CurrentUser = Depends(get_current_user),
	) -> StreamingResponse:
		"""Stream SSE events with token chunks and final status."""
		prompt = request.prompt.strip()
		if not prompt:
			raise HTTPException(status_code=400, detail="prompt is required")

		session_id = _resolve_or_create_session_id(
			prompt=prompt,
			request_session_id=request.session_id,
			cookie_session_id=http_request.cookies.get(CHAT_SESSION_COOKIE_NAME),
			user_id=current_user.user_id,
		)
		message_count_before_user_message = _get_session_message_count(session_id=session_id)
		_store_message_or_raise(session_id=session_id, content=prompt, role="user")

		async def event_generator() -> AsyncIterator[str]:
			assistant_response = ""
			final_title: str | None = None
			try:
				compiled = compile_prompt_with_orchestrator(
					question=prompt,
					session_id=session_id,
				)
				conversation_memory = None

				if compiled.get("status") != "ok":
					token = str(compiled.get("message", "I do not have enough information."))
					assistant_response = token
					event = {"type": "token", "token": token}
					yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
				elif compiled.get("route") == "code_reviewer":
					token = str(compiled.get("answer", ""))
					assistant_response = token
					event = {"type": "token", "token": token}
					yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
				else:
					chat_llm = compiled.get("chat_llm")
					if chat_llm is None:
						token = "I do not have enough information to answer right now."
						assistant_response = token
						event = {"type": "token", "token": token}
						yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
					else:
						conversation_memory = compiled.get("conversation_memory")
						generation_chain = build_generation_chain(chat_llm)
						chain_input = {
							"memory_summary": compiled["memory_summary"],
							"question": prompt,
							"context": compiled["context"],
						}
						async for chunk in generation_chain.astream(chain_input):
							token = str(chunk)
							assistant_response += token
							event = {"type": "token", "token": token}
							yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

						if conversation_memory is not None:
							save_memory_turn(
								conversation_memory=conversation_memory,
								question=prompt,
								answer=assistant_response,
								session_id=session_id,
							)
				_store_message_or_raise(
					session_id=session_id,
					content=assistant_response,
					role="assistant",
				)
				final_title = _set_session_title_after_response(
					session_id=session_id,
					user_id=current_user.user_id,
					question=prompt,
					answer=assistant_response,
					message_count_before_user_message=message_count_before_user_message,
				)
				done_event: dict[str, Any] = {"type": "done"}
				if final_title:
					done_event["session_title"] = final_title
				yield f"data: {json.dumps(done_event, ensure_ascii=False)}\n\n"
			except Exception as exc:
				error_event = {"type": "error", "error": f"Failed to process prompt: {exc}"}
				yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"

		response = StreamingResponse(
			event_generator(),
			media_type="text/event-stream",
			headers={
				"Cache-Control": "no-cache",
				"Connection": "keep-alive",
				"X-Accel-Buffering": "no",
			},
		)
		_set_chat_session_cookie(response=response, session_id=session_id)
		return response

	return app


app = create_app()


def main() -> None:
	"""Run the backend with Uvicorn when executed directly."""
	import uvicorn

	uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
	main()


