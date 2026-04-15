from __future__ import annotations

import importlib
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Literal, cast

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyCookie, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr, Field

try:
	from langchain_ollama import ChatOllama
except Exception:
	ChatOllama = None

try:
	from backend.rag.generator import generate_stream
except ImportError:
	from rag.generator import generate_stream


MANAGING_SESSIONS_PATH = Path(__file__).resolve().parent / "managing_sessions"
if str(MANAGING_SESSIONS_PATH) not in sys.path:
	sys.path.insert(0, str(MANAGING_SESSIONS_PATH))

UserInterface = importlib.import_module("userInterface").UserInterface
SessionInterface = importlib.import_module("sessionInterface").SessionInterface
MessageInterface = importlib.import_module("messageInterface").MessageInterface

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-this-secret-before-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
ACCESS_COOKIE_NAME = "access_token"
CHAT_SESSION_COOKIE_NAME = "chat_session_id"
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").strip().lower() == "true"
_raw_samesite = os.getenv("COOKIE_SAMESITE", "lax").strip().lower()
if _raw_samesite not in {"lax", "strict", "none"}:
	_raw_samesite = "lax"
COOKIE_SAMESITE = cast(Literal["lax", "strict", "none"], _raw_samesite)
SESSION_TITLE_MODEL = os.getenv("SESSION_TITLE_MODEL", "llama3.1")
SESSION_TITLE_MAX_CHARS = int(os.getenv("SESSION_TITLE_MAX_CHARS", "80"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
_default_frontend_origins = [
	"http://127.0.0.1:5173",
	"http://localhost:5173",
	"http://127.0.0.1:3000",
	"http://localhost:3000",
	"http://127.0.0.1:8501",
	"http://localhost:8501",
]
_raw_frontend_origins = os.getenv("FRONTEND_ORIGINS", "")
if _raw_frontend_origins.strip():
	FRONTEND_ORIGINS = [origin.strip() for origin in _raw_frontend_origins.split(",") if origin.strip()]
else:
	FRONTEND_ORIGINS = _default_frontend_origins

cookie_scheme = APIKeyCookie(name=ACCESS_COOKIE_NAME, auto_error=False)


class PromptRequest(BaseModel):
	"""Request model for frontend prompt streaming."""
	prompt: str = Field(..., min_length=1)
	session_id: str | None = Field(default=None, min_length=1)
	user_id: str | None = Field(default=None, min_length=1)


class RegisterRequest(BaseModel):
	username: str = Field(..., min_length=2)
	email: EmailStr
	password: str = Field(..., min_length=8)


class CurrentUser(BaseModel):
	user_id: str
	email: str


def _auth_error(detail: str = "Could not validate credentials") -> HTTPException:
	return HTTPException(
		status_code=status.HTTP_401_UNAUTHORIZED,
		detail=detail,
		headers={"WWW-Authenticate": "Bearer"},
	)


def _create_access_token(user_id: str, email: str) -> str:
	expires_at = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
	payload = {"sub": email, "uid": user_id, "exp": expires_at}
	return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def _set_auth_cookie(response: Response, token: str) -> None:
	response.set_cookie(
		key=ACCESS_COOKIE_NAME,
		value=token,
		httponly=True,
		secure=COOKIE_SECURE,
		samesite=COOKIE_SAMESITE,
		max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
		expires=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
		path="/",
	)


def _clear_auth_cookie(response: Response) -> None:
	response.delete_cookie(key=ACCESS_COOKIE_NAME, path="/")


def _set_chat_session_cookie(response: Response, session_id: str) -> None:
	response.set_cookie(
		key=CHAT_SESSION_COOKIE_NAME,
		value=session_id,
		httponly=True,
		secure=COOKIE_SECURE,
		samesite=COOKIE_SAMESITE,
		max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
		expires=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
		path="/",
	)


def _clear_chat_session_cookie(response: Response) -> None:
	response.delete_cookie(key=CHAT_SESSION_COOKIE_NAME, path="/")


def _clear_legacy_session_cookie(response: Response) -> None:
	response.delete_cookie(key="session_id", path="/")


def _clear_logout_cookies(response: Response) -> None:
	_clear_auth_cookie(response)
	_clear_chat_session_cookie(response)
	_clear_legacy_session_cookie(response)


def _fallback_session_title(question: str) -> str:
	clean_question = " ".join(question.split()).strip()
	if not clean_question:
		return "New Chat"
	fallback = " ".join(clean_question.split(" ")[:8]).strip()
	if len(fallback) > SESSION_TITLE_MAX_CHARS:
		fallback = fallback[:SESSION_TITLE_MAX_CHARS].rstrip(" ,.;:-")
	return fallback or "New Chat"


def _sanitize_session_title(raw_title: str, fallback_title: str) -> str:
	cleaned = str(raw_title or "").strip().strip('"').strip("'")
	cleaned = re.sub(r"^title\s*:\s*", "", cleaned, flags=re.IGNORECASE)
	cleaned = " ".join(cleaned.split())
	if not cleaned:
		cleaned = fallback_title
	if len(cleaned) > SESSION_TITLE_MAX_CHARS:
		cleaned = cleaned[:SESSION_TITLE_MAX_CHARS].rstrip(" ,.;:-")
	return cleaned or fallback_title


def _suggest_session_title(question: str, answer: str) -> str:
	fallback_title = _fallback_session_title(question)
	if ChatOllama is None:
		return fallback_title

	prompt = f"""
Please set a title for this question response.
Return one short title only.
Do not include quotes.
Do not include markdown.
Keep it under 8 words.

Question: {question}
Response: {answer}
""".strip()

	try:
		title_llm = ChatOllama(
			model=SESSION_TITLE_MODEL,
			temperature=0.0,
			verbose=False,
			base_url=OLLAMA_BASE_URL,
		)
		response = title_llm.invoke(prompt)
		response_text = str(getattr(response, "content", response)).strip()
		return _sanitize_session_title(response_text, fallback_title)
	except Exception:
		return fallback_title


def _get_session_message_count(session_id: str) -> int | None:
	try:
		result = json.loads(SessionInterface().get_messages(session_id=session_id))
	except Exception:
		return None

	if str(result.get("status", "")) != "200":
		return None

	messages = result.get("messages", [])
	if not isinstance(messages, list):
		return None
	return len(messages)


def _is_placeholder_title(title: str | None) -> bool:
	normalized = str(title or "").strip().lower()
	return normalized in {"", "new chat", "chat session"}


def _get_session_title_for_user(user_id: str, session_id: str) -> str | None:
	try:
		sessions = _get_sessions_or_raise(user_id=user_id)
	except Exception:
		return None

	for item in sessions:
		if str(item.get("session_id", "")).strip() == session_id:
			return str(item.get("title", "")).strip()
	return None


def _set_session_title_after_response(
	session_id: str,
	user_id: str,
	question: str,
	answer: str,
	message_count_before_user_message: int | None,
) -> str | None:
	current_title = _get_session_title_for_user(user_id=user_id, session_id=session_id)
	should_update_first_turn = message_count_before_user_message == 0
	should_update_placeholder = _is_placeholder_title(current_title)
	if not (should_update_first_turn or should_update_placeholder):
		return None

	title = _suggest_session_title(question=question, answer=answer)
	if not title or _is_placeholder_title(title):
		return None

	try:
		result = json.loads(SessionInterface().set_title(session_id=session_id, title=title))
		if str(result.get("status", "")) == "200":
			return title
	except Exception:
		return None

	return None


def _get_user_id_from_interface(email: str) -> str:
	result = json.loads(UserInterface().get_user_id(email=email))
	if str(result.get("status", "")) != "200":
		raise _auth_error()
	user_id = str(result.get("user_id", "")).strip()
	if not user_id:
		raise _auth_error()
	return user_id


def _resolve_or_create_session_id(
	prompt: str,
	request_session_id: str | None,
	cookie_session_id: str | None,
	user_id: str,
) -> str:
	provided_session_id = (request_session_id or "").strip()
	if provided_session_id:
		return provided_session_id

	cached_session_id = (cookie_session_id or "").strip()
	if cached_session_id:
		return cached_session_id

	create_result = json.loads(
		SessionInterface().create_session(
			user_id=user_id,
			title="New Chat",
		)
	)
	if str(create_result.get("status", "")) != "202":
		raise HTTPException(
			status_code=500,
			detail=str(create_result.get("message", "Failed to create session")),
		)

	session_id = str(create_result.get("session_id", "")).strip()
	if not session_id:
		raise HTTPException(status_code=500, detail="Session created without a session_id")
	return session_id


def _create_session_or_raise(user_id: str, title: str = "New Chat") -> str:
	create_result = json.loads(
		SessionInterface().create_session(
			user_id=user_id,
			title=title,
		)
	)
	if str(create_result.get("status", "")) != "202":
		raise HTTPException(
			status_code=500,
			detail=str(create_result.get("message", "Failed to create session")),
		)

	session_id = str(create_result.get("session_id", "")).strip()
	if not session_id:
		raise HTTPException(status_code=500, detail="Session created without a session_id")
	return session_id


def _get_sessions_or_raise(user_id: str) -> list[dict[str, Any]]:
	result_json, sessions = SessionInterface().get_session(user_id=user_id)
	result = json.loads(result_json)
	status_code = str(result.get("status", ""))

	if status_code == "404":
		return []
	if status_code != "200":
		raise HTTPException(
			status_code=500,
			detail=str(result.get("message", "Failed to fetch sessions")),
		)

	resolved_sessions = sessions if isinstance(sessions, list) else result.get("sessions", [])
	if not isinstance(resolved_sessions, list):
		return []

	clean_sessions: list[dict[str, Any]] = []
	for item in resolved_sessions:
		if not isinstance(item, dict):
			continue
		session_id = str(item.get("session_id", "")).strip()
		if not session_id:
			continue
		clean_sessions.append(
			{
				"session_id": session_id,
				"title": str(item.get("title", "Chat session")),
				"updated_at": item.get("updated_at"),
			}
		)
	return clean_sessions


def _get_messages_or_raise(session_id: str) -> list[dict[str, Any]]:
	result = json.loads(SessionInterface().get_messages(session_id=session_id))
	if str(result.get("status", "")) != "200":
		raise HTTPException(
			status_code=500,
			detail=str(result.get("message", "Failed to fetch messages")),
		)

	messages = result.get("messages", [])
	if not isinstance(messages, list):
		return []
	return [msg for msg in messages if isinstance(msg, dict)]


def _store_message_or_raise(session_id: str, content: str, role: str) -> None:
	result = json.loads(
		MessageInterface().send_message(session_id=session_id, content=content, role=role)
	)
	if str(result.get("status", "")) != "200":
		raise HTTPException(
			status_code=500,
			detail=str(result.get("error", result.get("message", "Failed to store message"))),
		)


def _register_user_or_raise(username: str, email: str, password: str) -> str:
	register_result = json.loads(
		UserInterface().add_user(username=username, email=email, password=password)
	)
	status_code = str(register_result.get("status", ""))
	if status_code == "400":
		raise HTTPException(status_code=409, detail="User already exists")
	if status_code != "202":
		raise HTTPException(
			status_code=500,
			detail=str(register_result.get("message", "Could not create account")),
		)

	id_result = json.loads(UserInterface().get_user_id(email=email))
	if str(id_result.get("status", "")) != "200":
		raise HTTPException(status_code=500, detail="Account created but user id was not found")

	user_id = str(id_result.get("user_id", "")).strip()
	if not user_id:
		raise HTTPException(status_code=500, detail="Account created but user id is missing")
	return user_id


def _verify_login_or_raise(email: str, password: str) -> str:
	verify_result = json.loads(
		UserInterface().verify_user(email=email, password=password)
	)
	verify_status = str(verify_result.get("status", ""))
	if verify_status in {"400", "404"}:
		raise _auth_error("Incorrect email or password")
	if verify_status != "200":
		raise HTTPException(
			status_code=500,
			detail=str(verify_result.get("message", "Could not verify credentials")),
		)

	return _get_user_id_from_interface(email)


def get_current_user(token: str | None = Depends(cookie_scheme)) -> CurrentUser:
	if not token:
		raise _auth_error("Not authenticated")

	try:
		payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
	except JWTError as exc:
		raise _auth_error("Invalid or expired token") from exc

	email = str(payload.get("sub", "")).strip().lower()
	if not email:
		raise _auth_error()
	user_id = _get_user_id_from_interface(email)
	return CurrentUser(user_id=user_id, email=email)


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
				async for token in generate_stream(question=prompt, session_id=session_id):
					assistant_response += token
					event = {"type": "token", "token": token}
					yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
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


