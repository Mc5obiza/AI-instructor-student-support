from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


load_dotenv(Path(__file__).resolve().parent.parent / ".env")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_BASE_URL = os.getenv(
	"OLLAMA_BASE_URL",
	os.getenv("OLLAMA_HOST", "http://localhost:11434"),
).rstrip("/")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

_DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
DEFAULT_LLM_MODEL = OPENROUTER_MODEL if _DEFAULT_PROVIDER == "openrouter" else OLLAMA_MODEL


def create_chat_client(
	model: str | None = None,
	temperature: float = 0.0,
	verbose: bool = False,
	custom_get_token_ids: Callable[[str], list[int]] | None = None,
) -> BaseChatModel:
	"""Create the configured Ollama or OpenRouter chat model."""
	provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

	if provider == "ollama":
		options: dict[str, Any] = {
			"model": model or OLLAMA_MODEL,
			"temperature": temperature,
			"verbose": verbose,
			"base_url": OLLAMA_BASE_URL,
		}
		if custom_get_token_ids is not None:
			options["custom_get_token_ids"] = custom_get_token_ids
		return ChatOllama(**options)

	if provider == "openrouter":
		api_key = os.getenv("OPENROUTERAPIKEY") or os.getenv("OPENROUTER_API_KEY")
		if not api_key:
			raise RuntimeError(
				"OPENROUTERAPIKEY is required when LLM_PROVIDER is set to 'openrouter'."
			)
		return ChatOpenAI(
			model=model or OPENROUTER_MODEL,
			temperature=temperature,
			verbose=verbose,
			base_url=OPENROUTER_BASE_URL,
			api_key=SecretStr(api_key),
		)

	raise ValueError(
		f"Unsupported LLM_PROVIDER {provider!r}; choose 'ollama' or 'openrouter'."
	)
