from __future__ import annotations

import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


load_dotenv(Path(__file__).resolve().parent.parent / ".env")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_BASE_URL = os.getenv(
	"OLLAMA_BASE_URL",
	os.getenv("OLLAMA_HOST", "http://localhost:11434"),
).rstrip("/")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
_DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter").strip().lower()
DEFAULT_LLM_MODEL = OPENROUTER_MODEL if _DEFAULT_PROVIDER == "openrouter" else OLLAMA_MODEL


def _approximate_token_ids(text: str) -> list[int]:
	"""Return stable word-and-punctuation tokens for models without tokenizer support."""
	pieces = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
	return list(range(len(pieces)))


class _OpenRouterChatOpenAI(ChatOpenAI):
	def get_num_tokens_from_messages(
		self,
		messages: Sequence[BaseMessage],
		tools: Sequence[Any] | None = None,
		*,
		allow_fetching_images: bool = True,
	) -> int:
		"""Fall back to approximate token counting for OpenRouter model IDs."""
		try:
			return super().get_num_tokens_from_messages(
				messages,
				tools=tools,
				allow_fetching_images=allow_fetching_images,
			)
		except NotImplementedError:
			return BaseChatModel.get_num_tokens_from_messages(
				self,
				list(messages),
				tools=tools,
			)


def create_chat_client(
	model: str | None = None,
	temperature: float = 0.0,
	verbose: bool = False,
	custom_get_token_ids: Callable[[str], list[int]] | None = None,
) -> BaseChatModel:
	"""Create the configured OpenRouter or Ollama chat model."""
	provider = os.getenv("LLM_PROVIDER", "openrouter").strip().lower()

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
		api_key = os.getenv("OPENROUTER_API_KEY")
		if not api_key:
			raise RuntimeError(
				"OPENROUTER_API_KEY is required when LLM_PROVIDER is set to 'openrouter'."
			)
		return _OpenRouterChatOpenAI(
			model=model or OPENROUTER_MODEL,
			temperature=temperature,
			verbose=verbose,
			base_url=OPENROUTER_BASE_URL,
			api_key=SecretStr(api_key),
			custom_get_token_ids=custom_get_token_ids or _approximate_token_ids,
		)

	raise ValueError(
		f"Unsupported LLM_PROVIDER {provider!r}; choose 'ollama' or 'openrouter'."
	)
