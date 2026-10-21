from __future__ import annotations

from typing import Any

import pytest

from backend import llm_clients

pytestmark = pytest.mark.unit


def test_create_chat_client_uses_ollama_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
	class Ollama:
		def __init__(self, **kwargs: Any) -> None:
			self.kwargs = kwargs

	monkeypatch.delenv("LLM_PROVIDER", raising=False)
	monkeypatch.setattr(llm_clients, "ChatOllama", Ollama)

	client = llm_clients.create_chat_client(model="llama3.1-test")

	assert isinstance(client, Ollama)
	assert client.kwargs["model"] == "llama3.1-test"
	assert client.kwargs["base_url"] == llm_clients.OLLAMA_BASE_URL


def test_create_chat_client_uses_openrouter_sdk_configuration(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	class OpenAI:
		def __init__(self, **kwargs: Any) -> None:
			self.kwargs = kwargs

	monkeypatch.setenv("LLM_PROVIDER", "openrouter")
	monkeypatch.setenv("OPENROUTERAPIKEY", "test-key")
	monkeypatch.setattr(llm_clients, "ChatOpenAI", OpenAI)

	client = llm_clients.create_chat_client()

	assert isinstance(client, OpenAI)
	assert client.kwargs["model"] == llm_clients.OPENROUTER_MODEL
	assert client.kwargs["base_url"] == "https://openrouter.ai/api/v1"
	assert client.kwargs["api_key"].get_secret_value() == "test-key"


def test_openrouter_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setenv("LLM_PROVIDER", "openrouter")
	monkeypatch.delenv("OPENROUTERAPIKEY", raising=False)
	monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

	with pytest.raises(RuntimeError, match="OPENROUTERAPIKEY"):
		llm_clients.create_chat_client()


def test_create_chat_client_rejects_unknown_provider(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	monkeypatch.setenv("LLM_PROVIDER", "unsupported")

	with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
		llm_clients.create_chat_client()
