from __future__ import annotations

from typing import Any

import pytest
from langchain_classic.memory import ConversationSummaryBufferMemory
from langchain_core.messages import HumanMessage

from backend import llm_clients

pytestmark = pytest.mark.unit


def test_create_chat_client_uses_openrouter_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
	class CapturingChatModel:
		def __init__(self, **kwargs: Any) -> None:
			self.kwargs = kwargs

	monkeypatch.delenv("LLM_PROVIDER", raising=False)
	monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
	monkeypatch.setattr(llm_clients, "_OpenRouterChatOpenAI", CapturingChatModel)

	client = llm_clients.create_chat_client()

	assert isinstance(client, CapturingChatModel)
	assert client.kwargs["model"] == llm_clients.OPENROUTER_MODEL
	assert client.kwargs["base_url"] == llm_clients.OPENROUTER_BASE_URL


def test_create_chat_client_uses_openrouter_sdk_configuration(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	class OpenAI:
		def __init__(self, **kwargs: Any) -> None:
			self.kwargs = kwargs

	monkeypatch.setenv("LLM_PROVIDER", "openrouter")
	monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
	monkeypatch.setattr(llm_clients, "_OpenRouterChatOpenAI", OpenAI)

	client = llm_clients.create_chat_client()

	assert isinstance(client, OpenAI)
	assert client.kwargs["model"] == llm_clients.OPENROUTER_MODEL
	assert client.kwargs["base_url"] == "https://openrouter.ai/api/v1"
	assert client.kwargs["api_key"].get_secret_value() == "test-key"


def test_openrouter_free_model_uses_fallback_for_message_token_count(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	monkeypatch.setenv("LLM_PROVIDER", "openrouter")
	monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

	client = llm_clients.create_chat_client(model="openrouter/free")

	assert client.get_num_tokens_from_messages(
		[HumanMessage(content="How does gradient descent work?")]
	) > 0
	memory = ConversationSummaryBufferMemory(llm=client, max_token_limit=100)
	memory.save_context({"input": "hello"}, {"output": "hi"})
	assert len(memory.chat_memory.messages) == 2


def test_openrouter_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setenv("LLM_PROVIDER", "openrouter")
	monkeypatch.delenv("OPENROUTERAPIKEY", raising=False)
	monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

	with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
		llm_clients.create_chat_client()


def test_create_chat_client_uses_ollama_when_explicitly_configured(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	class Ollama:
		def __init__(self, **kwargs: Any) -> None:
			self.kwargs = kwargs

	monkeypatch.setenv("LLM_PROVIDER", "ollama")
	monkeypatch.setattr(llm_clients, "ChatOllama", Ollama)

	client = llm_clients.create_chat_client(model="llama3.1-test")

	assert isinstance(client, Ollama)
	assert client.kwargs["model"] == "llama3.1-test"
	assert client.kwargs["base_url"] == llm_clients.OLLAMA_BASE_URL


def test_create_chat_client_rejects_unknown_provider(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	monkeypatch.setenv("LLM_PROVIDER", "unsupported")

	with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
		llm_clients.create_chat_client()
