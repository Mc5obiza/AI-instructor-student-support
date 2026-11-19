from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from backend.rag import reranker

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def clear_model_cache() -> Iterator[None]:
	reranker.clear_reranker_cache()
	yield
	reranker.clear_reranker_cache()


def test_auto_device_prefers_cuda_when_available(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	monkeypatch.setattr(reranker.torch.cuda, "is_available", lambda: True)

	assert reranker.resolve_reranker_device("auto") == "cuda"


def test_auto_device_falls_back_to_cpu(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	monkeypatch.setattr(reranker.torch.cuda, "is_available", lambda: False)

	assert reranker.resolve_reranker_device("auto") == "cpu"


def test_explicit_cuda_fails_when_cuda_is_unavailable(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	monkeypatch.setattr(reranker.torch.cuda, "is_available", lambda: False)

	with pytest.raises(RuntimeError, match="CUDA is unavailable"):
		reranker.resolve_reranker_device("cuda")


def test_specific_cuda_device_must_exist(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	monkeypatch.setattr(reranker.torch.cuda, "is_available", lambda: True)
	monkeypatch.setattr(reranker.torch.cuda, "device_count", lambda: 1)

	with pytest.raises(ValueError, match="index 1"):
		reranker.resolve_reranker_device("cuda:1")


def test_load_reranker_passes_selected_device_to_cross_encoder(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	captured: dict[str, Any] = {}

	def cross_encoder(model_name: str, **kwargs: Any) -> object:
		captured["model_name"] = model_name
		captured.update(kwargs)
		return object()

	monkeypatch.setattr(reranker, "resolve_reranker_device", lambda _: "cuda:0")
	monkeypatch.setattr(reranker, "CrossEncoder", cross_encoder)

	model, device = reranker.load_reranker(
		"BAAI/bge-reranker-base",
		local_files_only=True,
	)

	assert model is not None
	assert device == "cuda:0"
	assert captured == {
		"model_name": "BAAI/bge-reranker-base",
		"device": "cuda:0",
		"local_files_only": True,
	}


def test_load_reranker_returns_the_same_model_instance(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	instances: list[object] = []

	def cross_encoder(model_name: str, *, device: str, local_files_only: bool) -> object:
		assert model_name == "model"
		assert device == "cuda"
		assert not local_files_only
		model = object()
		instances.append(model)
		return model

	monkeypatch.setattr(reranker, "resolve_reranker_device", lambda _: "cuda")
	monkeypatch.setattr(reranker, "CrossEncoder", cross_encoder)

	first_model, first_device = reranker.load_reranker("model")
	second_model, second_device = reranker.load_reranker("model")

	assert first_model is second_model
	assert first_device == second_device == "cuda"
	assert len(instances) == 1


def test_device_setting_is_validated(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr(reranker.torch.cuda, "is_available", lambda: False)

	with pytest.raises(ValueError, match="RERANKER_DEVICE"):
		reranker.resolve_reranker_device("gpu")
