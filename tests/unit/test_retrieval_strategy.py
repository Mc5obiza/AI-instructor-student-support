from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from backend.rag import retrieval_strategy

pytestmark = pytest.mark.unit


def test_retrieval_strategy_parses_registered_hybrid_reranker_config() -> None:
	strategy = retrieval_strategy.RetrievalStrategy.from_config(
		{
			"retrieval_method": "hybrid_rerank",
			"candidate_k": 20,
			"top_k": 5,
			"reranker_model": "BAAI/bge-reranker-base",
			"source_run_id": "evaluation-run",
		}
	)

	assert strategy == retrieval_strategy.RetrievalStrategy(
		retrieval_method="hybrid_rerank",
		candidate_k=20,
		top_k=5,
		reranker_model="BAAI/bge-reranker-base",
		source_run_id="evaluation-run",
	)


@pytest.mark.parametrize(
	"config",
	[
		{
			"retrieval_method": "hybrid_rerank",
			"candidate_k": 20,
			"top_k": 5,
			"reranker_model": "",
			"source_run_id": "evaluation-run",
		},
		{
			"retrieval_method": "hybrid",
			"candidate_k": 4,
			"top_k": 5,
			"source_run_id": "evaluation-run",
		},
	],
)
def test_retrieval_strategy_rejects_invalid_registry_settings(config: dict[str, Any]) -> None:
	with pytest.raises(ValueError):
		retrieval_strategy.RetrievalStrategy.from_config(config)


def test_load_registered_strategy_uses_config_tag_and_active_alias(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	from mlflow import tracking

	calls: list[tuple[str, str]] = []
	config = {
		"retrieval_method": "hybrid_rerank",
		"candidate_k": 20,
		"top_k": 5,
		"reranker_model": "BAAI/bge-reranker-base",
		"source_run_id": "evaluation-run",
	}

	class Client:
		def __init__(self, tracking_uri: str) -> None:
			assert tracking_uri == "sqlite:///test.db"

		def get_model_version_by_alias(self, model_name: str, alias: str) -> Any:
			calls.append((model_name, alias))
			return SimpleNamespace(
				version="3",
				tags={retrieval_strategy.STRATEGY_CONFIG_TAG: json.dumps(config)},
			)

	monkeypatch.setattr(tracking, "MlflowClient", Client)

	result = retrieval_strategy.load_registered_retrieval_strategy(
		tracking_uri="sqlite:///test.db",
		model_name="test-strategy",
		alias="champion",
	)

	assert result.source_run_id == "evaluation-run"
	assert calls == [("test-strategy", "champion")]
