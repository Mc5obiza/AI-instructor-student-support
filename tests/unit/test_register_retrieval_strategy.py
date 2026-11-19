from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from scripts.register_retrieval_strategy import build_strategy_config, select_best_run

pytestmark = pytest.mark.unit


def make_run(
	run_id: str,
	method: str,
	hit_rate: float,
	mrr: float,
) -> Any:
	return SimpleNamespace(
		info=SimpleNamespace(run_id=run_id),
		data=SimpleNamespace(
			params={
				"retrieval_method": method,
				"candidate_k": "20",
				"top_k": "5",
				"reranker_model": "BAAI/bge-reranker-base" if method == "hybrid_rerank" else "none",
				"chunk_size": "500",
				"chunk_overlap": "100",
				"index_space": "cosine",
				"ef_search": "50",
			},
			metrics={
				"hit_rate_at_5": hit_rate,
				"mrr_at_5": mrr,
			},
		),
	)


def test_select_best_run_prioritizes_hit_rate_then_mrr() -> None:
	dense = make_run("dense-run", "dense", 0.9, 0.99)
	hybrid_rerank = make_run("rerank-run", "hybrid_rerank", 1.0, 0.983)

	class Client:
		def get_experiment_by_name(self, name: str) -> Any:
			assert name == "rag-retrieval-ablation"
			return SimpleNamespace(experiment_id="experiment-id")

		def search_runs(self, experiment_ids: list[str], **_: Any) -> list[Any]:
			assert experiment_ids == ["experiment-id"]
			return [dense, hybrid_rerank]

	assert select_best_run(Client(), "rag-retrieval-ablation") is hybrid_rerank
	assert build_strategy_config(hybrid_rerank)["retrieval_method"] == "hybrid_rerank"


def test_select_best_run_requires_finished_top_five_results() -> None:
	class Client:
		def get_experiment_by_name(self, _: str) -> Any:
			return SimpleNamespace(experiment_id="experiment-id")

		def search_runs(self, *_: Any, **__: Any) -> list[Any]:
			return []

	with pytest.raises(ValueError, match="no finished top-5"):
		select_best_run(Client(), "rag-retrieval-ablation")
