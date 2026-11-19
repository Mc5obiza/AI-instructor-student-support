from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REGISTERED_MODEL_NAME = "AIInstructorRetrievalStrategy"
REGISTERED_MODEL_ALIAS = "champion"
STRATEGY_CONFIG_TAG = "strategy_config"


@dataclass(frozen=True)
class RetrievalStrategy:
	"""Registered retrieval settings selected from an MLflow evaluation run."""

	retrieval_method: str
	candidate_k: int
	top_k: int
	reranker_model: str | None
	source_run_id: str

	@classmethod
	def from_config(cls, config: dict[str, Any]) -> RetrievalStrategy:
		"""Validate and parse the serialized registry configuration."""
		method = config.get("retrieval_method")
		if method not in {"dense", "hybrid", "hybrid_rerank"}:
			raise ValueError(f"Unsupported registered retrieval method: {method!r}")

		candidate_k = config.get("candidate_k")
		top_k = config.get("top_k")
		if (
			not isinstance(candidate_k, int)
			or isinstance(candidate_k, bool)
			or not isinstance(top_k, int)
			or isinstance(top_k, bool)
		):
			raise ValueError("Registered candidate_k and top_k must be integers.")
		if candidate_k < 1 or top_k < 1 or top_k > candidate_k:
			raise ValueError("Registered retrieval cutoffs must satisfy 1 <= top_k <= candidate_k.")

		reranker_model = config.get("reranker_model")
		if method == "hybrid_rerank" and (
			not isinstance(reranker_model, str) or not reranker_model.strip()
		):
			raise ValueError("The hybrid_rerank strategy must specify a reranker model.")
		if method != "hybrid_rerank":
			reranker_model = None

		source_run_id = config.get("source_run_id")
		if not isinstance(source_run_id, str) or not source_run_id.strip():
			raise ValueError("Registered retrieval strategy is missing its source run ID.")

		return cls(
			retrieval_method=method,
			candidate_k=candidate_k,
			top_k=top_k,
			reranker_model=reranker_model,
			source_run_id=source_run_id,
		)


def default_tracking_uri() -> str:
	"""Resolve the configured MLflow store or the repository-local tracking DB."""
	configured_uri = os.environ.get("MLFLOW_TRACKING_URI")
	if configured_uri:
		return configured_uri
	repository_root = Path(__file__).resolve().parents[2]
	return f"sqlite:///{(repository_root / 'mlflow.db').as_posix()}"


def load_registered_retrieval_strategy(
	tracking_uri: str | None = None,
	model_name: str = REGISTERED_MODEL_NAME,
	alias: str = REGISTERED_MODEL_ALIAS,
) -> RetrievalStrategy:
	"""Load the active strategy config from the MLflow Model Registry."""
	from mlflow.exceptions import MlflowException
	from mlflow.tracking import MlflowClient

	client = MlflowClient(tracking_uri=tracking_uri or default_tracking_uri())
	try:
		model_version = client.get_model_version_by_alias(model_name, alias)
	except MlflowException as error:
		raise RuntimeError(
			f"Could not resolve MLflow model {model_name!r} alias {alias!r}."
		) from error

	serialized_config = model_version.tags.get(STRATEGY_CONFIG_TAG)
	if not serialized_config:
		raise ValueError(
			f"MLflow model {model_name!r} version {model_version.version} "
			f"is missing the {STRATEGY_CONFIG_TAG!r} tag."
		)
	try:
		config = json.loads(serialized_config)
	except json.JSONDecodeError as error:
		raise ValueError("Registered retrieval strategy config is invalid JSON.") from error
	if not isinstance(config, dict):
		raise ValueError("Registered retrieval strategy config must be a JSON object.")
	return RetrievalStrategy.from_config(config)
