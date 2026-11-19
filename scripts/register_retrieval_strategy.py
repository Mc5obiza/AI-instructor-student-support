from __future__ import annotations

import argparse
import json
from typing import Any

import mlflow
import pandas as pd
from mlflow.pyfunc import PythonModel
from mlflow.tracking import MlflowClient

from backend.rag.retrieval_strategy import (
	REGISTERED_MODEL_ALIAS,
	REGISTERED_MODEL_NAME,
	STRATEGY_CONFIG_TAG,
	default_tracking_uri,
)

DEFAULT_EXPERIMENT_NAME = "rag-retrieval-ablation"
FINAL_TOP_K = 5


class RetrievalStrategyManifest(PythonModel):
	"""MLflow model packaging the selected retrieval strategy configuration."""

	def __init__(self, config: dict[str, Any]) -> None:
		self.config = config

	def predict(
		self,
		context: Any,
		model_input: pd.DataFrame,
		params: dict[str, Any] | None = None,
	) -> pd.DataFrame:
		"""Return the registered configuration for each supplied query row."""
		return pd.DataFrame(
			[self.config.copy() for _ in range(len(model_input))],
			index=model_input.index,
		)


def select_best_run(client: MlflowClient, experiment_name: str) -> Any:
	"""Select the best finished top-5 run by hit rate, then MRR."""
	experiment = client.get_experiment_by_name(experiment_name)
	if experiment is None:
		raise ValueError(f"MLflow experiment {experiment_name!r} was not found.")

	runs = client.search_runs(
		[experiment.experiment_id],
		filter_string=f"attributes.status = 'FINISHED' and params.top_k = '{FINAL_TOP_K}'",
		max_results=1000,
	)
	candidates = [
		run for run in runs
		if run.data.params.get("retrieval_method") in {"dense", "hybrid", "hybrid_rerank"}
		and "hit_rate_at_5" in run.data.metrics
		and "mrr_at_5" in run.data.metrics
	]
	if not candidates:
		raise ValueError(
			f"Experiment {experiment_name!r} has no finished top-{FINAL_TOP_K} retrieval runs."
		)
	return max(
		candidates,
		key=lambda run: (
			run.data.metrics["hit_rate_at_5"],
			run.data.metrics["mrr_at_5"],
		),
	)


def build_strategy_config(run: Any) -> dict[str, Any]:
	"""Build the runtime settings and evidence from the selected evaluation run."""
	params = run.data.params
	metrics = run.data.metrics
	try:
		config = {
			"retrieval_method": params["retrieval_method"],
			"candidate_k": int(params["candidate_k"]),
			"top_k": int(params["top_k"]),
			"reranker_model": params.get("reranker_model", "none"),
			"source_run_id": run.info.run_id,
			"hit_rate_at_5": metrics["hit_rate_at_5"],
			"mrr_at_5": metrics["mrr_at_5"],
			"mean_total_query_latency_ms": metrics.get("mean_total_query_latency_ms"),
			"chunk_size": int(params["chunk_size"]),
			"chunk_overlap": int(params["chunk_overlap"]),
			"index_space": params["index_space"],
			"ef_search": int(params["ef_search"]),
		}
	except (KeyError, TypeError, ValueError) as error:
		raise ValueError(f"Selected MLflow run {run.info.run_id} is missing required settings.") from error

	if config["retrieval_method"] != "hybrid_rerank":
		config["reranker_model"] = None
	return config


def register_strategy(
	tracking_uri: str | None = None,
	experiment_name: str = DEFAULT_EXPERIMENT_NAME,
	model_name: str = REGISTERED_MODEL_NAME,
	alias: str = REGISTERED_MODEL_ALIAS,
) -> dict[str, Any]:
	"""Register the best measured strategy and point the active alias to it."""
	resolved_tracking_uri = tracking_uri or default_tracking_uri()
	mlflow.set_tracking_uri(resolved_tracking_uri)
	mlflow.set_experiment(experiment_name)
	client = MlflowClient(tracking_uri=resolved_tracking_uri)
	selected_run = select_best_run(client, experiment_name)
	config = build_strategy_config(selected_run)

	with mlflow.start_run(run_name="register-retrieval-strategy") as registration_run:
		mlflow.set_tags(
			{
				"source_evaluation_run_id": selected_run.info.run_id,
				"retrieval_method": config["retrieval_method"],
				"model_type": "retrieval-strategy-manifest",
			}
		)
		mlflow.log_dict(config, "strategy/retrieval_strategy.json")
		mlflow.pyfunc.log_model(
			name="retrieval_strategy",
			python_model=RetrievalStrategyManifest(config),
			input_example=pd.DataFrame({"query": ["retrieval strategy"]}),
			registered_model_name=model_name,
		)
		registration_run_id = registration_run.info.run_id

	versions = client.search_model_versions(f"name='{model_name}'")
	registered_versions = [
		version for version in versions if version.run_id == registration_run_id
	]
	if not registered_versions:
		raise RuntimeError(f"MLflow did not create a version for model {model_name!r}.")
	model_version = max(registered_versions, key=lambda version: int(version.version))
	client.set_model_version_tag(
		model_name,
		model_version.version,
		STRATEGY_CONFIG_TAG,
		json.dumps(config, sort_keys=True),
	)
	client.set_model_version_tag(
		model_name,
		model_version.version,
		"source_evaluation_run_id",
		selected_run.info.run_id,
	)
	client.update_model_version(
		model_name,
		model_version.version,
		description=(
			f"Selected from {experiment_name}; top-5 hit rate "
			f"{config['hit_rate_at_5']:.3f}, MRR {config['mrr_at_5']:.3f}."
		),
	)
	client.set_registered_model_alias(model_name, alias, model_version.version)

	return {
		"tracking_uri": resolved_tracking_uri,
		"model_name": model_name,
		"version": model_version.version,
		"alias": alias,
		"source_run_id": selected_run.info.run_id,
		"registration_run_id": registration_run_id,
		"strategy": config,
	}


def build_argument_parser() -> argparse.ArgumentParser:
	"""Build command-line options for strategy registration."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--tracking-uri", default=None)
	parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
	parser.add_argument("--model-name", default=REGISTERED_MODEL_NAME)
	parser.add_argument("--alias", default=REGISTERED_MODEL_ALIAS)
	return parser


def main() -> None:
	"""Register the winning strategy from the selected MLflow experiment."""
	args = build_argument_parser().parse_args()
	result = register_strategy(
		tracking_uri=args.tracking_uri,
		experiment_name=args.experiment_name,
		model_name=args.model_name,
		alias=args.alias,
	)
	print(json.dumps(result, indent=2))


if __name__ == "__main__":
	main()
