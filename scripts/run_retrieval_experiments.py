from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import chromadb
import mlflow
from chromadb.utils import embedding_functions
from langchain_core.documents import Document

from backend.rag.pdf_processor import (
	DEFAULT_EMBEDDING_MODEL,
	DEFAULT_EMBEDDING_URL,
	DEFAULT_RAW_DATA_DIR,
	build_documents,
)
from backend.rag.retriever import (
	bm25_score,
	compute_bm25_idf,
	fuse_rankings_by_rrf,
	tokenize_text,
)
from backend.rag.reranker import load_reranker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "retrieval_golden_dataset.json"
DEFAULT_TRACKING_DB = PROJECT_ROOT / "mlflow.db"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"
DEFAULT_RETRIEVAL_METHODS = ("dense", "hybrid", "hybrid_rerank")
DEFAULT_CHUNK_SIZES = (300, 500, 700)
DEFAULT_CHUNK_OVERLAPS = (50, 100)
DEFAULT_TOP_KS = (5, 10, 20)
DEFAULT_INDEX_SPACES = ("cosine", "l2", "ip")
DEFAULT_EF_SEARCH = (50, 100)
DEFAULT_EF_CONSTRUCTION = (100,)
DEFAULT_MAX_NEIGHBORS = (16,)
EMBEDDING_BATCH_SIZE = 64
INDEX_BATCH_SIZE = 128

SourceKey = tuple[str, str, str]


def parse_int_values(value: str) -> tuple[int, ...]:
	"""Parse a comma-separated list of positive integers."""
	try:
		values = tuple(int(part.strip()) for part in value.split(","))
	except ValueError as error:
		raise argparse.ArgumentTypeError("Expected comma-separated integers.") from error
	if not values or any(number < 1 for number in values):
		raise argparse.ArgumentTypeError("Values must be positive integers.")
	if len(set(values)) != len(values):
		raise argparse.ArgumentTypeError("Values must not contain duplicates.")
	return values


def parse_non_negative_values(value: str) -> tuple[int, ...]:
	"""Parse a comma-separated list of unique non-negative integers."""
	try:
		values = tuple(int(part.strip()) for part in value.split(","))
	except ValueError as error:
		raise argparse.ArgumentTypeError("Expected comma-separated integers.") from error
	if not values or any(number < 0 for number in values):
		raise argparse.ArgumentTypeError("Values must be non-negative integers.")
	if len(set(values)) != len(values):
		raise argparse.ArgumentTypeError("Values must not contain duplicates.")
	return values


def parse_index_spaces(value: str) -> tuple[str, ...]:
	"""Parse and validate a comma-separated list of Chroma HNSW spaces."""
	spaces = tuple(space.strip().lower() for space in value.split(","))
	if not spaces or any(space not in {"cosine", "l2", "ip"} for space in spaces):
		raise argparse.ArgumentTypeError("Spaces must be cosine, l2, or ip.")
	if len(set(spaces)) != len(spaces):
		raise argparse.ArgumentTypeError("Spaces must not contain duplicates.")
	return spaces


def load_golden_dataset(dataset_path: Path) -> dict[str, Any]:
	"""Load and validate the golden retrieval dataset structure."""
	dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
	if not isinstance(dataset, dict) or not isinstance(dataset.get("cases"), list):
		raise ValueError("Golden dataset must be a JSON object with a cases list.")
	if not dataset["cases"]:
		raise ValueError("Golden dataset must contain at least one case.")

	seen_ids: set[str] = set()
	for case in dataset["cases"]:
		if not isinstance(case, dict):
			raise ValueError("Every golden dataset case must be an object.")
		case_id = case.get("id")
		if not isinstance(case_id, str) or not case_id.strip() or case_id in seen_ids:
			raise ValueError("Golden dataset case IDs must be non-empty and unique.")
		seen_ids.add(case_id)
		if not isinstance(case.get("query"), str) or not case["query"].strip():
			raise ValueError(f"Case {case_id!r} must have a non-empty query.")
		sources = case.get("relevant_sources")
		if not isinstance(sources, list) or not sources:
			raise ValueError(f"Case {case_id!r} must have relevant_sources.")
		for source in sources:
			if not isinstance(source, dict) or any(
				not isinstance(source.get(field), str) or not source[field]
				for field in ("course", "section", "source_file")
			):
				raise ValueError(
					f"Every relevant source in case {case_id!r} must include "
					"course, section, and source_file."
				)
	return dataset


def source_key(metadata: dict[str, Any]) -> SourceKey:
	"""Normalize source identity, ignoring section whitespace from OCR artifacts."""
	return (
		str(metadata.get("course", "")),
		re.sub(r"\s+", "", str(metadata.get("section", ""))).casefold(),
		str(metadata.get("source_file", "")),
	)


def validate_labels_against_documents(
	cases: Sequence[dict[str, Any]],
	documents: Sequence[Any],
) -> None:
	"""Ensure every gold source remains represented after re-chunking."""
	available_sources = {source_key(document.metadata) for document in documents}
	missing = [
		(case["id"], source_key(source))
		for case in cases
		for source in case["relevant_sources"]
		if source_key(source) not in available_sources
	]
	if missing:
		raise ValueError(f"Golden labels absent from re-chunked documents: {missing}")


def compute_retrieval_metrics(
	cases: Sequence[dict[str, Any]],
	retrieved_metadata: Sequence[Sequence[dict[str, Any]]],
	top_k: int,
) -> dict[str, float]:
	"""Compute source recall, hit rate, and MRR at the requested chunk cutoff."""
	if top_k < 1:
		raise ValueError("top_k must be positive.")
	if len(cases) != len(retrieved_metadata):
		raise ValueError("Each golden case must have exactly one retrieval result list.")
	if not cases:
		raise ValueError("Cannot calculate metrics for an empty dataset.")

	hits = 0
	source_recalls: list[float] = []
	reciprocal_ranks: list[float] = []
	for case, result_metadata in zip(cases, retrieved_metadata):
		relevant = {source_key(source) for source in case["relevant_sources"]}
		ranked = list(result_metadata[:top_k])
		ranked_keys = [source_key(metadata) for metadata in ranked]
		retrieved_sources = set(ranked_keys)
		matched_sources = relevant & retrieved_sources
		hits += bool(matched_sources)
		source_recalls.append(len(matched_sources) / len(relevant))
		first_relevant_rank = next(
			(rank for rank, key in enumerate(ranked_keys, start=1) if key in relevant),
			None,
		)
		reciprocal_ranks.append(1.0 / first_relevant_rank if first_relevant_rank else 0.0)

	case_count = len(cases)
	return {
		"hit_rate": hits / case_count,
		"source_recall": sum(source_recalls) / case_count,
		"mrr": sum(reciprocal_ranks) / case_count,
	}


def _embed_texts(
	embedding_function: Any,
	texts: Sequence[str],
) -> tuple[list[list[float]], float]:
	"""Embed texts in batches and report elapsed embedding time."""
	embeddings: list[list[float]] = []
	started_at = time.perf_counter()
	for offset in range(0, len(texts), EMBEDDING_BATCH_SIZE):
		batch = embedding_function(list(texts[offset : offset + EMBEDDING_BATCH_SIZE]))
		embeddings.extend(vector.tolist() for vector in batch)
	return embeddings, time.perf_counter() - started_at


def _index_documents(
	collection: Any,
	documents: Sequence[Any],
	embeddings: Sequence[list[float]],
) -> None:
	"""Insert precomputed embeddings and document metadata into a Chroma index."""
	for offset in range(0, len(documents), INDEX_BATCH_SIZE):
		batch = documents[offset : offset + INDEX_BATCH_SIZE]
		collection.add(
			ids=[f"chunk::{index}" for index in range(offset, offset + len(batch))],
			documents=[document.page_content for document in batch],
			metadatas=[document.metadata for document in batch],
			embeddings=list(embeddings[offset : offset + len(batch)]),
		)


def _retrieve_metadata(
	collection: Any,
	query_embeddings: Sequence[list[float]],
	candidate_k: int,
) -> tuple[list[list[Document]], float]:
	"""Retrieve a fixed-size semantic candidate pool and report mean latency."""
	results: list[list[Document]] = []
	total_seconds = 0.0
	for query_embedding in query_embeddings:
		started_at = time.perf_counter()
		result = collection.query(
			query_embeddings=[query_embedding],
			n_results=candidate_k,
			include=["documents", "metadatas"],
		)
		total_seconds += time.perf_counter() - started_at
		documents = result["documents"][0] or []
		metadatas = result["metadatas"][0] or []
		results.append(
			[
				Document(page_content=text, metadata=metadata or {})
				for text, metadata in zip(documents, metadatas)
			]
		)
	return results, total_seconds * 1000 / len(query_embeddings)


def _bm25_rank(
	query: str,
	documents: Sequence[Document],
	document_tokens: Sequence[list[str]],
	idf: dict[str, float],
	avgdl: float,
	top_k: int,
) -> list[Document]:
	"""Rank the corpus with the same BM25 scoring used by the RAG retriever."""
	query_terms = tokenize_text(query)
	if not query_terms:
		return list(documents[:top_k])
	scores = [
		bm25_score(query_terms, tokens, idf, avgdl)
		for tokens in document_tokens
	]
	order = sorted(
		range(len(documents)),
		key=lambda index: (-scores[index], index),
	)
	return [documents[index] for index in order[:top_k]]


def _rank_candidates(
	*,
	method: str,
	query: str,
	semantic_candidates: Sequence[Document],
	corpus_documents: Sequence[Document],
	document_tokens: Sequence[list[str]],
	idf: dict[str, float],
	avgdl: float,
	candidate_k: int,
	final_k: int,
) -> tuple[list[Document], float]:
	"""Apply dense, hybrid RRF, or production hybrid-plus-cross-encoder ranking."""
	if method == "dense":
		candidates = list(semantic_candidates[:candidate_k])
	elif method in {"hybrid", "hybrid_rerank"}:
		bm25_candidates = _bm25_rank(
			query,
			corpus_documents,
			document_tokens,
			idf,
			avgdl,
			candidate_k,
		)
		candidates = fuse_rankings_by_rrf(
			[list(semantic_candidates), bm25_candidates],
			top_k=candidate_k,
		)
	else:
		raise ValueError(f"Unknown retrieval method: {method}")

	return candidates[:final_k], 0.0


def _rerank_candidate_lists(
	cases: Sequence[dict[str, Any]],
	candidate_lists: Sequence[Sequence[Document]],
	reranker: Any,
) -> tuple[list[list[Document]], float]:
	"""Score all query-candidate pairs in one cross-encoder batch."""
	pairs = [
		[case["query"], document.page_content]
		for case, candidates in zip(cases, candidate_lists)
		for document in candidates
	]
	started_at = time.perf_counter()
	scores = reranker.predict(pairs)
	elapsed = time.perf_counter() - started_at

	ranked_lists: list[list[Document]] = []
	offset = 0
	for candidates in candidate_lists:
		candidate_scores = scores[offset : offset + len(candidates)]
		offset += len(candidates)
		ordered_indices = sorted(
			range(len(candidates)),
			key=lambda index: float(candidate_scores[index]),
			reverse=True,
		)
		ranked_lists.append([candidates[index] for index in ordered_indices])
	return ranked_lists, elapsed


def _build_case_evaluations(
	cases: Sequence[dict[str, Any]],
	ranked_documents: Sequence[Sequence[Document]],
	top_k: int,
) -> list[dict[str, Any]]:
	"""Build per-query predictions with source labels for auditing the metrics."""
	evaluations: list[dict[str, Any]] = []
	for case, documents in zip(cases, ranked_documents):
		relevant = {source_key(source) for source in case["relevant_sources"]}
		ranked_sources = [
			source_key(document.metadata)
			for document in documents[:top_k]
		]
		first_relevant_rank = next(
			(rank for rank, source in enumerate(ranked_sources, start=1) if source in relevant),
			None,
		)
		evaluations.append(
			{
				"case_id": case["id"],
				"query": case["query"],
				"relevant_sources": [
					{
						"course": source["course"],
						"section": source["section"],
						"source_file": source["source_file"],
					}
					for source in case["relevant_sources"]
				],
				"retrieved_sources": [
					{
						"rank": rank,
						"course": document.metadata.get("course", ""),
						"section": document.metadata.get("section", ""),
						"source_file": document.metadata.get("source_file", ""),
						"chunk_id": document.metadata.get("chunk_id", ""),
						"relevant": source_key(document.metadata) in relevant,
					}
					for rank, document in enumerate(documents[:top_k], start=1)
				],
				"first_relevant_rank": first_relevant_rank,
			}
		)
	return evaluations


def _run_index_sweep(
	*,
	client: Any,
	cases: Sequence[dict[str, Any]],
	documents: Sequence[Any],
	document_embeddings: Sequence[list[float]],
	query_embeddings: Sequence[list[float]],
	chunk_size: int,
	chunk_overlap: int,
	spaces: Sequence[str],
	ef_search_values: Sequence[int],
	ef_construction_values: Sequence[int],
	max_neighbors_values: Sequence[int],
	top_ks: Sequence[int],
	candidate_k: int,
	retrieval_methods: Sequence[str],
	reranker: Any | None,
	corpus_documents: Sequence[Document],
	document_tokens: Sequence[list[str]],
	idf: dict[str, float],
	avgdl: float,
	dataset_hash: str,
	document_embedding_seconds: float,
	reranker_model: str,
	reranker_device: str,
) -> int:
	"""Build isolated HNSW indexes and track one MLflow run per search setting."""
	run_count = 0
	for space, ef_construction, max_neighbors in itertools.product(
		spaces,
		ef_construction_values,
		max_neighbors_values,
	):
		collection_name = f"retrieval_{run_count}_{time.time_ns()}"
		collection = client.create_collection(
			name=collection_name,
			embedding_function=None,
			configuration={
				"hnsw": {
					"space": space,
					"ef_construction": ef_construction,
					"max_neighbors": max_neighbors,
				}
			},
		)
		index_started_at = time.perf_counter()
		try:
			_index_documents(collection, documents, document_embeddings)
			index_build_seconds = time.perf_counter() - index_started_at
			for ef_search in ef_search_values:
				collection.modify(configuration={"hnsw": {"ef_search": ef_search}})
				semantic_rankings, mean_dense_latency_ms = _retrieve_metadata(
					collection,
					query_embeddings,
					candidate_k,
				)
				for method in retrieval_methods:
					candidate_lists: list[list[Document]] = []
					for case, semantic_candidates in zip(cases, semantic_rankings):
						candidates, _ = _rank_candidates(
							method=method,
							query=case["query"],
							semantic_candidates=semantic_candidates[:candidate_k],
							corpus_documents=corpus_documents,
							document_tokens=document_tokens,
							idf=idf,
							avgdl=avgdl,
							candidate_k=candidate_k,
							final_k=candidate_k,
						)
						candidate_lists.append(candidates)
					if method == "hybrid_rerank":
						if reranker is None:
							raise ValueError("hybrid_rerank requires an initialized reranker.")
						ranked_per_query, rerank_seconds = _rerank_candidate_lists(
							cases,
							candidate_lists,
							reranker,
						)
					else:
						ranked_per_query = candidate_lists
						rerank_seconds = 0.0
					mean_rerank_latency_ms = rerank_seconds * 1000 / len(cases)
					for top_k in top_ks:
						retrieved_metadata = [
							[document.metadata for document in ranked[:top_k]]
							for ranked in ranked_per_query
						]
						metrics = compute_retrieval_metrics(cases, retrieved_metadata, top_k)
						run_name = (
							f"{method}-chunk-{chunk_size}-overlap-{chunk_overlap}-"
							f"{space}-efc-{ef_construction}-efs-{ef_search}-"
							f"candidate-{candidate_k}-k-{top_k}"
						)
						with mlflow.start_run(run_name=run_name, nested=True):
							mlflow.log_params(
								{
									"retrieval_method": method,
									"reranker_enabled": method == "hybrid_rerank",
									"reranker_model": (
										reranker_model
										if method == "hybrid_rerank"
										else "none"
									),
									"reranker_device": (
										reranker_device if method == "hybrid_rerank" else "none"
									),
									"index_type": "hnsw",
									"index_space": space,
									"chunk_size": chunk_size,
									"chunk_overlap": chunk_overlap,
									"candidate_k": candidate_k,
									"top_k": top_k,
									"ef_search": ef_search,
									"ef_construction": ef_construction,
									"max_neighbors": max_neighbors,
									"indexed_chunks": len(documents),
									"golden_cases": len(cases),
								}
							)
							mlflow.set_tags(
								{
									"dataset_name": "retrieval-golden-v1",
									"dataset_sha256": dataset_hash,
									"metric_scope": "source-section retrieval relevance",
								}
							)
							mlflow.log_metrics(
								{
									f"hit_rate_at_{top_k}": metrics["hit_rate"],
									f"source_recall_at_{top_k}": metrics["source_recall"],
									f"mrr_at_{top_k}": metrics["mrr"],
									"mean_dense_query_latency_ms": mean_dense_latency_ms,
									"mean_rerank_latency_ms": mean_rerank_latency_ms,
									"mean_total_query_latency_ms": (
										mean_dense_latency_ms + mean_rerank_latency_ms
									),
									"index_build_seconds": index_build_seconds,
									"document_embedding_seconds": document_embedding_seconds,
								}
							)
							mlflow.log_dict(
								{
									"retrieval_method": method,
									"candidate_k": candidate_k,
									"top_k": top_k,
									"metrics": metrics,
									"cases": _build_case_evaluations(
										cases,
										ranked_per_query,
										top_k,
									),
								},
								"evaluation/retrieval_results.json",
							)
						run_count += 1
		finally:
			client.delete_collection(collection_name)
	return run_count


def run_experiments(args: argparse.Namespace) -> int:
	"""Run the configured chunking and Chroma HNSW parameter sweep."""
	dataset = load_golden_dataset(args.dataset)
	cases = dataset["cases"][: args.limit_queries] if args.limit_queries else dataset["cases"]
	if not cases:
		raise ValueError("No golden cases selected for evaluation.")
	if max(args.chunk_overlap) >= min(args.chunk_size):
		raise ValueError("Every chunk overlap must be smaller than every chunk size.")
	if args.candidate_k < max(args.top_k):
		raise ValueError("--candidate-k must be greater than or equal to every --top-k.")
	if not args.retrieval_methods:
		raise ValueError("At least one retrieval method must be selected.")
	if any(method not in {"dense", "hybrid", "hybrid_rerank"} for method in args.retrieval_methods):
		raise ValueError("Retrieval methods must be dense, hybrid, or hybrid_rerank.")

	embedding_function = embedding_functions.OllamaEmbeddingFunction(
		model_name=args.embedding_model,
		url=args.embedding_url,
	)
	tracking_uri = args.tracking_uri or f"sqlite:///{args.tracking_db.resolve().as_posix()}"
	mlflow.set_tracking_uri(tracking_uri)
	mlflow.set_experiment(args.experiment_name)

	query_embeddings, query_embedding_seconds = _embed_texts(
		embedding_function,
		[case["query"] for case in cases],
	)
	reranker = None
	reranker_device = "none"
	if "hybrid_rerank" in args.retrieval_methods:
		reranker, reranker_device = load_reranker(
			args.reranker_model,
			local_files_only=True,
		)
	client = chromadb.EphemeralClient()
	dataset_hash = hashlib.sha256(args.dataset.read_bytes()).hexdigest()

	total_runs = 0
	with mlflow.start_run(run_name="retrieval-tuning-sweep") as parent_run:
		mlflow.set_tags(
			{
				"dataset_name": str(dataset.get("name", "unnamed-golden-dataset")),
				"dataset_sha256": dataset_hash,
				"index_type": "chroma-hnsw",
				"query_embedding_model": args.embedding_model,
				"metric_scope": "source-section retrieval relevance",
			}
		)
		mlflow.log_param("golden_cases", len(cases))
		mlflow.log_param("reranker_device", reranker_device)
		mlflow.log_metric("query_embedding_seconds", query_embedding_seconds)
		mlflow.log_artifact(str(args.dataset), artifact_path="evaluation")

		for chunk_size, chunk_overlap in itertools.product(
			args.chunk_size,
			args.chunk_overlap,
		):
			_, _, documents = build_documents(
				args.raw_data_dir,
				chunk_size_tokens=chunk_size,
				chunk_overlap_tokens=chunk_overlap,
			)
			if max(args.top_k) > len(documents):
				raise ValueError(
					f"Configured top_k={max(args.top_k)} exceeds the "
					f"{len(documents)} re-chunked documents."
				)
			validate_labels_against_documents(cases, documents)
			document_embeddings, embedding_seconds = _embed_texts(
				embedding_function,
				[document.page_content for document in documents],
			)
			document_tokens = [tokenize_text(document.page_content) for document in documents]
			idf = compute_bm25_idf(document_tokens)
			avgdl = sum(len(tokens) for tokens in document_tokens) / max(1, len(document_tokens))
			runs_for_chunking = _run_index_sweep(
				client=client,
				cases=cases,
				documents=documents,
				document_embeddings=document_embeddings,
				query_embeddings=query_embeddings,
				chunk_size=chunk_size,
				chunk_overlap=chunk_overlap,
				spaces=args.spaces,
				ef_search_values=args.ef_search,
				ef_construction_values=args.ef_construction,
				max_neighbors_values=args.max_neighbors,
				top_ks=args.top_k,
				candidate_k=args.candidate_k,
				retrieval_methods=args.retrieval_methods,
				reranker=reranker,
				corpus_documents=documents,
				document_tokens=document_tokens,
				idf=idf,
				avgdl=avgdl,
				dataset_hash=dataset_hash,
				document_embedding_seconds=embedding_seconds,
				reranker_model=args.reranker_model,
				reranker_device=reranker_device,
			)
			total_runs += runs_for_chunking
			mlflow.log_metric(
				"document_embedding_seconds",
				embedding_seconds,
				step=chunk_size * 1000 + chunk_overlap,
			)

		print(
			json.dumps(
				{
					"experiment_name": args.experiment_name,
					"tracking_uri": tracking_uri,
					"parent_run_id": parent_run.info.run_id,
					"child_run_count": total_runs,
					"golden_case_count": len(cases),
				},
				indent=2,
			)
		)
	return total_runs


def build_argument_parser() -> argparse.ArgumentParser:
	"""Build CLI options for reproducible retrieval tuning."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
	parser.add_argument("--raw-data-dir", type=Path, default=DEFAULT_RAW_DATA_DIR)
	parser.add_argument("--tracking-db", type=Path, default=DEFAULT_TRACKING_DB)
	parser.add_argument("--tracking-uri")
	parser.add_argument("--experiment-name", default="rag-retrieval-ablation")
	parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
	parser.add_argument("--embedding-url", default=DEFAULT_EMBEDDING_URL)
	parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
	parser.add_argument(
		"--retrieval-methods",
		type=lambda value: tuple(part.strip().lower() for part in value.split(",")),
		default=DEFAULT_RETRIEVAL_METHODS,
		help="Comma-separated ablations: dense, hybrid, hybrid_rerank.",
	)
	parser.add_argument("--chunk-size", type=parse_int_values, default=DEFAULT_CHUNK_SIZES)
	parser.add_argument("--chunk-overlap", type=parse_non_negative_values, default=DEFAULT_CHUNK_OVERLAPS)
	parser.add_argument("--top-k", type=parse_int_values, default=DEFAULT_TOP_KS)
	parser.add_argument("--candidate-k", type=int, default=20)
	parser.add_argument("--spaces", type=parse_index_spaces, default=DEFAULT_INDEX_SPACES)
	parser.add_argument("--ef-search", type=parse_int_values, default=DEFAULT_EF_SEARCH)
	parser.add_argument(
		"--ef-construction",
		type=parse_int_values,
		default=DEFAULT_EF_CONSTRUCTION,
	)
	parser.add_argument("--max-neighbors", type=parse_int_values, default=DEFAULT_MAX_NEIGHBORS)
	parser.add_argument(
		"--limit-queries",
		type=int,
		help="Optional positive limit for a quick smoke test; defaults to all golden cases.",
	)
	return parser


def main() -> None:
	"""CLI entry point."""
	args = build_argument_parser().parse_args()
	if args.limit_queries is not None and args.limit_queries < 1:
		raise ValueError("--limit-queries must be positive.")
	if not args.dataset.is_file():
		raise FileNotFoundError(f"Golden dataset was not found: {args.dataset}")
	if not args.raw_data_dir.is_dir():
		raise FileNotFoundError(f"Raw course data directory was not found: {args.raw_data_dir}")
	run_experiments(args)


if __name__ == "__main__":
	main()
