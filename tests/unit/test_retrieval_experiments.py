from __future__ import annotations

import argparse

import pytest

from scripts.run_retrieval_experiments import (
	_rank_candidates,
	_rerank_candidate_lists,
	compute_retrieval_metrics,
	parse_int_values,
	validate_labels_against_documents,
)
from langchain_core.documents import Document

pytestmark = pytest.mark.unit


def test_compute_retrieval_metrics_uses_stable_source_labels() -> None:
	cases = [
		{
			"id": "case-1",
			"relevant_sources": [
				{"course": "statistics", "section": "mean", "source_file": "stats.pdf"}
			],
		},
		{
			"id": "case-2",
			"relevant_sources": [
				{"course": "python", "section": "lists", "source_file": "python.pdf"}
			],
		},
	]
	retrieved = [
		[
			{"course": "statistics", "section": "other", "source_file": "stats.pdf"},
			{"course": "statistics", "section": "mean", "source_file": "stats.pdf"},
		],
		[{"course": "python", "section": "lists", "source_file": "python.pdf"}],
	]

	assert compute_retrieval_metrics(cases, retrieved, top_k=1) == {
		"hit_rate": 0.5,
		"source_recall": 0.5,
		"mrr": 0.5,
	}
	assert compute_retrieval_metrics(cases, retrieved, top_k=2) == {
		"hit_rate": 1.0,
		"source_recall": 1.0,
		"mrr": 0.75,
	}


def test_compute_retrieval_metrics_rejects_mismatched_results() -> None:
	with pytest.raises(ValueError, match="exactly one retrieval result list"):
		compute_retrieval_metrics([{"id": "case"}], [], top_k=5)


def test_compute_retrieval_metrics_ignores_section_whitespace_artifacts() -> None:
	cases = [
		{
			"id": "case-1",
			"relevant_sources": [
				{"course": "loss functions", "section": "3.classifi ction loss", "source_file": "Loss functions.pdf"}
			],
		}
	]
	retrieved = [
		[
			{
				"course": "loss functions",
				"section": "3.classifiction loss",
				"source_file": "Loss functions.pdf",
			}
		]
	]

	assert compute_retrieval_metrics(cases, retrieved, top_k=1) == {
		"hit_rate": 1.0,
		"source_recall": 1.0,
		"mrr": 1.0,
	}


def test_validate_labels_against_documents_reports_missing_sources() -> None:
	class Chunk:
		metadata = {"course": "statistics", "section": "mean", "source_file": "stats.pdf"}

	cases = [
		{
			"id": "case-1",
			"relevant_sources": [
				{"course": "python", "section": "lists", "source_file": "python.pdf"}
			],
		}
	]

	with pytest.raises(ValueError, match="absent from re-chunked documents"):
		validate_labels_against_documents(cases, [Chunk()])


def test_parse_int_values_requires_unique_positive_values() -> None:
	assert parse_int_values("5, 10,20") == (5, 10, 20)
	with pytest.raises(argparse.ArgumentTypeError, match="positive"):
		parse_int_values("0,5")
	with pytest.raises(argparse.ArgumentTypeError, match="duplicates"):
		parse_int_values("5,5")


def test_hybrid_reranker_changes_order_and_can_improve_reciprocal_rank() -> None:
	relevant = Document(
		page_content="The silhouette score measures cluster separation.",
		metadata={"course": "clustering", "section": "silhouette", "source_file": "metrics.pdf"},
	)
	irrelevant = Document(
		page_content="The score is mentioned but describes another statistic.",
		metadata={"course": "clustering", "section": "other", "source_file": "metrics.pdf"},
	)
	case = {
		"id": "silhouette",
		"query": "Explain this score",
		"relevant_sources": [
			{"course": "clustering", "section": "silhouette", "source_file": "metrics.pdf"}
		],
	}
	corpus = [irrelevant, relevant]
	tokens = [["score", "mentioned"], ["silhouette", "cluster", "separation"]]
	idf = {"silhouette": 1.0, "cluster": 1.0, "separation": 1.0, "score": 1.0}
	semantic = [irrelevant, relevant]

	class RelevantFirstReranker:
		def predict(self, pairs: list[list[str]]) -> list[int]:
			return [int("silhouette" in text.lower()) for _, text in pairs]

	unreranked, _ = _rank_candidates(
		method="hybrid",
		query=case["query"],
		semantic_candidates=semantic,
		corpus_documents=corpus,
		document_tokens=tokens,
		idf=idf,
		avgdl=2.5,
		candidate_k=2,
		final_k=2,
	)
	reranked, _ = _rerank_candidate_lists(
		[case],
		[unreranked],
		RelevantFirstReranker(),
	)

	assert [doc.page_content for doc in unreranked] != [
		doc.page_content for doc in reranked[0]
	]
	assert compute_retrieval_metrics(
		[case],
		[[doc.metadata for doc in unreranked]],
		top_k=1,
	)["mrr"] == 0.0
	assert compute_retrieval_metrics(
		[case],
		[[doc.metadata for doc in reranked[0]]],
		top_k=1,
	)["mrr"] == 1.0
