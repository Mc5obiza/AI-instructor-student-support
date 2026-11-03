# Retrieval Experiment Report

**Date:** 2026-11-03 | **Project:** AI LilMose3da
**Tracking store:** repository-root `mlflow.db`
**Experiments:** `rag-retrieval-tuning`, `rag-retrieval-ablation`

## Executive summary

The experiments show that retrieval quality on the current 30-query golden set improves from dense-only retrieval to hybrid dense + BM25 fusion, and improves again when the hybrid candidates are reranked with `BAAI/bge-reranker-base`.

- At **top-5**, hybrid retrieval with the cross-encoder achieved **100% hit rate/source recall and 0.983 MRR**, compared with **90% and 0.844 MRR** for hybrid without reranking.
- The reranker improved the first relevant source's rank for **6 of 30** queries, worsened **none**, and left **24** unchanged relative to hybrid at top-5.
- The reranker's cost was substantial in this CPU-only run: the measured batch-inference wall time amortized to about **11.4 seconds per query** for 20 candidates. The experiment demonstrates a quality improvement on this dataset, not that the added latency is acceptable in production.
- The separate chunk/index sweep found **300-token chunks with 50-token overlap** to be the strongest tested chunking on average at top-20. The best individual top-20 configuration reached **96.7% source recall**.
- These are exploratory results on a small, single-source-per-query dataset. Validate on a larger independently reviewed golden set and measure per-request latency on deployment hardware before changing production defaults.

## Dataset and label definition

The evaluation used [`data/processed/retrieval_golden_dataset.json`](../data/processed/retrieval_golden_dataset.json), dataset name `retrieval-golden-v1`.

| Property | Value |
|---|---|
| Golden queries | 30 |
| Indexed source snapshot | 712 chunks at the 500/100 setting |
| Labeled source fields | `course`, `section`, `source_file` |
| Label identity | Section comparison removes whitespace and ignores case; course and source file must match |
| Chunk ID in labels | Excluded, so labels survive re-chunking |
| Dataset SHA-256 | `f2041ab5910370009ec38b673d6a1a4ed79cd0a4ba8e59d610049eee3345fb99` |

Every query in the current dataset has one labeled relevant source-section. As a result, mean source recall and hit rate have the same value in these runs. This dataset measures retrieval relevance only; it does not grade generated answers, citation correctness, or answer faithfulness.

## Method

All commands ran in the existing `agents_env` Conda environment. Embeddings used the locally served Ollama `nomic-embed-text` model (768 dimensions). The existing production Chroma database was not changed; experiment indexes were temporary in-memory Chroma collections.

### Chunk and index sweep

The corrected sweep varied:

- Chunk sizes: `300`, `500`, `700` whitespace-counted token-like units.
- Overlap sizes: `50`, `100`.
- HNSW distance spaces: cosine, L2, inner product.
- HNSW `ef_search`: `50`, `100`.
- Final `top_k`: `5`, `10`, `20`.

HNSW `ef_construction` remained `100` and `max_neighbors` remained `16`. The design gives `3 × 2 × 3 × 2 × 3 = 108` configuration runs. Each run logged its parameters, retrieval metrics, and timing measurements to MLflow.

### Retrieval-method ablation

The ablation held the following settings fixed across methods:

- Chunk size/overlap: `500/100`.
- Index: Chroma HNSW cosine, `ef_construction=100`, `ef_search=50`, `max_neighbors=16`.
- Candidate pool: 20 chunks.
- Final `top_k`: `5`, `10`, or `20`.
- Golden queries: all 30.

The methods were:

1. **Dense:** Chroma semantic ranking only.
2. **Hybrid:** dense candidates combined with a full-corpus BM25 ranking using reciprocal rank fusion (RRF).
3. **Hybrid + reranker:** the same hybrid candidate set, reordered by the project's `BAAI/bge-reranker-base` CrossEncoder.

This design isolates whether reranking changes the order of the hybrid candidate set; it does not test whether reranking can recover relevant chunks absent from that set.

## Results: chunk and index sweep

The corrected sweep parent run is `5aa6007a71b4441fb21f0e10674b5133`. All 108 child configuration runs completed.

### Best observed configuration by final top-k

| Final top-k | Chunk / overlap | HNSW space | `ef_search` | Source recall / hit rate | MRR | Mean query latency |
|---:|---:|---|---:|---:|---:|---:|
| 5 | 300 / 50 | Inner product (`ip`) | 50 | 93.3% | 0.803 | 1.10 ms |
| 10 | 300 / 50 | Inner product (`ip`) | 50 | 93.3% | 0.803 | 0.83 ms |
| 20 | 300 / 50 | L2 | 100 | 96.7% | 0.805 | 1.10 ms |

The top-5 and top-10 winners had the same relevance scores and similar latency as other tied settings. They are the fastest tied winners observed by the sweep's latency measurements, not statistically established global optima.

### Chunk-size averages at top-20

Each row averages 12 settings: two overlaps, three distance spaces, and two `ef_search` values.

| Chunk size | Mean source recall / hit rate | Mean MRR | Mean query latency |
|---:|---:|---:|---:|
| 300 | 95.0% | 0.804 | 8.15 ms |
| 500 | 93.3% | 0.800 | 8.23 ms |
| 700 | 93.3% | 0.800 | 11.42 ms |

The 300/50 configuration is a promising candidate for further testing. The sweep does not establish that it should replace the existing 500/100 default: the query set is small, and the per-size means combine different HNSW spaces and search settings.

## Results: dense versus hybrid versus reranking

The ablation parent run is `fd5102d837e9419b89a35fcd1023415c`; all nine child runs finished. The table reports hit rate/source recall followed by MRR.

| Method | Top-5 hit rate / source recall | Top-5 MRR | Top-10 hit rate / source recall | Top-10 MRR | Top-20 hit rate / source recall | Top-20 MRR |
|---|---:|---:|---:|---:|---:|---:|
| Dense only | 90.0% | 0.794 | 93.3% | 0.800 | 93.3% | 0.800 |
| Dense + BM25/RRF | 90.0% | 0.844 | 100.0% | 0.857 | 100.0% | 0.857 |
| Hybrid + `bge-reranker-base` | **100.0%** | **0.983** | **100.0%** | **0.983** | **100.0%** | **0.983** |

### Reranker effect at top-5

Compared with hybrid retrieval without reranking:

- Hit rate/source recall increased from 90% to 100%: **+10 percentage points**.
- MRR increased from 0.844 to 0.983: **+0.139 absolute**.
- Six queries moved the first relevant source to a better position (including three queries with no relevant source in the hybrid top-5).
- Zero queries had a worse first relevant-source rank.
- Twenty-four queries were unchanged.

The reranker also improved top-5 MRR relative to dense-only retrieval by 0.189 absolute. Hybrid fusion alone improved top-10 coverage from 93.3% to 100% and top-5 MRR from 0.794 to 0.844.

### Representative MLflow child runs

| Method / top-k | Run ID |
|---|---|
| Dense / 5 | `d30f1fe5fc104ab0b4a9ef4622787fec` |
| Hybrid / 5 | `5fe3bd567cf94994a1717831155a44b9` |
| Hybrid + reranker / 5 | `2299a6933bdb454d8908ad58fba7721e` |
| Hybrid + reranker / 10 | `e065c17794ea41d3b6dc05ffb2c113c3` |
| Hybrid + reranker / 20 | `ced4f45a533a473aaf46eca04e2a5926` |

Every ablation child run logs `retrieval_method`, reranker status/model, index settings, chunk settings, candidate count and final top-k. It also logs hit rate, source recall, MRR, and an `evaluation/retrieval_results.json` artifact containing each query's gold sources, retrieved sources, ranks, and relevance flags.

## Runtime measurements and limitations

| Measurement | Dense / hybrid | Hybrid + reranker |
|---|---:|---:|
| Reported dense Chroma candidate-search time | 9.79 ms/query | 9.79 ms/query |
| Reported reranker time | 0 ms/query | 11,424.98 ms/query |
| Reported total | 9.79 ms/query | 11,434.77 ms/query |

Interpret the latency comparison cautiously:

1. The reranker scored all query/candidate pairs in a batched CPU inference call; its reported per-query value is batch wall time divided by 30. It is not a direct measurement of an isolated online request.
2. The recorded dense query time covers Chroma candidate retrieval. **BM25 scoring/RRF time is not included**, so the reported hybrid latency is not full end-to-end hybrid latency.
3. Timing depends on CPU load, warm-up, batch size, and runtime conditions. Repeat measurements under a defined deployment setup before using them for an SLA decision.
4. The tested methods stop at ranked source chunks. No LLM generation, answer quality, citation fidelity, or context-compression cost was measured.
5. Each golden query currently has one labeled source. The set covers only a small portion of the course corpus and does not provide confidence intervals or establish generalization.
6. This is a fixed-candidate reranking ablation. Since the reranker only reorders the 20 hybrid candidates, it cannot recover an answer source that is missing from those candidates.

## Failed run record

The `rag-retrieval-tuning` experiment contains one historical failed parent run: `cebdc98d153c4d919ec74d5ecdae8d44`. It failed before creating child runs because two gold section labels did not match the re-extracted source sections due to OCR whitespace inconsistencies. The label comparison was corrected to normalize section whitespace/case; the follow-up tuning sweep completed with 108 successful child runs. The failed run is retained for audit history and tagged with its cause and superseding sweep.

## Conclusions and recommended next steps

1. **Hybrid retrieval helps coverage:** dense + BM25/RRF reached full source coverage by top-10 in this set.
2. **The cross-encoder improved ranking quality here:** hybrid + `bge-reranker-base` achieved full top-5 coverage and an MRR of 0.983, up from 0.844 for hybrid without it.
3. **Quality comes with measured CPU cost:** the reranker is promising for quality-sensitive use, but this run does not support enabling it unconditionally in latency-sensitive production until inference is optimized and measured per request.
4. **Treat chunk tuning as provisional:** 300/50 performed best in this sweep, but a larger independently reviewed set should determine whether the difference is robust.
5. **Improve the next evaluation iteration:** add more queries per course, multiple relevant sources where appropriate, hard negatives, out-of-scope/unanswerable queries, and independent label review. Measure recall, MRR/nDCG, per-query latency distributions, and (separately) generated-answer faithfulness.
6. **Keep the evaluation comparisons reproducible:** use identical candidates for reranker on/off comparisons and log the dataset checksum and per-query ranked predictions, as the ablation runs now do.

## Reproduction and viewing

Run the chunk/index sweep or the controlled reranker comparison from the repository root:

```powershell
conda run -n agents_env --no-capture-output python -m scripts.run_retrieval_experiments
```

```powershell
conda run -n agents_env --no-capture-output python -m scripts.run_retrieval_experiments --chunk-size 500 --chunk-overlap 100 --top-k 5,10,20 --candidate-k 20 --spaces cosine --ef-search 50 --ef-construction 100 --max-neighbors 16 --experiment-name rag-retrieval-ablation --retrieval-methods dense,hybrid,hybrid_rerank
```

Start the MLflow UI against the same SQLite store (the explicit backend URI is important when an older `mlruns/` store exists):

```powershell
conda run -n agents_env --no-capture-output mlflow ui --backend-store-uri sqlite:///./mlflow.db
```

Select `rag-retrieval-tuning` for the parameter sweep or `rag-retrieval-ablation` for the reranker comparison. Each child run contains its parameter/metric record and per-query artifact.
