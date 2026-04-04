# Copilot Instructions

## Project Overview
This project is an AI-powered learning assistant built with a Retrieval-Augmented Generation (RAG) architecture.

The system allows students to ask questions about course material (PDFs and Jupyter notebooks).  
The system retrieves relevant document chunks and uses a local LLM to generate answers.

Core pipeline:

User Question
→ Embedding search
→ Vector retrieval
→ Reranking
→ Context compression
→ LLM answer generation

Models used:

- LLM: llama3.1 (Ollama)
- Embeddings: nomic-embed-text (Ollama)
- Reranker: bge-reranker-base
- Vector database: FAISS

All models run locally.

---

## Project Architecture

Repository structure:

backend/
rag/
pdf_processor.py
notebook_processor.py
retriever.py
reranker.py
compressor.py
generator.py

scripts/
ingest_data.py

data/
raw/
processed/

models/

frontend/

Guidelines:

- RAG logic lives inside `backend/rag`
- Scripts are used only for ingestion or setup
- Retrieval pipeline should be modular

---

## RAG Pipeline Rules

Always follow this pipeline design:

1. Query embedding
2. Vector search (ChromaDB)
3. Retrieve top K chunks
4. Rerank chunks
5. Compress context
6. Send context to LLM

Important rules:

- Retrieval must work without the LLM
- The LLM should only perform reasoning or answer generation
- Avoid using the LLM for search decisions

---

## Document Processing

Documents come from:

- PDFs
- Jupyter notebooks

Processing steps:

1. Extract text
2. Split into chunks
3. Store metadata

Metadata must include:

- course
- section
- source file
- chunk id

This allows source attribution.

---

## Code Design Principles

Follow these rules when generating code:

- modular architecture
- small focused functions
- clear separation of responsibilities
- avoid monolithic scripts

Example modules:

retriever.py → retrieval logic  
reranker.py → reranking logic  
compressor.py → context compression  
generator.py → LLM interaction

---

## Performance Guidelines

Prioritize efficiency:

- Avoid unnecessary LLM calls
- Prefer embeddings for retrieval
- Rerank only top 20 results
- Send maximum 5 chunks to the LLM

Target pipeline:

query
→ embedding search (20 chunks)
→ rerank (top 5)
→ compress context
→ LLM answer

---

## Coding Style

Language: Python

Follow these rules:

- type hints for functions
- clear docstrings
- meaningful variable names
- avoid deeply nested logic
- separate data logic from model logic
- When implementing code check the documentation:
  - for chromadb :https://docs.trychroma.com/docs
  - for langchain :https://langchain-doc.readthedocs.io/en/latest/index.html
  - for ollama : https://docs.ollama.com/api/introduction
- don't reimplement functionality that is available in libraries (e.g. don't write your own vector search if chromadb provides it)
- write unit tests for critical functions (e.g. retrieval, reranking)
- for Data processing, use existing libraries (e.g. PyPDF2 for PDFs, nbformat for notebooks)
- For the model use this one:
  - the LLM: llama3.1 (Ollama)
  - the embedding model: nomic-embed-text (Ollama)
  - the reranker: bge-reranker-base
- the vector database: ChromaDB
- for the LLM, use a prompt template that includes instructions for how to use the retrieved context to answer the question, and include the source attribution information in the prompt so that the LLM can include it in the answer.
- For the chunking, use a sliding window approach with a chunk size of 500 tokens and an overlap of 100 tokens to ensure that important context is not lost between chunks.
- Don t break between pages when chunking PDFs, as this can lead to loss of context. Instead, allow chunks to span across page boundaries if necessary.
- For the retrieval, use cosine similarity on the embeddings to find the most relevant chunks for a given query. You can use the built-in functionality of ChromaDB for this.

Example:

```python
def retrieve_chunks(query: str, k: int = 20) -> list[Chunk]:
    """Retrieve candidate chunks from the vector store."""
