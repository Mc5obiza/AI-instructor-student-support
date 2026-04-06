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
- always check copilot-instructions.md before implementing code to ensure that you are following the project guidelines and architecture.
- Don t create new environments or dependencies without checking with the team first, as we want to keep the project lightweight and avoid unnecessary complexity.
- Use rag_env conda envirenment for development, which includes all necessary dependencies for the project.
- always follow the architecture and design principles outlined in this document to ensure that the codebase remains maintainable and scalable as the project grows.
- Add safeguards to prevent the LLM from generating answers that are not supported by the retrieved context. For example, you can include instructions in the prompt to only answer questions that can be answered based on the retrieved chunks, and to indicate when the information is not sufficient to provide an answer.
- If the user question is ambiguous or lacks sufficient information, the system should ask clarifying questions to gather more details before attempting to generate an answer. This can help improve the accuracy and relevance of the responses generated by the LLM.
- If the user question is very broad, the system should attempt to narrow down the scope of the question by asking follow-up questions or providing options for the user to choose from. This can help guide the retrieval process and ensure that the most relevant information is retrieved for answering the question. 
- If the user prompt is not about datascience or course material, the system should respond with a polite message indicating that it is designed to assist with questions related to the course content, and suggest that the user ask a question related to the course material for better assistance.

Example:

```python
def retrieve_chunks(query: str, k: int = 20) -> list[Chunk]:
    """Retrieve candidate chunks from the vector store."""
