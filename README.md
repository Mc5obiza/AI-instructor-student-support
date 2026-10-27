# AI LilMose3da
### OpenRouter-Powered Agentic RAG Tutor for Data Science: LangGraph-Orchestrated, Retrieval-First, with Local Embeddings and Reranking

## 1. Project Title & Catchy Subtitle

**AI LilMose3da** is a local-first AI learning assistant engineered for Data Science education.
It combines a retrieval-augmented generation (RAG) pipeline with tool-based orchestration to route user intent between:

- High-precision course question answering
- Code explanation and debugging support

Chat generation uses OpenRouter; embeddings and reranking run locally to keep retrieval independent of the hosted LLM.

## 2. Overview & Impact

AI LilMose3da was built as a production-style academic assistant that can handle both conceptual learning and practical coding support in a single conversational system. The backend uses FastAPI for streaming inference, LangGraph for deterministic tool routing, ChromaDB for semantic retrieval, and a MariaDB-backed session layer for persistent conversational state.

The retrieval subsystem is intentionally multi-stage and hierarchical:

- Coarse course-level narrowing
- Section-level narrowing
- Dense chunk-level semantic retrieval
- Lightweight BM25 ranking over chunk text
- Reciprocal Rank Fusion (RRF) to merge semantic and BM25 signals
- Cross-encoder reranking with bge-reranker-base
- Context compression before final generation

This architecture improved student checkpoint outcomes from a baseline around **60%** to a consistent **87-92%** after deployment in local learning workflows.

> **Measured Outcome:** Structured AI-assisted study sessions powered by grounded retrieval increased checkpoint performance from **60%** to **87-92%**.

## 3. Architecture

### System Flow

The platform is built around a LangGraph router node that dispatches each query to a specialized execution path:

- **RAG path** for grounded course-content QA
- **Code reviewer path** for code explanation, debugging, and follow-up code edits

For RAG, the system performs hierarchical retrieval over three Chroma collections (`course`, `part`, `chunk`), gathers top semantic candidates (top-25 design target), adds a lightweight internal BM25 ranking step, fuses semantic and BM25 rankings via RRF, reranks to top-5 with `bge-reranker-base`, compresses context when needed, and then prompts the configured OpenRouter model for final grounded response generation.

FastAPI streams token outputs over SSE, while user accounts, sessions, titles, message history, and memory summaries are persisted through the backend session management layer backed by MariaDB.

```mermaid
flowchart TD
    U[Student Query] --> API[FastAPI /ask/stream SSE]
    API --> AUTH[JWT + Session Resolution]
    AUTH --> ORCH[LangGraph Orchestrator]

    ORCH --> ROUTER{Route Decision}

    ROUTER -->|code_reviewer| CR[Code Reviewer Toolchain]
    CR --> CRMEM[Load Code-Review Context]
    CRMEM --> CRLLM[Configured OpenRouter Model]
    CRLLM --> STREAM1[Token Stream Response]

    ROUTER -->|rag| GUARD[Scope + Clarity Guard]
    GUARD --> MEM[Conversation Summary Memory]
    MEM --> HIER[Hierarchical Retrieval]

    HIER --> C1[Course Collection Search]
    C1 --> C2[Section/Part Collection Search]
    C2 --> C3[Chunk Collection Semantic Search\nTop-K Candidate Chunks]
    C3 --> RR[bge-reranker-base\nTop-5 Relevance Selection]
    RR --> CMP[Context Compression / Budgeting]
    CMP --> GEN[Grounded Answer Generation\nOpenRouter]
    GEN --> STREAM2[Token Stream Response]

    STREAM1 --> STORE[Persist Session + Messages + Summary]
    STREAM2 --> STORE

    STORE --> DB[(MariaDB: users, sessions, messages)]
    HIER --> VDB[(ChromaDB Persistent Collections)]
```

### Data and Control Planes

- **Control plane:** LangGraph state machine handles deterministic routing and node transitions.
- **Retrieval plane:** ChromaDB semantic search + cross-encoder reranking.
- **Generation plane:** OpenRouter by default (or Ollama when explicitly selected), constrained by retrieved context and guardrails.
- **Persistence plane:** MariaDB stores user/session/message continuity and session-level memory summaries.

## 4. Key Features

- **Agentic Orchestration with LangGraph:** Query-level dynamic routing between RAG and code-review execution paths.
- **Hierarchical Semantic Retrieval:** Course -> section -> chunk narrowing to reduce retrieval noise and improve grounding quality.
- **Multi-Stage RAG Pipeline:** Candidate retrieval + cross-encoder reranking + context compression before generation.
- **Grounded Response Policy:** Generation is explicitly constrained to retrieved context with out-of-scope and ambiguity safeguards.
- **Persistent Conversational Memory:** Summary-based memory supports follow-up continuity across sessions.
- **Structured Session Infrastructure:** User auth, session lifecycle, title management, and message history persisted in backend storage.
- **Real-Time Streaming UX:** SSE token streaming from FastAPI for responsive chat interactions.
- **Flexible Model Providers:** Use OpenRouter by default or select local Ollama with `LLM_PROVIDER=ollama`.
- **Code Assistance Path:** Dedicated tooling for explaining, debugging, and iterating over code-centric student prompts.

## 5. Tech Stack

| Category | Technologies |
|---|---|
| **LLMs** | OpenRouter model (default) via the OpenAI-compatible LangChain client, or local Ollama |
| **Embedding Models** | `nomic-embed-text` via Ollama Embeddings API |
| **Reranking** | `BAAI/bge-reranker-base` (Sentence Transformers CrossEncoder) |
| **Orchestration** | LangGraph, LangChain Core |
| **RAG Retrieval** | ChromaDB (`PersistentClient`) with hierarchical collections (`courses`, `parts`, `chunks`) |
| **Backend API** | FastAPI, Pydantic, SSE streaming (`StreamingResponse`) |
| **Session/Data Persistence** | MariaDB-compatible relational schema accessed with PyMySQL |
| **Authentication** | JWT (`python-jose`), secure HTTP-only cookies |
| **Frontend** | React + Vite, Axios, Markdown/KaTeX rendering |
| **Infrastructure** | Dockerized local deployment, local model serving with Ollama |

---

This project is designed to showcase production-minded AI engineering: retrieval-grounded generation, explicit orchestration, memory-aware dialogue, and end-to-end persistence under a local-first deployment model.

## Testing

The default test suite uses deterministic in-memory vector collections and mocked model/API boundaries. It does not connect to MariaDB, call Ollama, download models, or read the persisted production vector database.
GitHub Actions runs this suite on Windows for pushes and pull requests targeting `main`; no live database is required.

OpenRouter is the default chat provider. Set `OPENROUTER_API_KEY` to your OpenRouter API key in the root `.env` file, and optionally set `OPENROUTER_MODEL` (default: `openai/gpt-4o-mini`). You can explicitly select local chat generation with `LLM_PROVIDER=ollama`. Ollama is still used for local embeddings in either configuration.

Activate the project environment from the repository root before installing test tools or running Python:

```powershell
conda activate agents_env
```

Install the test dependencies when setting up the environment:

```powershell
python -m pip install -r requirements-test.txt
```

Run the complete suite or one category:

```powershell
pytest
pytest -m unit
pytest -m integration
pytest -m feature
pytest -m e2e
```

Generate a coverage report for the retrieval, document-processing, orchestration, API, and auth modules:

```powershell
pytest --cov=backend.rag.retriever --cov=backend.rag.pdf_processor --cov=backend.rag.notebook_processor --cov=backend.rag.orchestrator --cov=backend.helper --cov=backend.main --cov-report=term-missing --cov-report=html
```

The optional HTML report is written to `htmlcov/`. Real-model and live-database tests are intentionally not part of the default suite; add explicit infrastructure configuration before running those checks.

## 6. Docker Deployment

You can run the full stack (frontend + backend + MariaDB + Ollama for embeddings) with Docker Compose. Chat generation requires an OpenRouter API key in `.env`.

### Configure secrets

Create a local env file from the template and set your real values:

```bash
cp .env.example .env
```

The `.env` file is ignored by git, so secrets stay out of source control.

### Start services

```bash
docker compose up --build -d
```

### Pull the Ollama embedding model (first run only)

```bash
docker compose exec ollama ollama pull nomic-embed-text
```

### Access apps

- Frontend: http://localhost:5173
- Backend API health: http://localhost:8000/health
- Ollama API: http://localhost:11434

### Stop services

```bash
docker compose down
```

### Troubleshooting authentication

Registration requires a username of at least 2 characters, a valid email, and a password of at least 8 characters.

If login returns a database access-denied error for `ai_user`, set `DB_PASSWORD` in the root `.env` to the password configured for that database account. With Docker Compose, MariaDB only creates the account from these environment values when its data volume is first initialized; changing `.env` later does not update the stored account password. Update the MariaDB account password to match `.env` or restore the original value, then restart the backend. Do not remove the database volume unless you intend to erase its data.
