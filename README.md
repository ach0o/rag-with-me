[한국어](README.ko.md)

# RAG Agent on LangGraph

A modular, production-grade RAG (Retrieval-Augmented Generation) agent built with Python. Every component — chunkers, vector stores, embedders, LLMs, retrievers, and rerankers — is behind an abstraction and swappable via YAML configuration.

The primary data source is the **97 Things Every Programmer Should Know** collection of markdown files.

## Architecture

**Hexagonal architecture (ports & adapters)** with a **pipeline pattern** for ingestion.

```
src/rag_agent/
├── domain/           # Pure business models, ports (Protocol classes). Zero external deps.
│   ├── models.py     # Document, Chunk, QueryResult
│   ├── pipeline.py   # Pipeline, PipelineStage
│   └── ports.py      # All port definitions
├── application/      # Use case orchestration. Imports domain/ only.
│   ├── ingest.py     # IngestUseCase + pipeline stages
│   └── query.py      # QueryUseCase
├── adapters/
│   ├── inbound/      # CLI argument parsing
│   └── outbound/     # Concrete implementations of ports
│       ├── chunkers/       # fixed-size, markdown-header, semantic
│       ├── doc_loaders/    # markdown
│       ├── embedders/      # Azure OpenAI
│       ├── llms/           # Azure OpenAI
│       ├── vector_stores/  # ChromaDB
│       ├── retrievers/     # dense, BM25 sparse, hybrid (RRF)
│       ├── rerankers/      # none, Cohere, cross-encoder
│       ├── document_repos/ # PostgreSQL
│       └── chunk_repos/    # PostgreSQL
├── config.py         # Pydantic models, YAML loader
└── main.py           # Composition root (wires adapters → use cases)
```

### Dependency Rule

- `domain/` imports nothing external — pure Python only
- `application/` imports from `domain/` only
- `adapters/` imports from `domain/` + external libraries
- `main.py` is the only place that imports from `adapters/`

## Swappable Components

| Slot | Available Adapters | Config key |
|------|-------------------|------------|
| LLM | Azure OpenAI | `llm.provider` |
| Embedder | Azure OpenAI | `embedder.provider` |
| Vector store | ChromaDB | `vector_store.provider` |
| Chunker | fixed-size, markdown-header, semantic | `chunker.strategy` |
| Retriever | dense, bm25_sparse, hybrid | `retriever.provider` |
| Reranker | none, cohere, cross_encoder | `reranker.provider` |
| Document repo | PostgreSQL | `database.enabled` |
| Chunk repo | PostgreSQL | `database.enabled` |

All swapping is done in `config/default.yaml` — no code changes needed.

## Data Flow

### Ingestion

```
CLI → DocLoader.load() → [PersistDocuments] → Chunker.chunk()
    → [PersistChunks] → Embedder.embed() → VectorStore.add()
```

Persistence stages are optional — enabled when `database.enabled: true`.

### Query

```
CLI → Retriever.retrieve(query) → [Reranker.rerank()] → build prompt
    → LLMProvider.generate(prompt) → QueryResult
```

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Docker (for PostgreSQL)

### Install

```bash
uv sync
```

### Environment

Copy `.env-example` to `.env` and fill in your keys:

```bash
cp .env-example .env
```

Required:
- `AZURE_OPENAI_API_KEY` — Azure OpenAI API key
- `AZURE_OPENAI_EMBEDDING_ENDPOINT` — Azure OpenAI embedding endpoint

Optional (for PostgreSQL persistence):
- `POSTGRES_PASSWORD` — PostgreSQL password for Docker

Optional (for Cohere reranker):
- `COHERE_API_KEY` — Cohere API key
- `COHERE_ENDPOINT` — Cohere rerank endpoint

### Database (optional)

```bash
docker compose up -d
```

Set `database.enabled: true` in `config/default.yaml`.

### Data Source

Clone the 97 Things repo into `data/`:

```bash
git clone https://github.com/97-things/97-things-every-programmer-should-know data/97-things
```

## Usage

### Ingest documents

```bash
python -m rag_agent ingest
```

### Query

```bash
python -m rag_agent query "What should every programmer know about code reviews?"
```

### Custom config

```bash
python -m rag_agent --config config/custom.yaml query "your question"
```

## Configuration

All config lives in `config/default.yaml`. Example:

```yaml
llm:
  provider: azure-openai
  model: gpt-5-mini
  temperature: 0.0
  max_tokens: 1024

chunker:
  strategy: semantic        # fixed-size | markdown-header | semantic
  threshold: 0.5
  min_chunk_size: 100

retriever:
  provider: hybrid          # dense | bm25_sparse | hybrid
  top_k: 5

reranker:
  provider: cross_encoder   # none | cohere | cross_encoder
  top_k: 3

database:
  url: "postgresql://rag:localdev@localhost:5432/rag_agent"
  enabled: true
```

Pydantic validates all config at startup — typos and invalid values are caught before any work begins.

## Tech Stack

| Layer | Choice |
|-------|--------|
| Language | Python 3.12 |
| Package manager | uv |
| Config | YAML + Pydantic |
| Vector store | ChromaDB |
| Database | PostgreSQL 16 |
| LLM / Embedder | Azure OpenAI |
| Infrastructure | Docker Compose |

## Roadmap

- [x] Phase 1 — Foundation (end-to-end RAG pipeline)
- [x] Phase 2 — Multiple adapters, PostgreSQL storage, rerankers
- [ ] Phase 3 — LangGraph agent (conditional re-query)
- [ ] Phase 4 — PDF/DOCX loaders, query expansion
- [ ] Phase 5 — Evaluation & observability
- [ ] Phase 6 — Portfolio polish (demo UI, CI)
