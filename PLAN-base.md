# RAG Agent on LangGraph — Project Plan

## Overview

A modular, production-grade RAG (Retrieval-Augmented Generation) agent built with LangGraph. The primary data source is the **97 Things Every Programmer Should Know** GitHub project (collection of markdown files), with architecture designed to extend to PDFs, DOCX, and other formats.

**Key design principle:** Every component is behind an abstraction. Chunkers, vector stores, embedders, LLMs, retrievers, and rerankers can all be swapped via configuration — enabling A/B comparison and iterative improvement.

---

## Tech Stack (Fixed)

These are confirmed choices that don't change:

| Layer | Choice | Reason |
|-------|--------|--------|
| Language | Python 3.12+ | LangGraph ecosystem |
| Package manager | uv | Fast, lockfile-based |
| Agent framework | LangGraph | Graph-based agent with explicit state |
| Infrastructure | Docker Compose | Local services (vector stores, etc.) |
| Config | YAML + Pydantic | Validation at startup |
| Testing | pytest | Standard |

## Tech Stack (Swappable — Decided at Implementation Time)

These are the *slots* the architecture supports. Pick one per slot for Phase 1, add more in Phase 2+.

| Slot | Options | Phase 1 pick |
|------|---------|-------------|
| LLM provider | Anthropic (Claude), OpenAI (GPT), Ollama (local) | TBD |
| Embedder | OpenAI embeddings, sentence-transformers (local) | TBD |
| Vector store | ChromaDB, FAISS, Qdrant | TBD |
| Chunker | Fixed-size, semantic, markdown-header-aware | Fixed-size (simplest baseline) |
| Retriever | Dense, sparse (BM25), hybrid | Dense (simplest baseline) |
| Reranker | None, Cohere, cross-encoder | None (add later) |
| CLI framework | typer, click, argparse | TBD |
| Domain models | dataclass, Pydantic BaseModel, NamedTuple | TBD |

---

## Architecture

**Hexagonal architecture (ports & adapters)** for the overall system. **Pipeline pattern** for the ingestion flow.

### Four Layers

**`core/domain/`** — Pure business models and rules. Zero external imports. Contains:
- Domain models: `Document`, `Chunk`, `QueryResult`
- Pipeline infrastructure: `Pipeline` class, `PipelineStage` protocol

**`core/use_cases/`** — Application-level orchestration. Each file is one use case. Imports from `core/domain/` and `ports/` only. Contains:
- `IngestUseCase` — assembles and runs the ingestion pipeline
- `QueryUseCase` — retrieves context, builds prompt, calls LLM
- `EvaluateUseCase` — runs metrics across configs (Phase 5)

**`ports/`** — Protocol definitions. The contracts adapters must satisfy. Imports from `core/domain/` for type hints only.

**`adapters/`** — Concrete implementations of ports. Imports from `ports/` and external libraries.

**`main.py`** — CLI entry point AND composition root. Reads YAML config, does simple if/elif to pick the right adapter for each port, instantiates, wires into use cases. This is the ONLY place that imports from `adapters/`.

### Dependency Rule

```
adapters/ → ports/ ← core/use_cases/ → core/domain/
              ↑
          main.py (reads config, picks adapters, calls use_cases)
```

- `core/domain/` imports NOTHING external. Pure Python standard library only.
- `core/use_cases/` imports from `core/domain/` and `ports/`. Never from `adapters/`.
- `ports/` imports from `core/domain/` for type hints. Nothing else.
- `adapters/` imports from `ports/` and external libraries.
- `main.py` is the only place that imports from `adapters/`.

If you see `import chromadb` inside `core/`, that's a bug.

### Pipeline Pattern (Ingestion Only)

The ingestion flow is a composable pipeline: `load → chunk → embed → store`.

Both `Pipeline` and `PipelineStage` live in `core/domain/` — they are domain concepts, NOT ports. `PipelineStage` doesn't represent an external system boundary; it's an internal composition mechanism. Ports are strictly for external boundaries (LLM APIs, vector stores, file systems).

The `IngestUseCase` defines stage classes that wrap ports into pipeline stages, assembles the pipeline, and runs it. Stages can be added, removed, or reordered.

The query flow is NOT a pipeline — it has conditional branching (re-query if context is bad), which is why LangGraph handles it in Phase 3.

### Design Decisions & Known Tradeoffs

1. **`PipelineStage.process()` uses `Any → Any`.** Ideally generic (`PipelineStage[In, Out]`) but composing generic stages in Python's type system is hard. Accepted tradeoff: each stage class is individually typed in its implementation, but `Pipeline.run()` doesn't enforce stage-to-stage type compatibility. Catch mismatches with integration tests.

2. **`Retriever` is a composite port.** `DenseRetriever` adapter internally uses `Embedder` + `VectorStore`. Its constructor receives both, wired by `main.py`. The CLI is the only place that knows about cross-adapter dependencies.

3. **Config params like `top_k` live in config, not port signatures.** Adapters read params from their constructor (set by `main.py` from YAML config). Port methods have no optional params — one source of truth.

4. **Pipeline stages create new objects, not mutate.** EmbedStage returns new `Chunk` instances with embeddings, rather than mutating input. Keeps stages pure.

---

## Project Structure

```
rag-agent/
├── pyproject.toml
├── uv.lock
├── docker-compose.yml
├── .env                        # API keys (gitignored)
├── config/
│   └── default.yaml            # component selection + params
│
├── src/
│   └── rag_agent/
│       ├── __init__.py
│       ├── main.py              # CLI entry point + composition root
│       ├── config.py            # Pydantic models, YAML loader
│       │
│       ├── core/
│       │   ├── domain/          # ★ DOMAIN (pure, zero deps)
│       │   │   ├── __init__.py
│       │   │   ├── models.py    # Document, Chunk, QueryResult
│       │   │   └── pipeline.py  # Pipeline, PipelineStage
│       │   │
│       │   └── use_cases/       # ★ USE CASES (orchestration)
│       │       ├── __init__.py
│       │       ├── ingest.py    # IngestUseCase + stage classes
│       │       ├── query.py     # QueryUseCase
│       │       └── evaluate.py  # EvaluateUseCase (Phase 5)
│       │
│       ├── ports/               # ★ PORTS (Protocol definitions)
│       │   ├── __init__.py
│       │   ├── llm.py
│       │   ├── embedder.py
│       │   ├── vector_store.py
│       │   ├── chunker.py
│       │   ├── doc_loader.py
│       │   ├── retriever.py
│       │   └── reranker.py
│       │
│       └── adapters/            # ★ ADAPTERS (implementations)
│           ├── __init__.py
│           ├── llms/            # one file per provider
│           ├── embedders/       # one file per provider
│           ├── vector_stores/   # one file per provider
│           ├── chunkers/        # one file per strategy
│           ├── doc_loaders/     # one file per format
│           ├── retrievers/      # one file per strategy
│           └── rerankers/       # one file per provider
│
├── data/
│   └── 97-things/               # Cloned source repo (gitignored)
├── eval_data/
│   └── gold_qa.json             # Gold-standard Q&A pairs
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fakes/                   # In-memory test doubles
└── README.md
```

---

## Detailed Specs

### Domain Models (`core/domain/models.py`)

Three domain models. All are plain data containers with no behavior and no external imports.

**`Document`**
- `content: str` — the raw text content of a loaded file
- `metadata: dict` — flexible metadata bag. Expected keys: `source` (file path), `title` (extracted from filename or front-matter). Extensible for future formats.

**`Chunk`**
- `content: str` — the text content of this chunk
- `metadata: dict` — inherited from parent `Document` plus chunk-specific info. Expected keys: `source`, `title`, `chunk_index` (position in the original document), `start_char` / `end_char` (character offsets). For markdown-header chunking: `headers` (list of parent headers).
- `embedding: list[float] | None` — populated by the embed stage, `None` before that

**`QueryResult`**
- `answer: str` — the LLM's generated answer
- `chunks: list[Chunk]` — the retrieved context chunks used to generate the answer
- `metadata: dict` — extensible. Expected keys: `retrieval_time_ms`, `generation_time_ms`, `model` (which LLM answered)

### Pipeline Infrastructure (`core/domain/pipeline.py`)

**`PipelineStage` (Protocol)**
- Single method: `process(self, data: Any) → Any`
- Defined HERE in domain, not in `ports/`. This is a domain concept for composing stages, not an external system boundary.
- Uses `Any → Any` (see tradeoff #1 above). Each concrete stage class types its own `process()` more specifically.

**`Pipeline`**
- Constructor: takes optional `list[PipelineStage]`
- `add_stage(self, stage: PipelineStage) → Pipeline` — appends a stage, returns self for fluent chaining
- `run(self, data: Any) → Any` — executes stages in order, passing each stage's output as the next stage's input. First stage receives the initial `data` argument.

### Ports (`ports/`)

All ports are Python `Protocol` classes. Adapters satisfy them via structural subtyping — no inheritance required.

**`DocLoader`** (`ports/doc_loader.py`)
- `load(self) → list[Document]`
- Loads all documents from the configured data source. Returns domain `Document` objects.

**`Chunker`** (`ports/chunker.py`)
- `chunk(self, document: Document) → list[Chunk]`
- Splits a single document into chunks. Preserves and extends metadata from the parent document.

**`Embedder`** (`ports/embedder.py`)
- `embed(self, texts: list[str]) → list[list[float]]`
- Takes a batch of text strings, returns a list of embedding vectors (same order, same length).

**`VectorStore`** (`ports/vector_store.py`)
- `add(self, chunks: list[Chunk]) → None` — stores chunks with their embeddings and metadata
- `search(self, embedding: list[float], top_k: int) → list[Chunk]` — finds the top_k most similar chunks to the given embedding vector

**`Retriever`** (`ports/retriever.py`)
- `retrieve(self, query: str) → list[Chunk]`
- Takes a natural language query string, returns relevant chunks. No `top_k` parameter — that's in the adapter's constructor via config.
- NOTE: `DenseRetriever` is a composite adapter. It receives `Embedder` + `VectorStore` in its constructor (wired by `main.py`), embeds the query, then searches. Other retriever strategies (BM25, hybrid) may have different constructor dependencies.

**`Reranker`** (`ports/reranker.py`)
- `rerank(self, query: str, chunks: list[Chunk]) → list[Chunk]`
- Takes the query and a list of retrieved chunks, returns a reordered (and possibly truncated) list. The "none" adapter returns chunks unchanged.

**`LLMProvider`** (`ports/llm.py`)
- `generate(self, prompt: str) → str`
- Takes a prompt string, returns the generated text. Model selection, temperature, max_tokens are in the adapter's constructor via config.

### Use Cases (`core/use_cases/`)

Each use case is a class with an `execute()` method. Constructor receives ports via dependency injection. Use cases never instantiate adapters — they receive already-wired ports.

**`IngestUseCase`** (`core/use_cases/ingest.py`)

This file contains both the use case class and the pipeline stage classes.

Stage classes (each wraps one port into a `PipelineStage`):
- `LoadStage` — constructor takes `DocLoader`. `process(None) → list[Document]`.
- `ChunkStage` — constructor takes `Chunker`. `process(list[Document]) → list[Chunk]`. Iterates documents, flattens all chunks into one list.
- `EmbedStage` — constructor takes `Embedder`. `process(list[Chunk]) → list[Chunk]`. Calls `embed()` on all chunk contents in batch, returns NEW `Chunk` objects with `embedding` field populated. Does NOT mutate input chunks.
- `StoreStage` — constructor takes `VectorStore`. `process(list[Chunk]) → list[Chunk]`. Calls `add()` to persist, returns the same chunks (passthrough for downstream logging/counting).

`IngestUseCase` class:
- Constructor takes `DocLoader`, `Chunker`, `Embedder`, `VectorStore`
- Internally builds a `Pipeline` with the four stages above in order
- `execute(self) → list[Chunk]` — runs the pipeline with `None` as initial input, returns the stored chunks

**`QueryUseCase`** (`core/use_cases/query.py`)
- Constructor takes `Retriever`, `LLMProvider`
- `execute(self, question: str) → QueryResult`
- Flow: calls `retriever.retrieve(question)` → assembles a prompt with the retrieved chunks as context → calls `llm.generate(prompt)` → returns a `QueryResult` containing the answer and the chunks used
- Prompt template: instructs the LLM to answer based on the provided context and say so if the context doesn't contain the answer

**`EvaluateUseCase`** (`core/use_cases/evaluate.py`) — Phase 5, not implemented in Phase 1.

### CLI (`main.py`)

- Two commands: `ingest` and `query`
- Both accept `--config` flag (defaults to `config/default.yaml`)
- `main.py` is also the composition root — it reads config, picks the right adapter for each port via simple if/elif on the config values, instantiates them, and wires them into use cases
- `ingest`: loads config → builds adapters → creates `IngestUseCase` → calls `execute()` → prints summary (document count, chunk count)
- `query`: loads config → builds adapters → creates `QueryUseCase` → calls `execute(question)` → prints answer and source citations
- Adding a new adapter = one new file in `adapters/` + one new elif branch in `main.py`

### Config (`config.py`)

- One Pydantic `BaseModel` per YAML section: `LLMConfig`, `EmbedderConfig`, `VectorStoreConfig`, `ChunkerConfig`, `RetrieverConfig`, `RerankerConfig`, `DataSourceConfig`
- Top-level `AppConfig` composes all of them, each with sensible defaults
- `Literal` types on provider/strategy fields (e.g., `Literal["anthropic", "openai", "ollama"]`) catch typos at startup
- `Field` constraints: `chunk_size` must be `gt=0`, `temperature` must be `ge=0.0, le=2.0`, etc.
- Cross-field `model_validator`: `chunk_overlap` must be less than `chunk_size`
- `AppConfig.from_yaml(path: str | Path) → AppConfig` class method: loads YAML with `yaml.safe_load`, passes to constructor, Pydantic validates
- Missing sections use defaults — you can have a YAML with just `llm.provider: openai` and everything else defaults

---

## Data Flow

### Ingestion Flow

```
CLI (main.py ingest)
  → AppConfig.from_yaml("config/default.yaml")    # validates config
  → main.py reads config, picks adapters:          # if/elif on config values
      → instantiates DocLoader adapter
      → instantiates Chunker adapter
      → instantiates Embedder adapter
      → instantiates VectorStore adapter
  → IngestUseCase(loader, chunker, embedder, store)
  → IngestUseCase.execute()
      → Pipeline.run(None)
          → LoadStage.process(None)
              → DocLoader.load()
              → returns list[Document]             # e.g., 97 Document objects
          → ChunkStage.process(list[Document])
              → Chunker.chunk(doc) for each doc
              → returns list[Chunk]                # e.g., 500 Chunk objects
          → EmbedStage.process(list[Chunk])
              → Embedder.embed([c.content for c in chunks])
              → returns NEW list[Chunk] with embeddings populated
          → StoreStage.process(list[Chunk])
              → VectorStore.add(chunks)            # persists to store
              → returns list[Chunk]                # passthrough
      → returns list[Chunk]
  → prints summary
```

### Query Flow

```
CLI (main.py query "What about code reviews?")
  → AppConfig.from_yaml("config/default.yaml")
  → main.py reads config, picks adapters:
      → instantiates Embedder + VectorStore
      → instantiates DenseRetriever(embedder, store, top_k)
      → instantiates LLMProvider adapter
  → QueryUseCase(retriever, llm)
  → QueryUseCase.execute("What about code reviews?")
      → Retriever.retrieve("What about code reviews?")
          → (inside DenseRetriever) Embedder.embed(["What about code reviews?"])
          → (inside DenseRetriever) VectorStore.search(embedding, top_k=5)
          → returns list[Chunk]                    # top 5 relevant chunks
      → assembles prompt: context from chunks + question
      → LLMProvider.generate(prompt)
      → returns QueryResult(answer=..., chunks=..., metadata=...)
  → prints answer + source citations
```

---

## Config System

### `config/default.yaml`

```yaml
llm:
  provider: TBD               # anthropic | openai | ollama
  model: TBD
  temperature: 0.0
  max_tokens: 1024

embedder:
  provider: TBD               # openai | sentence_transformer
  model: TBD

vector_store:
  provider: TBD               # chroma | faiss | qdrant
  collection_name: rag_97things
  persist_directory: ./data/vector_db

chunker:
  strategy: fixed_size         # fixed_size | semantic | markdown_header
  chunk_size: 500
  chunk_overlap: 50

retriever:
  strategy: dense              # dense | sparse_bm25 | hybrid
  top_k: 5

reranker:
  provider: none               # none | cohere | cross_encoder
  top_k: 3

data_source:
  path: ./data/97-things
  type: markdown               # markdown | pdf | docx
```

### Validation Examples

| Mistake in YAML | What happens |
|-----------------|-------------|
| `provider: antropic` (typo) | `ValidationError` at startup — caught before any work begins |
| `chunk_size: -100` | `ValidationError: must be greater than 0` |
| `chunk_overlap: 600` with `chunk_size: 500` | `ValidationError: chunk_overlap must be < chunk_size` |
| `temperature: "hot"` | `ValidationError: must be a number` |
| Missing `llm` section entirely | Uses defaults |

### Secrets

API keys in `.env` (gitignored), never in YAML. Adapters read `os.environ` directly. YAML selects which provider; `.env` provides credentials.

---

## Docker Compose

Docker from day 1. Contents depend on which vector store you pick — if you choose one that needs a server (e.g., Qdrant), add it here. If you pick an in-process store (e.g., ChromaDB, FAISS), Docker Compose may only hold future services.

---

## Phase 1 — Foundation (Get It Working End-to-End)

**Goal:** A working RAG pipeline — question in, grounded answer out. ONE implementation per slot. No LangGraph yet (linear chain). This becomes the baseline everything else improves on.

### Phase 1 Deliverables

1. Project scaffolding (uv, Docker, config)
2. Domain models + Pipeline infrastructure
3. All 7 ports defined
4. One adapter per port (your choice of provider)
5. `main.py` wiring config → adapters → use cases
6. IngestUseCase working end-to-end
7. QueryUseCase working end-to-end
8. CLI: `python -m rag_agent ingest` and `python -m rag_agent query "..."`
9. Manual testing with 5-10 questions, baseline notes saved

### Step-by-Step

#### Step 1.1 — Scaffolding
- `uv init`, `pyproject.toml`, directory structure
- `docker-compose.yml` (contents depend on your vector store choice)
- `config/default.yaml` (fill in TBD values when you decide)
- `.env` for API keys
- **You learn:** uv workflow, hexagonal layout, Docker Compose, config-driven architecture

#### Step 1.2 — Domain Models & Pipeline
- `core/domain/models.py`: `Document`, `Chunk`, `QueryResult` (see specs above)
- `core/domain/pipeline.py`: `PipelineStage` protocol, `Pipeline` class (see specs above)
- **You learn:** pure domain layer with zero deps, Protocol basics

#### Step 1.3 — Ports
- All 7 port files in `ports/` (see specs above for exact method signatures)
- Each is a `Protocol` class referencing domain models
- **You learn:** designing minimal interfaces, structural subtyping

#### Step 1.4 — Adapters
- One adapter per port in `adapters/` — whichever provider you choose
- Each satisfies its port's Protocol via structural subtyping
- `DenseRetriever` receives `Embedder` + `VectorStore` in constructor
- **You learn:** wrapping SDKs behind ports, composite adapters

#### Step 1.5 — Use Cases
- `IngestUseCase` with stage classes + pipeline assembly (see specs above)
- `QueryUseCase` with retrieve → prompt → generate flow (see specs above)
- **You learn:** use case pattern, pipeline pattern, dependency injection

#### Step 1.6 — CLI & Wiring
- Wire `main.py` as both CLI and composition root
- Reads config, if/elif to pick adapters, instantiates, passes to use cases
- Two commands: `ingest` and `query`
- Run 5-10 manual test questions, save baseline notes
- **You learn:** full end-to-end RAG loop, composition root pattern, where retrieval fails

---

## Phase 2-6 Summary

### Phase 2 — More Adapters & Testing
- Add 2nd adapter per port
- YAML config swap = different adapter wiring
- Unit tests with fakes in `tests/fakes/`, integration tests with real adapters

### Phase 3 — LangGraph Agent
- Convert `QueryUseCase` to LangGraph graph with explicit state
- Nodes: `query_analyzer → retriever → grader → generator`
- Conditional edges: poor context → re-query with rephrased question

### Phase 4 — Advanced RAG
- Hybrid retrieval (dense + BM25 with RRF fusion)
- Reranking layer
- Markdown-header-aware chunking
- PDF and DOCX loaders

### Phase 5 — Evaluation & Observability
- Gold-standard Q&A dataset (20-30 pairs)
- Retrieval recall@k, answer faithfulness metrics
- Comparison runner: config A vs config B

### Phase 6 — Portfolio Polish
- README with architecture diagrams
- Demo UI (Streamlit/Gradio)
- Docker Compose for full stack
- CI pipeline

---

## Learning Roadmap

| Phase | Concepts |
|-------|---------|
| 1 | RAG fundamentals, hexagonal architecture, pipeline pattern, Python protocols, composition root, embedding + vector store basics |
| 2 | Adding adapters, config-driven swapping, structural subtyping, unit testing with fakes |
| 3 | LangGraph state machines, conditional routing, agent loops, conversation memory |
| 4 | Hybrid search, reciprocal rank fusion, reranking, document parsing |
| 5 | RAG evaluation, metrics design, experiment tracking, observability |
| 6 | Documentation, demo UI, Docker networking, CI/CD |