# RAG Agent — Detailed Implementation Specs

> Companion to `rag-agent-project-plan.md` (base plan). This document provides per-file specs and metadata contracts. Refer to the base plan for architecture, dependency rules, and data flow.

---

## Metadata Contract

Metadata flows through the entire pipeline as `dict` fields on `Document`, `Chunk`, and `QueryResult`. This section defines the exact keys at each stage — what's set, what's inherited, and what's expected downstream.

### Metadata Lifecycle

```
DocLoader sets metadata on Document
  → ChunkStage copies Document metadata to each Chunk, adds chunk-specific keys
    → EmbedStage passes metadata through unchanged (creates new Chunk, copies metadata)
      → StoreStage persists metadata alongside embedding in vector store
        → VectorStore.search returns Chunks with metadata intact
          → QueryUseCase reads metadata to format source citations
            → QueryResult.metadata adds query-level metrics
```

### `Document.metadata` (set by DocLoader)

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `source` | `str` | Absolute or relative file path of the loaded document | `"data/97-things/thing_01.md"` |
| `title` | `str` | Human-readable title. Extracted from markdown front-matter `title:` field, or derived from filename if no front-matter (strip extension, replace hyphens/underscores with spaces, title-case). | `"Act with Prudence"` |
| `format` | `str` | File format. Matches the `data_source.type` config value. | `"markdown"` |
| `file_size_bytes` | `int` | File size in bytes at load time. Useful for debugging and evaluation. | `2048` |
| `loaded_at` | `str` | ISO 8601 timestamp of when the document was loaded. | `"2026-03-14T10:30:00Z"` |

Future formats (PDF, DOCX) may add keys like `page_count`, `author`, `created_date`. The contract is that `source`, `title`, and `format` are always present.

### `Chunk.metadata` (set by ChunkStage)

Inherits ALL keys from parent `Document.metadata`, plus:

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `chunk_index` | `int` | Zero-based position of this chunk within its parent document. First chunk = 0. | `3` |
| `total_chunks` | `int` | Total number of chunks the parent document was split into. Useful for understanding coverage. | `7` |
| `start_char` | `int` | Character offset where this chunk starts in the original document content. | `1500` |
| `end_char` | `int` | Character offset where this chunk ends (exclusive). `content == original[start_char:end_char]` | `2000` |

Markdown-header-aware chunker (Phase 4) will add:

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `headers` | `list[str]` | Ordered list of parent markdown headers above this chunk. | `["## Testing", "### Unit Tests"]` |
| `header_level` | `int` | Deepest header level this chunk belongs to. | `3` |

The EmbedStage and StoreStage pass `Chunk.metadata` through unchanged. When StoreStage calls `VectorStore.add()`, the adapter must persist metadata alongside the embedding so that `VectorStore.search()` returns Chunks with the full metadata intact.

### `Chunk.embedding` lifecycle

| Stage | Value |
|-------|-------|
| After LoadStage | N/A (Documents, not Chunks) |
| After ChunkStage | `None` |
| After EmbedStage | `list[float]` — populated with embedding vector |
| After StoreStage | `list[float]` — same, persisted to store |
| After VectorStore.search() | `list[float]` or `None` — depends on whether the store returns embeddings on search. Not required; the embedding's job is done after storage. |

### `QueryResult.metadata` (set by QueryUseCase)

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `retrieval_time_ms` | `float` | Wall-clock time for the retrieve step in milliseconds. | `145.3` |
| `generation_time_ms` | `float` | Wall-clock time for the LLM generate step in milliseconds. | `892.1` |
| `total_time_ms` | `float` | Total wall-clock time for the entire query execution. | `1037.4` |
| `model` | `str` | The LLM model string used for generation (read from config). | `"claude-sonnet-4-20250514"` |
| `retriever` | `str` | The retriever type used (read from config). | `"dense"` |
| `top_k` | `int` | How many chunks were retrieved. | `5` |
| `chunks_returned` | `int` | Actual number of chunks returned (may be less than `top_k` if store has fewer). | `5` |

---

## Per-File Specs

Each file below is described with: its purpose, what it imports, what it exports, and the exact contents it should have. Files are listed in dependency order — you can implement top-to-bottom without forward references.

---

### `src/rag_agent/__init__.py`

**Purpose:** Package marker. Empty file or contains just the package version string.

**Imports:** None.

**Exports:** Optionally `__version__: str`.

---

### `src/rag_agent/core/__init__.py`

**Purpose:** Package marker for the core module. Empty.

---

### `src/rag_agent/core/domain/__init__.py`

**Purpose:** Package marker. Empty.

---

### `src/rag_agent/core/domain/models.py`

**Purpose:** All domain data models. Pure data containers. Zero external imports (standard library only).

**Imports:** Only Python standard library (`dataclasses` or nothing, depending on your model choice — see base plan TBD).

**Exports:** `Document`, `Chunk`, `QueryResult`

**`Document`:**
- Fields: `content: str`, `metadata: dict` (defaults to empty dict)
- No methods. No validation. Just data.

**`Chunk`:**
- Fields: `content: str`, `metadata: dict` (defaults to empty dict), `embedding: list[float] | None` (defaults to `None`)
- No methods. No validation.

**`QueryResult`:**
- Fields: `answer: str`, `chunks: list[Chunk]`, `metadata: dict` (defaults to empty dict)
- No methods.

**Key rule:** These models must NEVER import from `ports/`, `adapters/`, or any external library. They are the innermost layer.

---

### `src/rag_agent/core/domain/pipeline.py`

**Purpose:** The `Pipeline` class and `PipelineStage` protocol. Domain-level composition infrastructure.

**Imports:** Only `typing` from standard library (`Any`, `Protocol`).

**Exports:** `PipelineStage`, `Pipeline`

**`PipelineStage`:**
- A `Protocol` class with one method: `process(self, data: Any) → Any`
- This is defined HERE, not in `ports/`. It's a domain composition concept, not an external system boundary.

**`Pipeline`:**
- Constructor: `__init__(self, stages: list[PipelineStage] | None = None)` — stores stages in a private list, defaults to empty
- `add_stage(self, stage: PipelineStage) → Pipeline` — appends to the list, returns `self` for fluent chaining
- `run(self, data: Any) → Any` — iterates stages in order. Sets `result = data`. For each stage, sets `result = stage.process(result)`. Returns final `result`.

**Key rule:** No imports from `ports/` or anywhere outside standard library. `Pipeline` and `PipelineStage` are pure domain.

---

### `src/rag_agent/core/use_cases/__init__.py`

**Purpose:** Package marker. Empty.

---

### `src/rag_agent/core/use_cases/ingest.py`

**Purpose:** The `IngestUseCase` and its four pipeline stage classes.

**Imports:**
- From `core/domain/models`: `Document`, `Chunk`
- From `core/domain/pipeline`: `Pipeline`
- From `ports/`: `DocLoader`, `Chunker`, `Embedder`, `VectorStore`
- NEVER imports from `adapters/`

**Exports:** `IngestUseCase` (and the stage classes, though they're mainly internal)

**`LoadStage`:**
- Constructor: takes `loader: DocLoader`
- `process(self, data: None) → list[Document]`: calls `self._loader.load()`, returns the result

**`ChunkStage`:**
- Constructor: takes `chunker: Chunker`
- `process(self, documents: list[Document]) → list[Chunk]`: iterates each document, calls `self._chunker.chunk(doc)`, flattens all results into a single `list[Chunk]`. Order: chunks from document 0 first, then document 1, etc.

**`EmbedStage`:**
- Constructor: takes `embedder: Embedder`
- `process(self, chunks: list[Chunk]) → list[Chunk]`: extracts `[c.content for c in chunks]`, calls `self._embedder.embed(texts)` as one batch, creates NEW `Chunk` objects with `content` and `metadata` copied from originals and `embedding` set from the result. Returns the new list. Does NOT mutate input chunks.

**`StoreStage`:**
- Constructor: takes `store: VectorStore`
- `process(self, chunks: list[Chunk]) → list[Chunk]`: calls `self._store.add(chunks)`, returns the same `chunks` list unchanged (passthrough for pipeline continuation)

**`IngestUseCase`:**
- Constructor: takes `loader: DocLoader`, `chunker: Chunker`, `embedder: Embedder`, `store: VectorStore`. Builds a `Pipeline` internally using `Pipeline().add_stage(LoadStage(loader)).add_stage(ChunkStage(chunker)).add_stage(EmbedStage(embedder)).add_stage(StoreStage(store))`
- `execute(self) → list[Chunk]`: calls `self._pipeline.run(None)`, returns the result

---

### `src/rag_agent/core/use_cases/query.py`

**Purpose:** The `QueryUseCase`.

**Imports:**
- From `core/domain/models`: `QueryResult`
- From `ports/`: `Retriever`, `LLMProvider`
- Standard library: `time` (for timing metrics)
- NEVER imports from `adapters/`

**Exports:** `QueryUseCase`

**`QueryUseCase`:**
- Constructor: takes `retriever: Retriever`, `llm: LLMProvider`, `config_metadata: dict` (optional, for recording model name and retriever type in QueryResult — passed by `main.py`)
- `execute(self, question: str) → QueryResult`:
  1. Record start time
  2. Call `self._retriever.retrieve(question)` → get `list[Chunk]`. Record retrieval time.
  3. Build the context string: for each chunk, format as `[Source: {chunk.metadata["source"]}, Title: {chunk.metadata["title"]}]\n{chunk.content}`. Join all with double newlines.
  4. Build the prompt: a system instruction telling the LLM to answer based on the provided context and explicitly say if the context doesn't contain the answer, followed by the context block, followed by the question.
  5. Call `self._llm.generate(prompt)` → get answer string. Record generation time.
  6. Build `QueryResult` with `answer`, `chunks`, and `metadata` containing timing and config info (see metadata contract above).
  7. Return the `QueryResult`.

---

### `src/rag_agent/core/use_cases/evaluate.py`

**Purpose:** Placeholder for Phase 5. Not implemented in Phase 1.

**Contents:** Empty file or a stub `EvaluateUseCase` class with `execute()` raising `NotImplementedError`.

---

### `src/rag_agent/ports/__init__.py`

**Purpose:** Package marker. Optionally re-exports all port protocols for convenience.

---

### `src/rag_agent/ports/doc_loader.py`

**Purpose:** `DocLoader` protocol definition.

**Imports:** `Protocol` from `typing`, `Document` from `core/domain/models`

**Exports:** `DocLoader`

**`DocLoader`:**
- Protocol with one method: `load(self) → list[Document]`
- No constructor requirements in the protocol — that's the adapter's business

---

### `src/rag_agent/ports/chunker.py`

**Purpose:** `Chunker` protocol definition.

**Imports:** `Protocol` from `typing`, `Document` and `Chunk` from `core/domain/models`

**Exports:** `Chunker`

**`Chunker`:**
- Protocol with one method: `chunk(self, document: Document) → list[Chunk]`
- Adapter is responsible for: splitting the content, creating `Chunk` objects, copying `Document.metadata` into each chunk's metadata and adding chunk-specific keys (`chunk_index`, `total_chunks`, `start_char`, `end_char`)

---

### `src/rag_agent/ports/embedder.py`

**Purpose:** `Embedder` protocol definition.

**Imports:** `Protocol` from `typing`

**Exports:** `Embedder`

**`Embedder`:**
- Protocol with one method: `embed(self, texts: list[str]) → list[list[float]]`
- Input: list of N text strings. Output: list of N embedding vectors, same order. Each vector is a list of floats. Dimensionality depends on the model (e.g., 1536 for OpenAI `text-embedding-3-small`).

---

### `src/rag_agent/ports/vector_store.py`

**Purpose:** `VectorStore` protocol definition.

**Imports:** `Protocol` from `typing`, `Chunk` from `core/domain/models`

**Exports:** `VectorStore`

**`VectorStore`:**
- Two methods:
  - `add(self, chunks: list[Chunk]) → None` — persists chunks. The adapter must store: `chunk.content`, `chunk.metadata` (all keys), and `chunk.embedding`. Chunks without embeddings (`embedding is None`) should be rejected or raise an error — that's an upstream pipeline bug.
  - `search(self, embedding: list[float], top_k: int) → list[Chunk]` — returns up to `top_k` chunks most similar to the given embedding. Returned `Chunk` objects must have `content` and `metadata` populated. `embedding` field on returned chunks is optional (some stores don't return it).

---

### `src/rag_agent/ports/retriever.py`

**Purpose:** `Retriever` protocol definition.

**Imports:** `Protocol` from `typing`, `Chunk` from `core/domain/models`

**Exports:** `Retriever`

**`Retriever`:**
- Protocol with one method: `retrieve(self, query: str) → list[Chunk]`
- No `top_k` parameter — that's in the adapter's constructor via config.
- Implementation note (not in the protocol itself): `DenseRetriever` is a composite adapter. Its constructor takes `Embedder`, `VectorStore`, and `top_k: int`. `retrieve()` embeds the query string, then calls `VectorStore.search()` with the embedding and `top_k`. Other retriever types (BM25, hybrid) will have different constructor signatures.

---

### `src/rag_agent/ports/reranker.py`

**Purpose:** `Reranker` protocol definition.

**Imports:** `Protocol` from `typing`, `Chunk` from `core/domain/models`

**Exports:** `Reranker`

**`Reranker`:**
- Protocol with one method: `rerank(self, query: str, chunks: list[Chunk]) → list[Chunk]`
- Returns a reordered and possibly truncated list. The "none" adapter returns the input unchanged.

---

### `src/rag_agent/ports/llm.py`

**Purpose:** `LLMProvider` protocol definition.

**Imports:** `Protocol` from `typing`

**Exports:** `LLMProvider`

**`LLMProvider`:**
- Protocol with one method: `generate(self, prompt: str) → str`
- Model name, temperature, max_tokens are in the adapter's constructor via config. The port is just `str → str`.

---

### `src/rag_agent/adapters/__init__.py`

**Purpose:** Package marker. Empty.

Each subdirectory (`llms/`, `embedders/`, `vector_stores/`, `chunkers/`, `doc_loaders/`, `retrievers/`, `rerankers/`) also has an `__init__.py` — all empty.

---

### Adapter Files (One Per Provider — You Choose)

Adapters are the files where your TBD choices materialize. Each adapter file:

**Imports:** The port protocol it satisfies (from `ports/`), domain models (from `core/domain/models`), and the external library it wraps.

**Exports:** One class that satisfies the port protocol via structural subtyping.

**Constructor:** Takes config-driven params (provider-specific settings like model name, API URL, etc.) and any port dependencies (for composite adapters like `DenseRetriever`).

Below are specs for EVERY adapter slot. You implement one per slot for Phase 1 — whichever provider you pick.

#### `adapters/doc_loaders/markdown_loader.py`

**Class:** `MarkdownLoader`

**Constructor:** takes `path: str` (directory path to the 97 Things repo, from `config.data_source.path`)

**`load() → list[Document]`:**
1. Walk the directory recursively, find all `.md` files
2. For each file: read the content as UTF-8 string
3. Extract title: if the file has YAML front-matter with a `title` field, use that. Otherwise derive from filename (strip `.md`, replace `-` and `_` with spaces, title-case).
4. Build `Document` with `content` and `metadata` set per the metadata contract: `source`, `title`, `format="markdown"`, `file_size_bytes`, `loaded_at`
5. Return the list, sorted by `source` for deterministic ordering

#### `adapters/chunkers/fixed_size.py`

**Class:** `FixedSizeChunker`

**Constructor:** takes `chunk_size: int`, `chunk_overlap: int` (from config)

**`chunk(document: Document) → list[Chunk]`:**
1. Split `document.content` into chunks of `chunk_size` characters with `chunk_overlap` character overlap between consecutive chunks
2. For each chunk: create a `Chunk` with `content` set to the substring, `metadata` copied from `document.metadata` plus chunk-specific keys: `chunk_index`, `total_chunks`, `start_char`, `end_char`
3. `embedding` is `None` at this point
4. Return the list

Edge cases: if the document is shorter than `chunk_size`, return a single chunk. Last chunk may be shorter than `chunk_size`.

#### `adapters/embedders/` (your choice)

**Class name:** depends on provider (e.g., `OpenAIEmbedder`, `SentenceTransformerEmbedder`)

**Constructor:** takes `model: str` (from config), plus any provider-specific params (API key from env var, device for local models, etc.)

**`embed(texts: list[str]) → list[list[float]]`:**
1. Call the provider's embedding API/model with the batch of texts
2. Return the embedding vectors in the same order
3. Handle batching if the provider has a max batch size — split into sub-batches, concatenate results

#### `adapters/vector_stores/` (your choice)

**Class name:** depends on provider (e.g., `ChromaStore`, `FaissStore`, `QdrantStore`)

**Constructor:** takes provider-specific params from config (collection name, persist directory, server URL, etc.)

**`add(chunks: list[Chunk]) → None`:**
1. Extract content, metadata, and embeddings from each chunk
2. Verify all chunks have non-None embeddings (raise error if not — this is a pipeline bug)
3. Upsert into the store. Use a deterministic ID per chunk — suggested: hash of `source + chunk_index`, or `f"{metadata['source']}::{metadata['chunk_index']}"`

**`search(embedding: list[float], top_k: int) → list[Chunk]`:**
1. Query the store for the `top_k` nearest vectors to `embedding`
2. Reconstruct `Chunk` objects from stored content and metadata
3. Return sorted by similarity (most similar first)

#### `adapters/retrievers/dense.py`

**Class:** `DenseRetriever`

**Constructor:** takes `embedder: Embedder`, `store: VectorStore`, `top_k: int` (from config). This is a composite adapter — `main.py` wires the dependencies.

**`retrieve(query: str) → list[Chunk]`:**
1. Call `self._embedder.embed([query])` — embed the query as a single-item batch
2. Take the first (only) embedding from the result
3. Call `self._store.search(embedding, self._top_k)`
4. Return the result

#### `adapters/rerankers/no_rerank.py`

**Class:** `NoReranker`

**Constructor:** no params needed

**`rerank(query: str, chunks: list[Chunk]) → list[Chunk]`:** returns `chunks` unchanged. This is the Phase 1 default — a passthrough.

#### `adapters/llms/` (your choice)

**Class name:** depends on provider (e.g., `AnthropicLLM`, `OpenAILLM`, `OllamaLLM`)

**Constructor:** takes `model: str`, `temperature: float`, `max_tokens: int` (from config), reads API key from `os.environ`

**`generate(prompt: str) → str`:**
1. Call the provider's API with the prompt, model, temperature, max_tokens
2. Return the generated text as a string

---

### `src/rag_agent/config.py`

**Purpose:** Pydantic config models and YAML loader.

**Imports:** `pydantic` (`BaseModel`, `Field`, `model_validator`), `typing` (`Literal`), `yaml`, `pathlib` (`Path`)

**Exports:** `AppConfig`, and all sub-config models

**Sub-models** (one per YAML section):

`LLMConfig`: fields `provider` (Literal of valid provider strings), `model: str`, `temperature: float` (constrained `ge=0.0, le=2.0`), `max_tokens: int` (constrained `gt=0`). All with defaults.

`EmbedderConfig`: fields `provider` (Literal), `model: str`. All with defaults.

`VectorStoreConfig`: fields `provider` (Literal), `collection_name: str`, `persist_directory: str`, plus any provider-specific fields with defaults (e.g., `qdrant_url: str`).

`ChunkerConfig`: fields `strategy` (Literal), `chunk_size: int` (constrained `gt=0`), `chunk_overlap: int` (constrained `ge=0`). Cross-field validator: `chunk_overlap` must be strictly less than `chunk_size`.

`RetrieverConfig`: fields `strategy` (Literal), `top_k: int` (constrained `gt=0`).

`RerankerConfig`: fields `provider` (Literal), `top_k: int` (constrained `gt=0`).

`DataSourceConfig`: fields `path: str`, `type` (Literal).

**`AppConfig`:** composes all sub-models as fields, each with its own default instance. Class method `from_yaml(path: str | Path) → AppConfig`: reads file, `yaml.safe_load`, passes result dict to `AppConfig(**raw)`, Pydantic validates, returns. Missing keys use defaults.

---

### `src/rag_agent/main.py`

**Purpose:** CLI entry point AND composition root. This is the only file that imports from both `adapters/` and `core/use_cases/`. It reads config, picks the right adapter for each port, instantiates everything, and wires it into use cases.

**Imports:** CLI framework (your TBD choice), `AppConfig` from `config`, adapter classes from `adapters/`, `IngestUseCase` and `QueryUseCase` from `core/use_cases/`

**Exports:** the CLI app

**Adapter wiring:** For each port, `main.py` reads the config value and does a simple if/elif to pick the adapter class. No dynamic imports, no string-to-class mapping. Example logic for chunker:
- Read `config.chunker.strategy`
- If `"fixed_size"` → import and instantiate `FixedSizeChunker(chunk_size=config.chunker.chunk_size, chunk_overlap=config.chunker.chunk_overlap)`
- If `"semantic"` → import and instantiate `SemanticChunker(...)`
- Else → raise error with clear message

Same pattern for every port. For composite adapters like `DenseRetriever`, build the dependencies first (embedder, vector store), then pass them to the retriever constructor.

You can organize this as a helper function per port (e.g., `_build_chunker(config) → Chunker`) to keep the command functions clean, or inline it — your call.

**Commands:**

**`ingest`:**
- Accepts `--config` flag (string, defaults to `"config/default.yaml"`)
- Loads `AppConfig.from_yaml(config_path)`
- Builds adapters: DocLoader, Chunker, Embedder, VectorStore
- Creates `IngestUseCase(loader, chunker, embedder, store)`
- Calls `use_case.execute()`
- Prints: number of documents loaded, number of chunks created and stored

**`query`:**
- Accepts positional `question: str` argument
- Accepts `--config` flag (string, defaults to `"config/default.yaml"`)
- Loads `AppConfig.from_yaml(config_path)`
- Builds adapters: Embedder, VectorStore, then Retriever (composite), then LLMProvider
- Creates `QueryUseCase(retriever, llm, config_metadata={"model": config.llm.model, "retriever": config.retriever.strategy})`
- Calls `use_case.execute(question)`
- Prints: the answer, then a "Sources:" section listing each chunk's `metadata["title"]` and `metadata["source"]`

---

### Config and project files (non-Python)

**`pyproject.toml`:** Standard uv/pip project file. `[project]` section with name, version, Python requirement, dependencies. `[tool.uv]` or `[build-system]` as needed. Dependencies are the external libraries for whichever adapters you chose.

**`config/default.yaml`:** As defined in base plan. Fill in TBD values when you decide on providers.

**`.env`:** One line per API key. Format: `PROVIDER_API_KEY=sk-...`. Gitignored.

**`.gitignore`:** Include `.env`, `data/97-things/` (cloned repo), `data/vector_db/` (persisted store), `__pycache__/`, `.venv/`, `uv.lock` (optional — some teams commit it).

**`docker-compose.yml`:** Depends on your vector store choice. If it needs a server, define the service here. If not, this file can be minimal or empty for Phase 1.

---

### Test files (Phase 1 — minimal)

**`tests/fakes/`:** In-memory test doubles. One file per port you want to fake. Each fake class satisfies the port protocol with hardcoded or configurable behavior. Not implemented in Phase 1 step-by-step but the directory exists for Phase 2.

**`tests/unit/`:** Phase 1 focuses on manual testing. Unit tests come in Phase 2 with fakes.

**`tests/integration/`:** Phase 1 manual testing counts as informal integration testing. Formal integration tests come in Phase 2.