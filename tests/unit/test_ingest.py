from rag_agent.domain.models import Document
from rag_agent.application.ingest import IngestUseCase
from tests.fakes import (
    FakeDocLoader,
    FakeChunker,
    FakeEmbedder,
    FakeVectorStore,
    FakeDocumentRepository,
    FakeChunkRepository,
)


def _make_docs() -> list[Document]:
    return [
        Document(content="First document", metadata={"source": "a.md"}),
        Document(content="Second document", metadata={"source": "b.md"}),
    ]


def test_ingest_returns_chunks_with_embeddings():
    # Given: two documents loaded through the full ingestion pipeline
    docs = _make_docs()
    use_case = IngestUseCase(
        loader=FakeDocLoader(docs),
        chunker=FakeChunker(),
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
    )

    # When: we execute the pipeline
    chunks = use_case.execute()

    # Then: we get one chunk per document, each with an embedding
    assert len(chunks) == 2
    assert all(c.embedding is not None for c in chunks)


def test_ingest_stores_chunks_in_vector_store():
    # Given: a vector store wired into the pipeline
    docs = _make_docs()
    store = FakeVectorStore()
    use_case = IngestUseCase(
        loader=FakeDocLoader(docs),
        chunker=FakeChunker(),
        embedder=FakeEmbedder(),
        vector_store=store,
    )

    # When: we execute the pipeline
    use_case.execute()

    # Then: the vector store should contain the embedded chunks
    assert len(store.chunks) == 2


def test_ingest_persists_documents_and_chunks():
    # Given: repositories wired into the pipeline
    docs = _make_docs()
    doc_repo = FakeDocumentRepository()
    chunk_repo = FakeChunkRepository()
    use_case = IngestUseCase(
        loader=FakeDocLoader(docs),
        chunker=FakeChunker(),
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        document_repository=doc_repo,
        chunk_repository=chunk_repo,
    )

    # When: we execute the pipeline
    use_case.execute()

    # Then: both repositories should contain the persisted data
    assert len(doc_repo.documents) == 2
    assert len(chunk_repo.chunks) == 2


def test_ingest_without_repos_skips_persistence():
    # Given: a pipeline with no repositories (database disabled)
    docs = _make_docs()
    use_case = IngestUseCase(
        loader=FakeDocLoader(docs),
        chunker=FakeChunker(),
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
    )

    # When: we execute the pipeline
    chunks = use_case.execute()

    # Then: it should still complete successfully
    assert len(chunks) == 2
