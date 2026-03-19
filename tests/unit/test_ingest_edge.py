from rag_agent.domain.models import Chunk, Document
from rag_agent.application.ingest import IngestUseCase
from tests.fakes import (
    FakeDocLoader,
    FakeChunker,
    FakeEmbedder,
    FakeVectorStore,
)


class MultiChunkFakeChunker:
    """Produces multiple chunks per document to test flattening."""

    def chunk(self, document: Document) -> list[Chunk]:
        return [
            Chunk(
                document_id=document.id,
                content=f"{document.content} - part {i}",
                metadata={**document.metadata, "chunk_index": i},
            )
            for i in range(3)
        ]


def test_ingest_with_empty_corpus():
    # Given: a loader that returns no documents
    use_case = IngestUseCase(
        loaders=[FakeDocLoader([])],
        chunker=FakeChunker(),
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
    )

    # When: we execute the pipeline
    chunks = use_case.execute()

    # Then: we get an empty list (no crash)
    assert chunks == []


def test_ingest_flattens_multiple_chunks_per_document():
    # Given: one document and a chunker that produces 3 chunks per doc
    docs = [Document(content="Hello world", metadata={"source": "test.md"})]
    store = FakeVectorStore()
    use_case = IngestUseCase(
        loaders=[FakeDocLoader(docs)],
        chunker=MultiChunkFakeChunker(),
        embedder=FakeEmbedder(),
        vector_store=store,
    )

    # When: we execute the pipeline
    chunks = use_case.execute()

    # Then: all 3 chunks are produced, embedded, and stored
    assert len(chunks) == 3
    assert len(store.chunks) == 3
    assert all(c.embedding is not None for c in chunks)


def test_ingest_preserves_chunk_document_id():
    # Given: a document with a known id
    doc = Document(content="text", metadata={"source": "a.md"})
    use_case = IngestUseCase(
        loaders=[FakeDocLoader([doc])],
        chunker=FakeChunker(),
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
    )

    # When: we execute the pipeline
    chunks = use_case.execute()

    # Then: each chunk should reference the parent document's id
    assert chunks[0].document_id == doc.id


def test_ingest_multiple_loaders():
    # Given: two loaders each returning different documents
    docs_a = [Document(content="From loader A", metadata={"source": "a.md"})]
    docs_b = [Document(content="From loader B", metadata={"source": "b.pdf"})]
    store = FakeVectorStore()
    use_case = IngestUseCase(
        loaders=[FakeDocLoader(docs_a), FakeDocLoader(docs_b)],
        chunker=FakeChunker(),
        embedder=FakeEmbedder(),
        vector_store=store,
    )

    # When: we execute the pipeline
    chunks = use_case.execute()

    # Then: documents from both loaders are merged and processed
    assert len(chunks) == 2
    assert len(store.chunks) == 2
    contents = {c.content for c in chunks}
    assert "From loader A" in contents
    assert "From loader B" in contents
