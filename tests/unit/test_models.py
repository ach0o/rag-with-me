from rag_agent.domain.models import Chunk, Document, QueryResult


def test_document_has_auto_id():
    # Given: a document created with content and metadata
    doc = Document(content="hello", metadata={"source": "test"})

    # Then: it should have an auto-generated id and preserve its content
    assert doc.id
    assert doc.content == "hello"


def test_document_ids_are_unique():
    # Given: two documents created separately
    doc1 = Document(content="a")
    doc2 = Document(content="b")

    # Then: each should receive a unique id
    assert doc1.id != doc2.id


def test_chunk_requires_document_id():
    # Given: a chunk created with a document reference
    chunk = Chunk(document_id="doc-1", content="text")

    # Then: it should store the document_id and default embedding to None
    assert chunk.document_id == "doc-1"
    assert chunk.embedding is None


def test_chunk_with_embedding():
    # Given: a chunk created with an explicit embedding
    chunk = Chunk(document_id="doc-1", content="text", embedding=[0.1, 0.2])

    # Then: the embedding should be preserved as-is
    assert chunk.embedding == [0.1, 0.2]


def test_query_result_defaults():
    # Given: a QueryResult created with only an answer
    result = QueryResult(answer="yes")

    # Then: chunks and metadata should default to empty collections
    assert result.answer == "yes"
    assert result.chunks == []
    assert result.metadata == {}
