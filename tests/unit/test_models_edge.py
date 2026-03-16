from rag_agent.domain.models import Chunk, Document, QueryResult


def test_document_with_empty_content():
    # Given: a document with empty string content
    doc = Document(content="")

    # Then: it should still have an id and empty content
    assert doc.id
    assert doc.content == ""


def test_document_with_empty_metadata():
    # Given: a document with no metadata provided
    doc = Document(content="text")

    # Then: metadata should default to empty dict
    assert doc.metadata == {}


def test_chunk_metadata_is_independent():
    # Given: two chunks created from the same document
    chunk1 = Chunk(document_id="d1", content="a", metadata={"x": 1})
    chunk2 = Chunk(document_id="d1", content="b", metadata={"x": 2})

    # Then: their metadata should be independent
    assert chunk1.metadata["x"] == 1
    assert chunk2.metadata["x"] == 2


def test_query_result_with_chunks():
    # Given: a QueryResult created with actual chunks
    chunks = [Chunk(document_id="d1", content="c1")]
    result = QueryResult(answer="yes", chunks=chunks)

    # Then: chunks should be accessible
    assert len(result.chunks) == 1
    assert result.chunks[0].content == "c1"
