from rag_agent.domain.models import Chunk
from rag_agent.adapters.outbound.retrievers.bm25_sparse_retriever import (
    BM25SparseRetriever,
)
from tests.fakes import FakeChunkRepository


def _make_corpus() -> list[Chunk]:
    return [
        Chunk(document_id="d1", content="python is a programming language"),
        Chunk(document_id="d1", content="java is also a programming language"),
        Chunk(document_id="d1", content="the weather is nice today"),
    ]


def test_bm25_empty_corpus():
    # Given: a chunk repository with no chunks
    repo = FakeChunkRepository()
    retriever = BM25SparseRetriever(chunk_repository=repo, top_k=5)

    # When: we retrieve
    results = retriever.retrieve("python")

    # Then: we get an empty list
    assert results == []


def test_bm25_ranks_relevant_chunk_first():
    # Given: a corpus with one chunk about python
    repo = FakeChunkRepository()
    repo.chunks = _make_corpus()
    retriever = BM25SparseRetriever(chunk_repository=repo, top_k=2)

    # When: we query for "python"
    results = retriever.retrieve("python")

    # Then: the python chunk should rank first
    assert len(results) == 2
    assert "python" in results[0].content


def test_bm25_no_matching_terms():
    # Given: a corpus with no chunks matching the query terms
    repo = FakeChunkRepository()
    repo.chunks = _make_corpus()
    retriever = BM25SparseRetriever(chunk_repository=repo, top_k=2)

    # When: we query with terms not in the corpus
    results = retriever.retrieve("kubernetes docker")

    # Then: we still get results (top_k chunks) but all with score 0
    assert len(results) == 2


def test_bm25_respects_top_k():
    # Given: a corpus with 3 chunks and top_k=1
    repo = FakeChunkRepository()
    repo.chunks = _make_corpus()
    retriever = BM25SparseRetriever(chunk_repository=repo, top_k=1)

    # When: we retrieve
    results = retriever.retrieve("programming language")

    # Then: only 1 result is returned
    assert len(results) == 1


def test_bm25_case_insensitive():
    # Given: a corpus with lowercase content
    repo = FakeChunkRepository()
    repo.chunks = _make_corpus()
    retriever = BM25SparseRetriever(chunk_repository=repo, top_k=1)

    # When: we query with uppercase
    results = retriever.retrieve("PYTHON")

    # Then: it should still match
    assert "python" in results[0].content
