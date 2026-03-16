from rag_agent.domain.models import Chunk
from rag_agent.adapters.outbound.retrievers.hybrid_retriever import HybridRetriever
from tests.fakes import FakeRetriever


def test_hybrid_merges_results():
    # Given: dense returns chunks A, B and sparse returns chunks C, A
    chunk_a = Chunk(id="a", document_id="d1", content="chunk A")
    chunk_b = Chunk(id="b", document_id="d1", content="chunk B")
    chunk_c = Chunk(id="c", document_id="d1", content="chunk C")

    dense = FakeRetriever([chunk_a, chunk_b])
    sparse = FakeRetriever([chunk_c, chunk_a])
    retriever = HybridRetriever(
        dense_retriever=dense, sparse_retriever=sparse, top_k=3
    )

    # When: we retrieve
    results = retriever.retrieve("test")

    # Then: chunk A should rank highest (appears in both lists)
    assert results[0].id == "a"
    assert len(results) == 3


def test_hybrid_with_empty_dense():
    # Given: dense returns nothing, sparse returns chunks
    chunk_a = Chunk(id="a", document_id="d1", content="chunk A")
    dense = FakeRetriever([])
    sparse = FakeRetriever([chunk_a])
    retriever = HybridRetriever(
        dense_retriever=dense, sparse_retriever=sparse, top_k=5
    )

    # When: we retrieve
    results = retriever.retrieve("test")

    # Then: we still get the sparse results
    assert len(results) == 1
    assert results[0].id == "a"


def test_hybrid_with_empty_sparse():
    # Given: sparse returns nothing, dense returns chunks
    chunk_a = Chunk(id="a", document_id="d1", content="chunk A")
    dense = FakeRetriever([chunk_a])
    sparse = FakeRetriever([])
    retriever = HybridRetriever(
        dense_retriever=dense, sparse_retriever=sparse, top_k=5
    )

    # When: we retrieve
    results = retriever.retrieve("test")

    # Then: we still get the dense results
    assert len(results) == 1
    assert results[0].id == "a"


def test_hybrid_both_empty():
    # Given: both retrievers return nothing
    dense = FakeRetriever([])
    sparse = FakeRetriever([])
    retriever = HybridRetriever(
        dense_retriever=dense, sparse_retriever=sparse, top_k=5
    )

    # When: we retrieve
    results = retriever.retrieve("test")

    # Then: we get an empty list
    assert results == []


def test_hybrid_respects_top_k():
    # Given: 4 unique chunks across both retrievers, top_k=2
    chunks = [
        Chunk(id=f"c{i}", document_id="d1", content=f"chunk {i}")
        for i in range(4)
    ]
    dense = FakeRetriever(chunks[:2])
    sparse = FakeRetriever(chunks[2:])
    retriever = HybridRetriever(
        dense_retriever=dense, sparse_retriever=sparse, top_k=2
    )

    # When: we retrieve
    results = retriever.retrieve("test")

    # Then: only 2 results returned
    assert len(results) == 2
