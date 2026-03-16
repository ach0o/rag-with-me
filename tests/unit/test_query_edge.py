from rag_agent.domain.models import Chunk
from rag_agent.application.query import QueryUseCase
from tests.fakes import FakeRetriever, FakeLLM, FakeReranker


def test_query_with_no_retrieved_chunks():
    # Given: a retriever that returns nothing
    llm = FakeLLM(response="I don't have enough context.")
    use_case = QueryUseCase(
        retriever=FakeRetriever([]),
        llm=llm,
    )

    # When: we execute a query
    result = use_case.execute("anything?")

    # Then: the LLM still gets called with an empty context and returns its answer
    assert result.answer == "I don't have enough context."
    assert result.chunks == []


def test_query_with_empty_question():
    # Given: an empty question string
    chunks = [Chunk(document_id="d1", content="some context")]
    llm = FakeLLM()
    use_case = QueryUseCase(
        retriever=FakeRetriever(chunks),
        llm=llm,
    )

    # When: we execute with an empty string
    result = use_case.execute("")

    # Then: it should still work without crashing
    assert result.answer == "fake answer"
    assert "" in llm.last_prompt


def test_query_reranker_receives_all_chunks():
    # Given: 5 chunks and a reranker
    chunks = [
        Chunk(document_id="d1", content=f"chunk {i}") for i in range(5)
    ]
    reranker = FakeReranker()
    use_case = QueryUseCase(
        retriever=FakeRetriever(chunks),
        llm=FakeLLM(),
        reranker=reranker,
    )

    # When: we execute a query
    result = use_case.execute("test")

    # Then: the reranker should have processed all 5 chunks
    assert reranker.called
    assert len(result.chunks) == 5
