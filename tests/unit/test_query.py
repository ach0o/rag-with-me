from rag_agent.domain.models import Chunk
from rag_agent.application.query import QueryUseCase
from tests.fakes import FakeRetriever, FakeLLM, FakeReranker


def _make_chunks() -> list[Chunk]:
    return [
        Chunk(document_id="doc-1", content="Code reviews help find bugs early."),
        Chunk(document_id="doc-1", content="Pair programming is also effective."),
    ]


def test_query_returns_answer_with_source_chunks():
    # Given: retrieved chunks and an LLM that returns a specific answer
    chunks = _make_chunks()
    llm = FakeLLM(response="Reviews catch bugs.")
    use_case = QueryUseCase(
        retriever=FakeRetriever(chunks),
        llm=llm,
    )

    # When: we execute a query
    result = use_case.execute("Why code reviews?")

    # Then: the result contains the LLM answer and the retrieved chunks
    assert result.answer == "Reviews catch bugs."
    assert len(result.chunks) == 2


def test_query_includes_context_and_question_in_prompt():
    # Given: chunks with known content
    chunks = _make_chunks()
    llm = FakeLLM()
    use_case = QueryUseCase(
        retriever=FakeRetriever(chunks),
        llm=llm,
    )

    # When: we execute a query
    use_case.execute("question?")

    # Then: the prompt sent to the LLM should contain both chunk content and the question
    assert "Code reviews help find bugs early." in llm.last_prompt
    assert "question?" in llm.last_prompt


def test_query_preserves_chunk_order_without_reranker():
    # Given: chunks in a specific order and no reranker
    chunks = _make_chunks()
    use_case = QueryUseCase(
        retriever=FakeRetriever(chunks),
        llm=FakeLLM(),
    )

    # When: we execute a query
    result = use_case.execute("test")

    # Then: chunks should be in the original retrieval order
    assert result.chunks[0].content == "Code reviews help find bugs early."


def test_query_applies_reranker_when_provided():
    # Given: a reranker that reverses chunk order
    chunks = _make_chunks()
    reranker = FakeReranker()
    use_case = QueryUseCase(
        retriever=FakeRetriever(chunks),
        llm=FakeLLM(),
        reranker=reranker,
    )

    # When: we execute a query
    result = use_case.execute("test")

    # Then: the reranker should have been called and chunks reordered
    assert reranker.called
    assert result.chunks[0].content == "Pair programming is also effective."
