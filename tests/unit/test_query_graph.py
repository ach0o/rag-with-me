from rag_agent.domain.models import Chunk
from rag_agent.application.query_graph import QueryGraphBuilder
from tests.fakes import FakeRetriever, FakeLLM, FakeReranker


def _make_chunks() -> list[Chunk]:
    return [
        Chunk(document_id="doc-1", content="Code reviews help find bugs early."),
        Chunk(document_id="doc-1", content="Pair programming is also effective."),
    ]


class SequenceLLM:
    """LLM that returns different responses in sequence.
    Useful for testing grader → rephraser → grader → generator flows."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._call_index = 0
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        response = self._responses[self._call_index]
        self._call_index += 1
        return response


def test_graph_good_context_no_rephrase():
    # Given: an LLM that grades context as "good" and then generates an answer
    chunks = _make_chunks()
    llm = SequenceLLM(["good", "The answer is 42."])
    builder = QueryGraphBuilder(
        retriever=FakeRetriever(chunks),
        llm=llm,
    )

    # When: we execute a query
    result = builder.execute("What is the answer?")

    # Then: no rephrase happened, answer is generated directly
    assert result.answer == "The answer is 42."
    assert result.metadata["attempts"] == 0
    assert result.metadata["context_quality"] == "good"
    assert len(result.chunks) == 2


def test_graph_poor_context_triggers_rephrase():
    # Given: an LLM that grades "poor" first, then rephrases, then grades "good", then generates
    chunks = _make_chunks()
    llm = SequenceLLM([
        "poor",                    # 1st grade: poor context
        "rephrased question",      # rephrase
        "good",                    # 2nd grade: good context
        "Better answer now.",      # generate
    ])
    builder = QueryGraphBuilder(
        retriever=FakeRetriever(chunks),
        llm=llm,
    )

    # When: we execute a query
    result = builder.execute("vague question")

    # Then: one rephrase attempt was made before generating
    assert result.answer == "Better answer now."
    assert result.metadata["attempts"] == 1
    assert result.metadata["context_quality"] == "good"


def test_graph_max_retries_exhausted():
    # Given: an LLM that always grades "poor" — forces max retries
    chunks = _make_chunks()
    llm = SequenceLLM([
        "poor",                    # 1st grade
        "rephrased v1",            # 1st rephrase
        "poor",                    # 2nd grade
        "rephrased v2",            # 2nd rephrase
        "poor",                    # 3rd grade — max_attempts reached, go to generate
        "Forced answer.",          # generate despite poor context
    ])
    builder = QueryGraphBuilder(
        retriever=FakeRetriever(chunks),
        llm=llm,
        max_attempts=2,
    )

    # When: we execute a query
    result = builder.execute("impossible question")

    # Then: it generates anyway after exhausting retries
    assert result.answer == "Forced answer."
    assert result.metadata["attempts"] == 2
    assert result.metadata["context_quality"] == "poor"


def test_graph_with_reranker():
    # Given: a reranker that reverses chunk order
    chunks = _make_chunks()
    reranker = FakeReranker()
    llm = SequenceLLM(["good", "Reranked answer."])
    builder = QueryGraphBuilder(
        retriever=FakeRetriever(chunks),
        llm=llm,
        reranker=reranker,
    )

    # When: we execute a query
    result = builder.execute("test")

    # Then: the reranker was applied and chunks are reordered
    assert reranker.called
    assert result.chunks[0].content == "Pair programming is also effective."
    assert result.answer == "Reranked answer."


def test_graph_unknown_grade_defaults_to_good():
    # Given: an LLM that returns an unexpected grade value
    chunks = _make_chunks()
    llm = SequenceLLM(["maybe", "Answer anyway."])
    builder = QueryGraphBuilder(
        retriever=FakeRetriever(chunks),
        llm=llm,
    )

    # When: we execute a query
    result = builder.execute("test")

    # Then: unknown grade defaults to "good", no rephrase
    assert result.metadata["context_quality"] == "good"
    assert result.metadata["attempts"] == 0
    assert result.answer == "Answer anyway."


def test_graph_empty_retrieval():
    # Given: a retriever that returns no chunks
    llm = SequenceLLM(["poor", "rephrased", "poor", "No context available."])
    builder = QueryGraphBuilder(
        retriever=FakeRetriever([]),
        llm=llm,
        max_attempts=1,
    )

    # When: we execute a query
    result = builder.execute("anything")

    # Then: it still produces an answer after exhausting retries
    assert result.chunks == []
    assert result.answer == "No context available."


def test_graph_preserves_original_question():
    # Given: an LLM that rephrases and then generates
    chunks = _make_chunks()
    llm = SequenceLLM(["poor", "better query", "good", "Final answer."])
    builder = QueryGraphBuilder(
        retriever=FakeRetriever(chunks),
        llm=llm,
    )

    # When: we execute with a specific question
    result = builder.execute("original question here")

    # Then: the generate prompt uses the original question, not the rephrased one
    generate_prompt = llm.prompts[-1]
    assert "original question here" in generate_prompt
