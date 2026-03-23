import pytest

from rag_agent.application.metrics.recall import RecallMetric
from rag_agent.application.metrics.precision import PrecisionMetric
from rag_agent.application.metrics.mrr import MRRMetric
from rag_agent.application.metrics.faithfulness import FaithfulnessMetric
from rag_agent.application.metrics.semantic_similarity import SemanticSimilarityMetric
from rag_agent.application.evaluate import EvalResult
from rag_agent.domain.models import Chunk
from tests.fakes import FakeLLM, FakeEmbedder


def _chunk(source: str, content: str = "some text") -> Chunk:
    return Chunk(document_id="doc1", content=content, metadata={"source": source})


# ---------------------------------------------------------------------------
# RecallMetric
# ---------------------------------------------------------------------------


class TestRecallMetric:
    def test_hit_returns_one(self):
        # Given: a chunk whose source matches the expected source
        metric = RecallMetric()
        chunks = [_chunk("docs/guide.pdf")]

        # When: scoring the item
        score = metric.score_item("guide.pdf", "answer", "answer", chunks)

        # Then: recall is 1.0
        assert score == 1.0

    def test_miss_returns_zero(self):
        # Given: chunks that do not contain the expected source
        metric = RecallMetric()
        chunks = [_chunk("docs/other.pdf")]

        # When: scoring the item
        score = metric.score_item("guide.pdf", "answer", "answer", chunks)

        # Then: recall is 0.0
        assert score == 0.0

    def test_empty_chunks_returns_zero(self):
        # Given: no chunks returned
        metric = RecallMetric()

        # When: scoring with an empty chunk list
        score = metric.score_item("guide.pdf", "answer", "answer", [])

        # Then: recall is 0.0
        assert score == 0.0

    def test_aggregate_averages_scores(self):
        # Given: two results, one hit and one miss
        metric = RecallMetric()
        results = [
            EvalResult(
                question="q1",
                expected_answer="a1",
                expected_source="s1",
                actual_answer="a1",
                retrieved_sources=["s1"],
                scores={metric.name: 1.0},
            ),
            EvalResult(
                question="q2",
                expected_answer="a2",
                expected_source="s2",
                actual_answer="a2",
                retrieved_sources=[],
                scores={metric.name: 0.0},
            ),
        ]

        # When: aggregating
        avg = metric.aggregate(results)

        # Then: average is 0.5
        assert avg == pytest.approx(0.5)

    def test_aggregate_empty_results(self):
        # Given: no results at all
        metric = RecallMetric()

        # When: aggregating an empty list
        avg = metric.aggregate([])

        # Then: returns 0.0
        assert avg == 0.0


# ---------------------------------------------------------------------------
# PrecisionMetric
# ---------------------------------------------------------------------------


class TestPrecisionMetric:
    def test_all_chunks_match(self):
        # Given: all chunks contain the expected source
        metric = PrecisionMetric()
        chunks = [_chunk("docs/guide.pdf"), _chunk("docs/guide.pdf")]

        # When: scoring
        score = metric.score_item("guide.pdf", "answer", "answer", chunks)

        # Then: precision is 1.0
        assert score == 1.0

    def test_half_chunks_match(self):
        # Given: one of two chunks matches
        metric = PrecisionMetric()
        chunks = [_chunk("docs/guide.pdf"), _chunk("docs/other.pdf")]

        # When: scoring
        score = metric.score_item("guide.pdf", "answer", "answer", chunks)

        # Then: precision is 0.5
        assert score == pytest.approx(0.5)

    def test_no_chunks_match(self):
        # Given: no chunks match
        metric = PrecisionMetric()
        chunks = [_chunk("docs/other.pdf"), _chunk("docs/another.pdf")]

        # When: scoring
        score = metric.score_item("guide.pdf", "answer", "answer", chunks)

        # Then: precision is 0.0
        assert score == 0.0

    def test_empty_chunks_returns_zero(self):
        # Given: empty chunk list
        metric = PrecisionMetric()

        # When: scoring
        score = metric.score_item("guide.pdf", "answer", "answer", [])

        # Then: precision is 0.0
        assert score == 0.0


# ---------------------------------------------------------------------------
# MRRMetric
# ---------------------------------------------------------------------------


class TestMRRMetric:
    def test_first_position(self):
        # Given: expected source is the first chunk
        metric = MRRMetric()
        chunks = [_chunk("docs/guide.pdf"), _chunk("docs/other.pdf")]

        # When: scoring
        score = metric.score_item("guide.pdf", "answer", "answer", chunks)

        # Then: reciprocal rank is 1.0
        assert score == 1.0

    def test_second_position(self):
        # Given: expected source is the second chunk
        metric = MRRMetric()
        chunks = [_chunk("docs/other.pdf"), _chunk("docs/guide.pdf")]

        # When: scoring
        score = metric.score_item("guide.pdf", "answer", "answer", chunks)

        # Then: reciprocal rank is 0.5
        assert score == pytest.approx(0.5)

    def test_not_found(self):
        # Given: expected source is not in any chunk
        metric = MRRMetric()
        chunks = [_chunk("docs/other.pdf")]

        # When: scoring
        score = metric.score_item("guide.pdf", "answer", "answer", chunks)

        # Then: reciprocal rank is 0.0
        assert score == 0.0

    def test_aggregate(self):
        # Given: two results — first position and not found
        metric = MRRMetric()
        results = [
            EvalResult(
                question="q1",
                expected_answer="a",
                expected_source="s",
                actual_answer="a",
                retrieved_sources=[],
                scores={metric.name: 1.0},
            ),
            EvalResult(
                question="q2",
                expected_answer="a",
                expected_source="s",
                actual_answer="a",
                retrieved_sources=[],
                scores={metric.name: 0.0},
            ),
        ]

        # When: aggregating
        avg = metric.aggregate(results)

        # Then: average is 0.5
        assert avg == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# FaithfulnessMetric
# ---------------------------------------------------------------------------


class TestFaithfulnessMetric:
    def test_valid_score_from_llm(self):
        # Given: a judge LLM that returns a valid numeric score
        llm = FakeLLM(response="0.85")
        metric = FaithfulnessMetric(judge_llm=llm)

        # When: scoring
        score = metric.score_item("src", "expected", "actual", [])

        # Then: the parsed score is returned
        assert score == pytest.approx(0.85)

    def test_score_clamped_to_one(self):
        # Given: a judge LLM that returns a value above 1.0
        llm = FakeLLM(response="1.5")
        metric = FaithfulnessMetric(judge_llm=llm)

        # When: scoring
        score = metric.score_item("src", "expected", "actual", [])

        # Then: the score is clamped to 1.0
        assert score == 1.0

    def test_garbage_returns_zero(self):
        # Given: a judge LLM that returns non-numeric text
        llm = FakeLLM(response="I cannot rate this")
        metric = FaithfulnessMetric(judge_llm=llm)

        # When: scoring
        score = metric.score_item("src", "expected", "actual", [])

        # Then: falls back to 0.0
        assert score == 0.0

    def test_prompt_contains_expected_and_actual(self):
        # Given: a judge LLM
        llm = FakeLLM(response="0.5")
        metric = FaithfulnessMetric(judge_llm=llm)

        # When: scoring with specific texts
        metric.score_item("src", "the expected answer", "the actual answer", [])

        # Then: the prompt sent to the LLM includes both answers
        assert "the expected answer" in llm.last_prompt
        assert "the actual answer" in llm.last_prompt


# ---------------------------------------------------------------------------
# SemanticSimilarityMetric
# ---------------------------------------------------------------------------


class TestSemanticSimilarityMetric:
    def test_identical_texts_high_similarity(self):
        # Given: an embedder that returns the same vector for any text
        #        (FakeEmbedder returns [0.1, 0.2, 0.3] for every input)
        embedder = FakeEmbedder()
        metric = SemanticSimilarityMetric(embedder=embedder)

        # When: scoring identical texts (same embedding)
        score = metric.score_item("src", "hello world", "hello world", [])

        # Then: cosine similarity of identical vectors is 1.0
        assert score == pytest.approx(1.0)

    def test_different_embeddings_lower_similarity(self):
        # Given: an embedder that returns different vectors for different texts
        class DifferentiatingEmbedder:
            def embed(self, texts: list[str]) -> list[list[float]]:
                results = []
                for t in texts:
                    if "expected" in t:
                        results.append([1.0, 0.0, 0.0])
                    else:
                        results.append([0.0, 1.0, 0.0])
                return results

        metric = SemanticSimilarityMetric(embedder=DifferentiatingEmbedder())

        # When: scoring texts with orthogonal embeddings
        score = metric.score_item("src", "expected", "actual", [])

        # Then: cosine similarity of orthogonal vectors is 0.0
        assert score == pytest.approx(0.0)

    def test_cosine_similarity_known_value(self):
        # Given: known vectors
        # When: computing cosine similarity directly
        sim = SemanticSimilarityMetric._cosine_similarity(
            [1.0, 0.0], [1.0, 0.0]
        )

        # Then: identical unit vectors yield 1.0
        assert sim == pytest.approx(1.0)

    def test_cosine_similarity_zero_vector(self):
        # Given: a zero vector
        # When: computing cosine similarity
        sim = SemanticSimilarityMetric._cosine_similarity(
            [0.0, 0.0], [1.0, 0.0]
        )

        # Then: returns 0.0 to avoid division by zero
        assert sim == 0.0
