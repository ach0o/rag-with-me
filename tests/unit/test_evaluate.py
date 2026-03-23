import json

import pytest

from rag_agent.application.evaluate import EvaluateUseCase, EvalResult, EvalSummary
from rag_agent.application.metrics.recall import RecallMetric
from rag_agent.application.metrics.precision import PrecisionMetric
from rag_agent.domain.models import Chunk, QueryResult


class FakeQueryExecutor:
    """Minimal fake that satisfies the query_executor duck-type used by EvaluateUseCase."""

    def __init__(self, answer: str = "fake answer", chunks: list[Chunk] | None = None):
        self._answer = answer
        self._chunks = chunks or []
        self.questions: list[str] = []

    def execute(self, question: str) -> QueryResult:
        self.questions.append(question)
        return QueryResult(answer=self._answer, chunks=self._chunks)


def _make_dataset(tmp_path, items: list[dict]) -> str:
    path = tmp_path / "gold_qa.json"
    path.write_text(json.dumps(items))
    return str(path)


# ---------------------------------------------------------------------------
# EvaluateUseCase
# ---------------------------------------------------------------------------


class TestEvaluateUseCase:
    def test_full_flow_populates_scores(self, tmp_path):
        # Given: a dataset with one question, a matching chunk, and two metrics
        chunk = Chunk(
            document_id="d1",
            content="relevant text",
            metadata={"source": "docs/guide.pdf"},
        )
        dataset_path = _make_dataset(
            tmp_path,
            [
                {
                    "question": "What is X?",
                    "expected_answer": "X is Y",
                    "expected_source": "guide.pdf",
                }
            ],
        )
        executor = FakeQueryExecutor(answer="X is Y", chunks=[chunk])
        metrics = [RecallMetric(), PrecisionMetric()]
        use_case = EvaluateUseCase(
            query_executor=executor,
            metrics=metrics,
            dataset_path=dataset_path,
        )

        # When: running the evaluation
        summary = use_case.execute()

        # Then: each result has scores for both metrics
        assert len(summary.results) == 1
        result = summary.results[0]
        assert "retrieval/recall_at_k" in result.scores
        assert "retrieval/precision_at_k" in result.scores
        assert result.scores["retrieval/recall_at_k"] == 1.0
        assert result.scores["retrieval/precision_at_k"] == 1.0

    def test_summary_aggregates_correctly(self, tmp_path):
        # Given: a dataset with two questions; one has a matching chunk, the other does not
        matching_chunk = Chunk(
            document_id="d1",
            content="text",
            metadata={"source": "docs/guide.pdf"},
        )
        non_matching_chunk = Chunk(
            document_id="d2",
            content="text",
            metadata={"source": "docs/other.pdf"},
        )
        dataset_path = _make_dataset(
            tmp_path,
            [
                {
                    "question": "Q1",
                    "expected_answer": "A1",
                    "expected_source": "guide.pdf",
                },
                {
                    "question": "Q2",
                    "expected_answer": "A2",
                    "expected_source": "guide.pdf",
                },
            ],
        )

        # The executor alternates chunks per call
        class AlternatingExecutor:
            def __init__(self):
                self._call = 0

            def execute(self, question: str) -> QueryResult:
                self._call += 1
                if self._call == 1:
                    return QueryResult(answer="A1", chunks=[matching_chunk])
                return QueryResult(answer="A2", chunks=[non_matching_chunk])

        metrics = [RecallMetric()]
        use_case = EvaluateUseCase(
            query_executor=AlternatingExecutor(),
            metrics=metrics,
            dataset_path=dataset_path,
        )

        # When: running evaluation
        summary = use_case.execute()

        # Then: summary score is the average of 1.0 and 0.0
        assert summary.total_questions == 2
        assert summary.scores["retrieval/recall_at_k"] == pytest.approx(0.5)

    def test_eval_result_scores_dict_populated(self, tmp_path):
        # Given: a single-item dataset and one metric
        chunk = Chunk(
            document_id="d1",
            content="text",
            metadata={"source": "docs/guide.pdf"},
        )
        dataset_path = _make_dataset(
            tmp_path,
            [
                {
                    "question": "Q",
                    "expected_answer": "A",
                    "expected_source": "guide.pdf",
                }
            ],
        )
        executor = FakeQueryExecutor(answer="A", chunks=[chunk])
        use_case = EvaluateUseCase(
            query_executor=executor,
            metrics=[RecallMetric()],
            dataset_path=dataset_path,
        )

        # When: running evaluation
        summary = use_case.execute()

        # Then: the result contains the question text and expected fields
        result = summary.results[0]
        assert result.question == "Q"
        assert result.expected_answer == "A"
        assert result.actual_answer == "A"
        assert result.retrieved_sources == ["docs/guide.pdf"]
        assert isinstance(result.scores, dict)
        assert len(result.scores) == 1

    def test_empty_dataset(self, tmp_path):
        # Given: an empty dataset
        dataset_path = _make_dataset(tmp_path, [])
        executor = FakeQueryExecutor()
        metrics = [RecallMetric()]
        use_case = EvaluateUseCase(
            query_executor=executor,
            metrics=metrics,
            dataset_path=dataset_path,
        )

        # When: running evaluation
        summary = use_case.execute()

        # Then: no results and aggregate returns 0.0
        assert summary.total_questions == 0
        assert summary.results == []
        assert summary.scores["retrieval/recall_at_k"] == 0.0

    def test_executor_receives_questions(self, tmp_path):
        # Given: a dataset with two questions
        dataset_path = _make_dataset(
            tmp_path,
            [
                {
                    "question": "First?",
                    "expected_answer": "A",
                    "expected_source": "s",
                },
                {
                    "question": "Second?",
                    "expected_answer": "B",
                    "expected_source": "s",
                },
            ],
        )
        executor = FakeQueryExecutor()
        use_case = EvaluateUseCase(
            query_executor=executor,
            metrics=[],
            dataset_path=dataset_path,
        )

        # When: running evaluation
        use_case.execute()

        # Then: the executor was called with each question in order
        assert executor.questions == ["First?", "Second?"]
