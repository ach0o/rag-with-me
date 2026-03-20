import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rag_agent.application.query import QueryUseCase
from rag_agent.application.query_graph import QueryGraphBuilder
from rag_agent.domain.models import Chunk
from rag_agent.domain.ports import LLMProvider


@dataclass
class EvalResult:
    question: str
    expected_answer: str
    expected_source: str
    actual_answer: str
    retrieved_sources: list[str]
    source_hit: bool
    faithfulness_score: float


@dataclass
class EvalSummary:
    results: list[EvalResult]
    total_questions: int
    scores: dict[str, float]


FAITHFULNESS_PROMPT = """Rate how well the actual answer matches the expected answer on a scale of 0.0 to 1.0.

- 1.0 = covers the same key points, factually consistent
- 0.5 = partially correct, missing important details
- 0.0 = wrong or completely unrelated

Expected answer: {expected}

Actual answer: {actual}

Reply with only a number between 0.0 and 1.0, nothing else.

Score:"""


class EvaluateUseCase:
    def __init__(
        self,
        query_executor: QueryUseCase | QueryGraphBuilder,
        judge_llm: LLMProvider,
        dataset_path: str = "eval_data/gold_qa.json",
    ) -> None:
        self._query_executor = query_executor
        self._judge_llm = judge_llm
        self._dataset_path = Path(dataset_path)
        self._metrics: dict[str, Callable[[list[EvalResult]], float]] = {
            "recall_at_k": self._calc_recall,
            "avg_faithfulness": self._calc_avg_faithfulness,
        }

    def _load_dataset(self) -> list[dict[str, str]]:
        with open(self._dataset_path) as f:
            return json.load(f)

    def _evaluate_item(self, item: dict[str, str]) -> EvalResult:
        query_result = self._query_executor.execute(item["question"])
        return EvalResult(
            question=item["question"],
            expected_answer=item["expected_answer"],
            expected_source=item["expected_source"],
            actual_answer=query_result.answer,
            retrieved_sources=[
                c.metadata.get("source", "unknown") for c in query_result.chunks
            ],
            source_hit=self._check_source_hit(
                item["expected_source"], query_result.chunks
            ),
            faithfulness_score=self._score_faithfulness(
                item["expected_answer"], query_result.answer
            ),
        )

    def _check_source_hit(self, expected_source: str, chunks: list[Chunk]) -> bool:
        return any(expected_source in c.metadata.get("source", "") for c in chunks)

    def _score_faithfulness(self, expected: str, actual: str) -> float:
        prompt = FAITHFULNESS_PROMPT.format(expected=expected, actual=actual)
        raw = self._judge_llm.generate(prompt).strip()
        try:
            return max(0.0, min(1.0, float(raw)))
        except ValueError:
            return 0.0

    def _calc_recall(self, results: list[EvalResult]) -> float:
        if not results:
            return 0.0
        return sum(1 for r in results if r.source_hit) / len(results)

    def _calc_avg_faithfulness(self, results: list[EvalResult]) -> float:
        if not results:
            return 0.0
        return sum(r.faithfulness_score for r in results) / len(results)

    def execute(self) -> EvalSummary:
        dataset = self._load_dataset()
        results: list[EvalResult] = []

        for i, item in enumerate(dataset):
            result = self._evaluate_item(item)
            results.append(result)
            print(
                f"  [{i + 1}/{len(dataset)}] recall={'HIT' if result.source_hit else 'MISS'} "
                f"faith={result.faithfulness_score:.1f} | {result.question[:60]}"
            )

        scores = {name: calc(results) for name, calc in self._metrics.items()}

        return EvalSummary(
            results=results,
            total_questions=len(results),
            scores=scores,
        )
