import json
from dataclasses import dataclass, field
from pathlib import Path

from rag_agent.application.query import QueryUseCase
from rag_agent.application.query_graph import QueryGraphBuilder
from rag_agent.domain.metrics import Metric


@dataclass
class EvalResult:
    question: str
    expected_answer: str
    expected_source: str
    actual_answer: str
    retrieved_sources: list[str]
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class EvalSummary:
    results: list[EvalResult]
    total_questions: int
    scores: dict[str, float]


class EvaluateUseCase:
    def __init__(
        self,
        query_executor: QueryUseCase | QueryGraphBuilder,
        metrics: list[Metric],
        dataset_path: str = "eval_data/gold_qa.json",
    ) -> None:
        self._query_executor = query_executor
        self._metrics = metrics
        self._dataset_path = Path(dataset_path)

    def _load_dataset(self) -> list[dict[str, str]]:
        with open(self._dataset_path) as f:
            return json.load(f)

    def _evaluate_item(self, item: dict[str, str]) -> EvalResult:
        query_result = self._query_executor.execute(item["question"])

        scores: dict[str, float] = {}
        for metric in self._metrics:
            score = metric.score_item(
                item["expected_source"],
                item["expected_answer"],
                query_result.answer,
                query_result.chunks,
            )
            if score is not None:
                scores[metric.name] = score

        return EvalResult(
            question=item["question"],
            expected_answer=item["expected_answer"],
            expected_source=item["expected_source"],
            actual_answer=query_result.answer,
            retrieved_sources=[
                c.metadata.get("source", "unknown") for c in query_result.chunks
            ],
            scores=scores,
        )

    def execute(self) -> EvalSummary:
        dataset = self._load_dataset()
        results: list[EvalResult] = []

        for i, item in enumerate(dataset):
            result = self._evaluate_item(item)
            results.append(result)
            scores_str = " ".join(f"{k}={v:.2f}" for k, v in result.scores.items())
            print(f"  [{i + 1}/{len(dataset)}] {scores_str} | {result.question[:50]}")

        summary_scores = {m.name: m.aggregate(results) for m in self._metrics}

        return EvalSummary(
            results=results,
            total_questions=len(results),
            scores=summary_scores,
        )
