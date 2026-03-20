from typing import Protocol

from rag_agent.domain.models import Chunk


class Metric(Protocol):
    name: str

    def score_item(
        self,
        expected_source: str,
        expected_answer: str,
        actual_answer: str,
        chunks: list[Chunk],
    ) -> float | None: ...

    def aggregate(self, results: list) -> float: ...
