import math

from rag_agent.domain.models import Chunk
from rag_agent.domain.ports import Embedder


class SemanticSimilarityMetric:
    name = "answer/semantic_similarity"

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    def score_item(
        self,
        expected_source: str,
        expected_answer: str,
        actual_answer: str,
        chunks: list[Chunk],
    ) -> float:
        vecs = self._embedder.embed([expected_answer, actual_answer])
        return self._cosine_similarity(vecs[0], vecs[1])

    def aggregate(self, results: list) -> float:
        scores = [r.scores.get(self.name, 0.0) for r in results]
        return sum(scores) / len(scores) if scores else 0.0

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
