from rag_agent.domain.models import Chunk


class RecallMetric:
    name = "retrieval/recall_at_k"

    def score_item(
        self,
        expected_source: str,
        expected_answer: str,
        actual_answer: str,
        chunks: list[Chunk],
    ) -> float:
        return (
            1.0
            if any(expected_source in c.metadata.get("source", "") for c in chunks)
            else 0.0
        )

    def aggregate(self, results: list) -> float:
        scores = [r.scores.get(self.name, 0.0) for r in results]
        return sum(scores) / len(scores) if scores else 0.0
