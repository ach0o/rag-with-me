from rag_agent.domain.models import Chunk


class MRRMetric:
    name = "retrieval/mrr"

    def score_item(
        self,
        expected_source: str,
        expected_answer: str,
        actual_answer: str,
        chunks: list[Chunk],
    ) -> float:
        for i, chunk in enumerate(chunks):
            if expected_source in chunk.metadata.get("source", ""):
                return 1.0 / (i + 1)
        return 0.0

    def aggregate(self, results: list) -> float:
        scores = [r.scores.get(self.name, 0.0) for r in results]
        return sum(scores) / len(scores) if scores else 0.0
