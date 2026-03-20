from rag_agent.domain.models import Chunk


class PrecisionMetric:
    name = "retrieval/precision_at_k"

    def score_item(
        self,
        expected_source: str,
        expected_answer: str,
        actual_answer: str,
        chunks: list[Chunk],
    ) -> float:
        if not chunks:
            return 0.0
        hits = sum(1 for c in chunks if expected_source in c.metadata.get("source", ""))
        return hits / len(chunks)

    def aggregate(self, results: list) -> float:
        scores = [r.scores.get(self.name, 0.0) for r in results]
        return sum(scores) / len(scores) if scores else 0.0
