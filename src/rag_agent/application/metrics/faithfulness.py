from rag_agent.domain.models import Chunk
from rag_agent.domain.ports import LLMProvider

FAITHFULNESS_PROMPT = """Rate how well the actual answer matches the expected answer on a scale of 0.0 to
1.0.

- 1.0 = covers the same key points, factually consistent
- 0.5 = partially correct, missing important details
- 0.0 = wrong or completely unrelated

Expected answer: {expected}

Actual answer: {actual}

Reply with only a number between 0.0 and 1.0, nothing else.

Score:"""


class FaithfulnessMetric:
    name = "answer/faithfulness"

    def __init__(self, judge_llm: LLMProvider) -> None:
        self._judge_llm = judge_llm

    def score_item(
        self,
        expected_source: str,
        expected_answer: str,
        actual_answer: str,
        chunks: list[Chunk],
    ) -> float:
        prompt = FAITHFULNESS_PROMPT.format(
            expected=expected_answer, actual=actual_answer
        )
        raw = self._judge_llm.generate(prompt).strip()
        try:
            return max(0.0, min(1.0, float(raw)))
        except ValueError:
            return 0.0

    def aggregate(self, results: list) -> float:
        scores = [r.scores.get(self.name, 0.0) for r in results]
        return sum(scores) / len(scores) if scores else 0.0
