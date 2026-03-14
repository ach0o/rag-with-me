import math
from collections import Counter

from rag_agent.domain.models import Chunk
from rag_agent.domain.ports import ChunkRepository


class BM25SparseRetriever:
    def __init__(
        self,
        chunk_repository: ChunkRepository,
        top_k: int = 5,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._chunk_repository = chunk_repository
        self._top_k = top_k
        self._k1 = k1
        self._b = b
        self._chunks: list[Chunk] = []
        self._doc_freqs: Counter[str] = Counter()
        self._doc_lens: list[int] = []
        self._avg_dl: float = 0.0
        self._token_counts: list[Counter[str]] = []
        self._build_index()

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def _build_index(self) -> None:
        self._chunks = self._chunk_repository.get_all()
        if not self._chunks:
            return

        self._token_counts = []
        self._doc_lens = []
        self._doc_freqs = Counter()

        for chunk in self._chunks:
            tokens = self._tokenize(chunk.content)
            token_count = Counter(tokens)
            self._token_counts.append(token_count)
            self._doc_lens.append(len(tokens))
            for token in set(tokens):
                self._doc_freqs[token] += 1

        self._avg_dl = sum(self._doc_lens) / len(self._doc_lens)

    def _score(self, query_tokens: list[str], doc_idx: int) -> float:
        score = 0.0
        n = len(self._chunks)
        dl = self._doc_lens[doc_idx]
        token_count = self._token_counts[doc_idx]

        for token in query_tokens:
            df = self._doc_freqs.get(token, 0)
            if df == 0:
                continue
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
            tf = token_count.get(token, 0)
            numerator = tf * (self._k1 + 1)
            denominator = tf + self._k1 * (1 - self._b + self._b * dl / self._avg_dl)
            score += idf * numerator / denominator
        return score

    def retrieve(self, query: str) -> list[Chunk]:
        if not self._chunks:
            return []

        query_tokens = self._tokenize(query)
        scores = [
            (
                idx,
                self._score(query_tokens, idx),
            )
            for idx in range(len(self._chunks))
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [self._chunks[idx] for idx, _ in scores[: self._top_k]]
