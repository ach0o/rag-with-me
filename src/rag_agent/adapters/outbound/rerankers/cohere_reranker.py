import os

import cohere

from rag_agent.domain.models import Chunk


class CohereReranker:
    def __init__(self, model: str = "rerank-v4.0-fast", top_k: int = 3) -> None:
        self._client = cohere.Client(
            api_key=os.environ["COHERE_API_KEY"],
            base_url=os.environ["COHERE_ENDPOINT"],
        )
        self._model = model
        self._top_k = top_k

    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return []

        response = self._client.rerank(
            model=self._model,
            query=query,
            documents=[{"text": chunk.content} for chunk in chunks],
            top_n=self._top_k,
        )
        return [chunks[result.index] for result in response.results]
