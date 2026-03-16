from rag_agent.domain.ports import Embedder


class FakeEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


# Protocol conformance check — ensures FakeEmbedder satisfies the Embedder port.
# If this breaks, the port interface changed and the fake needs updating.
_: Embedder = FakeEmbedder()
