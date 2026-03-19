from rag_agent.domain.ports import ImageDescriber


class FakeImageDescriber:
    def __init__(self, response: str = "a test image") -> None:
        self._response = response
        self.call_count: int = 0

    def describe(self, image_bytes: bytes) -> str:
        self.call_count += 1
        return self._response


# Protocol conformance check
_: ImageDescriber = FakeImageDescriber()
