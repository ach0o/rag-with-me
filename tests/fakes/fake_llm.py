from rag_agent.domain.ports import LLMProvider


class FakeLLM:
    def __init__(self, response: str = "fake answer") -> None:
        self._response = response
        self.last_prompt: str | None = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self._response


# Protocol conformance check — ensures FakeLLM satisfies the LLMProvider port.
# If this breaks, the port interface changed and the fake needs updating.
_: LLMProvider = FakeLLM()
