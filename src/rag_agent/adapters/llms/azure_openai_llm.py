import os

from openai import OpenAI


class AzureOpenAILLM:
    def __init__(
        self,
        model: str = "gpt-5-mini",
        temperature: float = 1,
        max_tokens: int = 1024,
    ) -> None:
        self._client = OpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self.last_usage: dict | None = None

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
            max_completion_tokens=self._max_tokens,
        )
        self.last_usage = {
            "model": self._model,
            "total_tokens": response.usage.total_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }
        return response.choices[0].message.content or ""
