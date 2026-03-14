import os

from openai import OpenAI


class AzureOpenAIEmbedder:
    def __init__(
        self,
        model: str = "text-embedding-3-small",
    ) -> None:
        # TODO: add settings class, load .env from entry and set that class
        self._client = OpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        self._model = model
        self.last_usage: dict | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            input=texts,
            model=self._model,
        )
        self.last_usage = {
            "model": self._model,
            "total_tokens": response.usage.total_tokens,
            "num_texts": len(texts),
            "embedding_dim": len(response.data[0].embedding) if response.data else 0,
        }
        return [item.embedding for item in response.data]
