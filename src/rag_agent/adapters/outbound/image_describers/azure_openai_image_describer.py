import base64
import os

from openai import OpenAI


class AzureOpenAIImageDescriber:
    def __init__(self, model: str = "gpt-5-mini") -> None:
        self._client = OpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        self._model = model

    def describe(self, image_bytes: bytes) -> str:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image concisely for use as search context. Focus on key information, text content, and diagram meaning.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                }
            ],
            max_completion_tokens=256,
        )
        return response.choices[0].message.content or ""
