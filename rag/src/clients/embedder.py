import httpx

from errors import EmbedderError
from models.config import EmbedderConfig


class EmbedderClient:
    """
    Thin client over an OpenAI-style embeddings endpoint.
    """

    def __init__(self, config: EmbedderConfig, client: httpx.AsyncClient) -> None:
        """
        Store the embedder settings and the shared HTTP client.

        :param config: Embedder service settings.
        :param client: Shared async HTTP client.
        """
        self._config = config
        self._client = client

    async def embed(self, texts: list[str], *, prefix: str | None = None) -> list[list[float]]:
        """
        Embed a list of texts, batching by the configured batch size.

        :param texts: The texts to embed.
        :param prefix: Optional prefix prepended to every text before embedding.
        :raises EmbedderError: If the service is unreachable or returns an unusable response.
        :return: One embedding vector per input text, in order.
        """
        if prefix:
            texts = [f"{prefix}: {text}" for text in texts]
        headers = {}
        if self._config.api_key is not None:
            headers["Authorization"] = f"Bearer {self._config.api_key.get_secret_value()}"
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._config.batch_size):
            batch = texts[start : start + self._config.batch_size]
            try:
                response = await self._client.post(
                    f"{self._config.url}/embeddings",
                    headers=headers,
                    json={"model": self._config.model, "input": batch},
                    timeout=self._config.timeout_sec,
                )
                response.raise_for_status()
                payload = response.json()
                data = sorted(payload["data"], key=lambda item: item["index"])
                vectors.extend(item["embedding"] for item in data)
            except httpx.HTTPError as e:
                raise EmbedderError(f"embed request failed: {e}") from e
            except (KeyError, ValueError) as e:
                raise EmbedderError(f"unparseable embed response: {e}") from e
        return vectors
