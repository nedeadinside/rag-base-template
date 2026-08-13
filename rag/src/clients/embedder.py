import asyncio

import httpx

from src.errors import EmbedderError
from src.models import EmbedderConfig


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

    async def _embed_batch(self, batch: list[str], headers: dict[str, str]) -> list[list[float]]:
        """
        Embed one batch of texts in a single request.

        :param batch: The texts to embed, at most one configured batch size worth.
        :param headers: Request headers carrying the optional api key.
        :raises EmbedderError: If the service is unreachable or returns an unusable response.
        :return: One embedding vector per input text, in order.
        """
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
            return [item["embedding"] for item in data]
        except httpx.HTTPStatusError as e:
            raise EmbedderError(f"embed request failed: HTTP {e.response.status_code}") from e
        except httpx.HTTPError as e:
            raise EmbedderError(f"embed request failed: {type(e).__name__}") from e
        except (KeyError, ValueError) as e:
            raise EmbedderError(f"unparseable embed response: {type(e).__name__}") from e

    async def embed(self, texts: list[str], *, prefix: str | None = None) -> list[list[float]]:
        """
        Embed a list of texts, splitting them into batches sent concurrently.

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
        semaphore = asyncio.Semaphore(self._config.max_concurrency)

        async def embed_batch(batch: list[str]) -> list[list[float]]:
            async with semaphore:
                return await self._embed_batch(batch, headers)

        size = self._config.batch_size
        batches = [texts[start : start + size] for start in range(0, len(texts), size)]
        results = await asyncio.gather(*(embed_batch(batch) for batch in batches))
        return [vector for result in results for vector in result]
