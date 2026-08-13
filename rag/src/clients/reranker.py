import httpx

from src.errors import RerankerError
from src.models import RerankerConfig


class RerankerClient:
    """
    Thin client over a Jina-style reranking endpoint.
    """

    def __init__(self, config: RerankerConfig, client: httpx.AsyncClient) -> None:
        """
        Store the reranker settings and the shared HTTP client.

        :param config: Reranker service settings.
        :param client: Shared async HTTP client.
        """
        self._config = config
        self._client = client

    async def rerank(self, query: str, documents: list[str], *, top_n: int) -> list[tuple[int, float]]:
        """
        Rerank documents against a query, keeping the top matches.

        :param query: The query to score documents against.
        :param documents: Candidate documents, in caller order.
        :param top_n: Maximum number of ranked results to keep.
        :raises RerankerError: If the service is unreachable or returns an unusable response.
        :return: Pairs of original document index and relevance score, sorted by score descending.
        """
        headers = {}
        if self._config.api_key is not None:
            headers["Authorization"] = f"Bearer {self._config.api_key.get_secret_value()}"
        try:
            response = await self._client.post(
                f"{self._config.url}/rerank",
                headers=headers,
                json={
                    "model": self._config.model,
                    "query": query,
                    "documents": documents,
                    "top_n": top_n,
                },
                timeout=self._config.timeout_sec,
            )
            response.raise_for_status()
            results = response.json()["results"]
            ranked = [(item["index"], item["relevance_score"]) for item in results]
        except httpx.HTTPStatusError as e:
            raise RerankerError(f"rerank request failed: HTTP {e.response.status_code}") from e
        except httpx.HTTPError as e:
            raise RerankerError(f"rerank request failed: {type(e).__name__}") from e
        except (KeyError, ValueError, TypeError) as e:
            raise RerankerError(f"unparseable rerank response: {type(e).__name__}") from e
        ranked.sort(key=lambda pair: pair[1], reverse=True)
        return ranked[:top_n]
