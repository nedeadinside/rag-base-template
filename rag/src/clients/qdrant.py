from typing import TYPE_CHECKING, Any

import httpx

from errors import QdrantError

if TYPE_CHECKING:
    from config.models import QdrantConfig


class QdrantClient:
    """
    Thin client over the Qdrant REST API.
    """

    def __init__(self, config: "QdrantConfig", client: httpx.AsyncClient) -> None:
        """
        Store the Qdrant settings and the shared HTTP client.

        :param config: Qdrant vector store settings.
        :param client: Shared async HTTP client.
        """
        self._config = config
        self._client = client

    def _headers(self) -> dict[str, str]:
        """
        Build the request headers, adding the API key when configured.

        :return: The headers to send with every request.
        """
        if self._config.api_key is not None:
            return {"api-key": self._config.api_key.get_secret_value()}
        return {}

    async def ensure_collection(self, collection: str, *, vector_size: int) -> None:
        """
        Ensure the collection exists with the expected schema, creating it when missing.

        :param collection: Name of the target collection.
        :param vector_size: Dimensionality the collection vectors must have.
        :raises QdrantError: If the store is unreachable, the schema does not match, or the request fails.
        """
        base = f"{self._config.url}/collections/{collection}"
        try:
            response = await self._client.get(base, headers=self._headers(), timeout=self._config.timeout_sec)
        except httpx.HTTPError as e:
            raise QdrantError(f"ensure collection failed: {e}") from e

        if response.status_code == httpx.codes.OK:
            try:
                vectors = response.json()["result"]["config"]["params"]["vectors"]
                if not isinstance(vectors, dict) or "size" not in vectors:
                    raise QdrantError(f"collection {collection} does not use a single unnamed vector")
                if vectors["size"] != vector_size:
                    raise QdrantError(f"collection {collection} vector size {vectors['size']} != {vector_size}")
            except (KeyError, TypeError) as e:
                raise QdrantError(f"ensure collection failed: unexpected schema for {collection}: {e}") from e
            return

        if response.status_code == httpx.codes.NOT_FOUND:
            body: dict[str, Any] = {"vectors": {"size": vector_size, "distance": "Cosine"}}
            try:
                create = await self._client.put(
                    base, headers=self._headers(), json=body, timeout=self._config.timeout_sec
                )
                create.raise_for_status()
            except httpx.HTTPError as e:
                raise QdrantError(f"ensure collection failed: {e}") from e
            return

        try:
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise QdrantError(f"ensure collection failed: {e}") from e

    async def upsert(self, collection: str, points: list[dict[str, Any]]) -> None:
        """
        Upsert points into the collection in batches, waiting for each batch to persist.

        :param collection: Name of the target collection.
        :param points: Qdrant point dicts with id, vector, and payload.
        :raises QdrantError: If the store is unreachable or the request fails.
        """
        url = f"{self._config.url}/collections/{collection}/points"
        batch_size = self._config.upsert_batch_size
        try:
            for start in range(0, len(points), batch_size):
                batch = points[start : start + batch_size]
                response = await self._client.put(
                    url,
                    headers=self._headers(),
                    params={"wait": "true"},
                    json={"points": batch},
                    timeout=self._config.timeout_sec,
                )
                response.raise_for_status()
        except httpx.HTTPError as e:
            raise QdrantError(f"upsert failed: {e}") from e
