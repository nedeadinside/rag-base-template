from typing import TYPE_CHECKING, Any

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.common.client_exceptions import QdrantException
from qdrant_client.http.exceptions import ApiException

from errors import QdrantError

if TYPE_CHECKING:
    from config.models import QdrantConfig


class QdrantClient:
    """
    Thin wrapper over the async Qdrant SDK client.
    """

    def __init__(self, config: "QdrantConfig") -> None:
        """
        Build the underlying async Qdrant client from the settings.

        :param config: Qdrant vector store settings.
        """
        self._config = config
        api_key = config.api_key.get_secret_value() if config.api_key is not None else None
        self._client = AsyncQdrantClient(url=config.url, api_key=api_key, timeout=config.timeout_sec)

    async def close(self) -> None:
        """
        Close the underlying client and its connections.
        """
        await self._client.close()

    async def ensure_collection(self, collection: str, *, vector_size: int) -> None:
        """
        Ensure the collection exists with the expected schema, creating it when missing.

        :param collection: Name of the target collection.
        :param vector_size: Dimensionality the collection vectors must have.
        :raises QdrantError: If the store is unreachable, the schema does not match, or the request fails.
        """
        try:
            if await self._client.collection_exists(collection):
                info = await self._client.get_collection(collection)
                vectors = info.config.params.vectors
                if not isinstance(vectors, models.VectorParams):
                    raise QdrantError(f"collection {collection} does not use a single unnamed vector")
                if vectors.size != vector_size:
                    raise QdrantError(f"collection {collection} vector size {vectors.size} != {vector_size}")
                return
            try:
                await self._client.create_collection(
                    collection,
                    vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
                )
            except (ApiException, QdrantException):
                if not await self._client.collection_exists(collection):
                    raise
        except (ApiException, QdrantException) as e:
            raise QdrantError(f"ensure collection failed: {e}") from e

    async def upsert(self, collection: str, points: list[dict[str, Any]]) -> None:
        """
        Upsert points into the collection in batches, waiting for each batch to persist.

        :param collection: Name of the target collection.
        :param points: Point dicts with id, vector, and payload.
        :raises QdrantError: If the store is unreachable or the request fails.
        """
        batch_size = self._config.upsert_batch_size
        try:
            for start in range(0, len(points), batch_size):
                batch = points[start : start + batch_size]
                await self._client.upsert(
                    collection,
                    points=[models.PointStruct(**point) for point in batch],
                    wait=True,
                )
        except (ApiException, QdrantException) as e:
            raise QdrantError(f"upsert failed: {e}") from e

    async def search(
        self,
        collection: str,
        vector: list[float],
        *,
        limit: int,
        query_filter: "models.Filter | None" = None,
    ) -> list[models.ScoredPoint]:
        """
        Search the collection for the points nearest to a query vector.

        :param collection: Name of the target collection.
        :param vector: Query embedding vector.
        :param limit: Maximum number of hits to return.
        :param query_filter: Optional Qdrant filter applied to the search.
        :raises QdrantError: If the store is unreachable or the request fails.
        :return: Scored points with id, score, and payload.
        """
        try:
            response = await self._client.query_points(
                collection,
                query=vector,
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
            )
        except (ApiException, QdrantException) as e:
            raise QdrantError(f"search failed: {e}") from e
        return response.points
