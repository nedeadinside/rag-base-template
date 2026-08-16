from typing import Any

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.common.client_exceptions import QdrantException
from qdrant_client.http.exceptions import ApiException, UnexpectedResponse

from src.errors import QdrantError
from src.models import QdrantConfig


def build_metadata_filter(metadata: dict[str, Any] | None) -> models.Filter | None:
    """
    Build a filter matching points whose ingested metadata equals the given values.

    :param metadata: Metadata keys and the values they must match, or None for no filtering.
    :return: A filter requiring every key to match, or None when no metadata was given.
    """
    if not metadata:
        return None
    return models.Filter(
        must=[
            models.FieldCondition(
                key=f"metadata.{key}",
                match=models.MatchAny(any=value) if isinstance(value, list) else models.MatchValue(value=value),
            )
            for key, value in metadata.items()
        ]
    )


class QdrantClient:
    """
    Thin wrapper over the async Qdrant SDK client.
    """

    def __init__(self, config: QdrantConfig) -> None:
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

    async def ping(self) -> bool:
        """
        Check whether the Qdrant store is reachable.

        :return: True if the store responded, False on any connectivity or API error.
        """
        try:
            await self._client.get_collections()
        except (UnexpectedResponse, ApiException, QdrantException):
            return False
        return True

    async def ensure_collection(self, collection: str, *, vector_size: int) -> None:
        """
        Ensure the collection exists with the expected schema, creating it when missing.

        :param collection: Name of the target collection.
        :param vector_size: Dimensionality the collection dense vectors must have.
        :raises QdrantError: If the store is unreachable, the schema is incompatible, or the request fails.
        """
        try:
            if await self._client.collection_exists(collection):
                info = await self._client.get_collection(collection)
                vectors = info.config.params.vectors
                if not isinstance(vectors, dict) or self._config.dense_vector not in vectors:
                    raise QdrantError(
                        f"collection {collection} uses an incompatible schema "
                        f"(expected named vector {self._config.dense_vector!r}); reindex it"
                    )
                dense = vectors[self._config.dense_vector]
                if dense.size != vector_size:
                    raise QdrantError(f"collection {collection} vector size {dense.size} != {vector_size}")
                return
            try:
                await self._client.create_collection(
                    collection,
                    vectors_config={
                        self._config.dense_vector: models.VectorParams(
                            size=vector_size, distance=models.Distance.COSINE
                        )
                    },
                    sparse_vectors_config={
                        self._config.sparse_vector: models.SparseVectorParams(modifier=models.Modifier.IDF)
                    },
                )
            except (ApiException, QdrantException):
                if not await self._client.collection_exists(collection):
                    raise
        except UnexpectedResponse as e:
            raise QdrantError(f"ensure collection failed: HTTP {e.status_code}") from e
        except (ApiException, QdrantException) as e:
            raise QdrantError(f"ensure collection failed: {type(e).__name__}") from e

    async def upsert(self, collection: str, points: list[models.PointStruct]) -> None:
        """
        Upsert points into the collection in batches, waiting for each batch to persist.

        :param collection: Name of the target collection.
        :param points: Points to write, with id, vector, and payload.
        :raises QdrantError: If the store is unreachable or the request fails.
        """
        batch_size = self._config.upsert_batch_size
        try:
            for start in range(0, len(points), batch_size):
                await self._client.upsert(
                    collection,
                    points=points[start : start + batch_size],
                    wait=True,
                )
        except UnexpectedResponse as e:
            raise QdrantError(f"upsert failed: HTTP {e.status_code}") from e
        except (ApiException, QdrantException) as e:
            raise QdrantError(f"upsert failed: {type(e).__name__}") from e

    async def delete_document(self, collection: str, document_id: str) -> None:
        """
        Delete every point belonging to a document from the collection.

        :param collection: Name of the target collection.
        :param document_id: Id of the document whose points must be removed.
        :raises QdrantError: If the store is unreachable or the request fails.
        """
        try:
            await self._client.delete(
                collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
                    )
                ),
                wait=True,
            )
        except UnexpectedResponse as e:
            raise QdrantError(f"delete document failed: HTTP {e.status_code}") from e
        except (ApiException, QdrantException) as e:
            raise QdrantError(f"delete document failed: {type(e).__name__}") from e

    async def document_exists(self, collection: str, document_id: str) -> bool:
        """
        Check whether the collection already holds at least one point for a document.

        Uses a filtered scroll capped at a single point, with payload and vectors disabled, so the
        match is exact and the request stops at the first hit instead of counting every point.

        :param collection: Name of the target collection.
        :param document_id: Id of the document to look up.
        :raises QdrantError: If the store is unreachable or the request fails.
        :return: True if the collection holds at least one point for the document, False otherwise.
        """
        try:
            points, _ = await self._client.scroll(
                collection,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
                ),
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
        except UnexpectedResponse as e:
            raise QdrantError(f"document exists check failed: HTTP {e.status_code}") from e
        except (ApiException, QdrantException) as e:
            raise QdrantError(f"document exists check failed: {type(e).__name__}") from e
        return len(points) > 0

    async def search(
        self,
        collection: str,
        vector: list[float],
        query_text: str,
        *,
        limit: int,
        prefetch_limit: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[models.ScoredPoint]:
        """
        Search the collection with dense and BM25 sparse prefetches merged by reciprocal rank fusion.

        :param collection: Name of the target collection.
        :param vector: Query embedding vector for the dense prefetch.
        :param query_text: Original query text, scored server-side by BM25 for the sparse prefetch.
        :param limit: Maximum number of fused hits to return.
        :param prefetch_limit: Maximum number of hits each prefetch retrieves before fusion.
        :param metadata_filter: Ingested metadata the hits must match, restricting both prefetches.
        :raises QdrantError: If the store is unreachable or the request fails.
        :return: Scored points with id, score, and payload.
        """
        query_filter = build_metadata_filter(metadata_filter)
        prefetch = [
            models.Prefetch(
                query=vector,
                using=self._config.dense_vector,
                limit=prefetch_limit,
                filter=query_filter,
            ),
            models.Prefetch(
                query=self._config.bm25.document(query_text),
                using=self._config.sparse_vector,
                limit=prefetch_limit,
                filter=query_filter,
            ),
        ]
        try:
            response = await self._client.query_points(
                collection,
                prefetch=prefetch,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
        except UnexpectedResponse as e:
            raise QdrantError(f"search failed: HTTP {e.status_code}") from e
        except (ApiException, QdrantException) as e:
            raise QdrantError(f"search failed: {type(e).__name__}") from e
        return response.points
