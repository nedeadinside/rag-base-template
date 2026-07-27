from typing import Any

from pydantic import BaseModel
from qdrant_client import models

from clients.embedder import EmbedderClient
from clients.qdrant import QdrantClient
from clients.reranker import RerankerClient


class RetrievedChunk(BaseModel):
    """
    A single reranked chunk returned from retrieval.
    """

    text: str
    score: float
    document_id: str | None
    metadata: dict[str, Any]


async def run(
    query: str,
    collection: str,
    *,
    query_filter: models.Filter | None = None,
    query_prefix: str | None = None,
    embedder: EmbedderClient,
    qdrant: QdrantClient,
    reranker: RerankerClient,
    top_k: int,
    top_n: int,
) -> list[RetrievedChunk]:
    """
    Retrieve chunks for a query: embed it, search the vector store, and rerank the candidates.

    :param query: The natural-language query.
    :param collection: Target Qdrant collection.
    :param query_filter: Optional Qdrant filter applied to the search.
    :param query_prefix: Optional prefix prepended to the query before embedding.
    :param embedder: Client for the embedding service.
    :param qdrant: Client for the vector store.
    :param reranker: Client for the reranking service.
    :param top_k: Number of candidates to pull from Qdrant.
    :param top_n: Number of chunks to keep after reranking.
    :raises RagError: If embedding, search, or reranking fails.
    :return: The reranked chunks, best first.
    """
    qvec = (await embedder.embed([query], prefix=query_prefix))[0]
    hits = await qdrant.search(collection, qvec, limit=top_k, query_filter=query_filter)
    if not hits:
        return []
    payloads = [hit.payload or {} for hit in hits]
    docs = [payload.get("text", "") for payload in payloads]
    ranked = await reranker.rerank(query, docs, top_n=top_n)
    return [
        RetrievedChunk(
            text=docs[index],
            score=score,
            document_id=payloads[index].get("document_id"),
            metadata=payloads[index],
        )
        for index, score in ranked
    ]
