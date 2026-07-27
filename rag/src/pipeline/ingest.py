import uuid
from typing import Any

from pydantic import BaseModel

from clients.docling import DoclingClient
from clients.embedder import EmbedderClient
from clients.qdrant import QdrantClient


class IngestResult(BaseModel):
    """
    Outcome of an ingestion run.
    """

    document_id: str
    collection: str
    chunks: int


async def run(
    file_path: str,
    collection: str,
    metadata: dict[str, Any],
    *,
    document_id: str,
    docling: DoclingClient,
    embedder: EmbedderClient,
    qdrant: QdrantClient,
    chunk_size: int,
    passage_prefix: str | None = None,
) -> IngestResult:
    """
    Ingest one document: chunk it, embed the chunks, and upsert them into the vector store.

    :param file_path: Path to the spooled document on disk.
    :param collection: Target Qdrant collection.
    :param metadata: Caller-supplied metadata copied onto every point payload.
    :param document_id: Document identifier used to derive deterministic point ids.
    :param docling: Client for the docling chunking service.
    :param embedder: Client for the embedding service.
    :param qdrant: Client for the vector store.
    :param chunk_size: Target chunk size in tokens, forwarded to docling.
    :param passage_prefix: Optional prefix prepended to every chunk before embedding.
    :raises RagError: If chunking, embedding, or the upsert fails.
    :return: The ingestion result with the produced chunk count.
    """
    chunks = await docling.chunk(file_path, max_tokens=chunk_size)
    if not chunks:
        return IngestResult(document_id=document_id, collection=collection, chunks=0)
    vectors = await embedder.embed([chunk.text for chunk in chunks], prefix=passage_prefix)
    await qdrant.ensure_collection(collection, vector_size=len(vectors[0]))
    points = [
        {
            "id": str(uuid.uuid5(uuid.UUID(document_id), str(i))),
            "vector": vector,
            "payload": {
                **metadata,
                "document_id": document_id,
                "text": chunk.text,
                "headings": chunk.headings,
            },
        }
        for i, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
    ]
    await qdrant.upsert(collection, points)
    return IngestResult(document_id=document_id, collection=collection, chunks=len(points))
