import uuid
from pathlib import Path
from typing import Any

from qdrant_client import models

from clients import DoclingClient, EmbedderClient, QdrantClient
from errors import EmptyDocumentError
from models import ChunkPayload, IngestResult


async def run(
    file_path: str,
    collection: str,
    metadata: dict[str, Any],
    *,
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
    :param docling: Client for the docling chunking service.
    :param embedder: Client for the embedding service.
    :param qdrant: Client for the vector store.
    :param chunk_size: Target chunk size in tokens, forwarded to docling.
    :param passage_prefix: Optional prefix prepended to every chunk before embedding.
    :raises RagError: If the document yields no text, or chunking, embedding, or the upsert fails.
    :return: The ingestion result with the derived document id and produced chunk count.
    """
    chunks = await docling.chunk(file_path, max_tokens=chunk_size)
    if not chunks:
        raise EmptyDocumentError(f"no text extracted from {Path(file_path).name}")
    document_id = str(uuid.uuid5(uuid.NAMESPACE_OID, "\n".join(chunk.text for chunk in chunks)))
    vectors = await embedder.embed([chunk.text for chunk in chunks], prefix=passage_prefix)
    await qdrant.ensure_collection(collection, vector_size=len(vectors[0]))
    points = [
        models.PointStruct(
            id=str(uuid.uuid5(uuid.UUID(document_id), str(i))),
            vector=vector,
            payload=ChunkPayload(
                document_id=document_id,
                text=chunk.text,
                headings=chunk.headings,
                metadata=metadata,
            ).model_dump(),
        )
        for i, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
    ]
    await qdrant.upsert(collection, points)
    return IngestResult(document_id=document_id, collection=collection, chunks=len(points))
