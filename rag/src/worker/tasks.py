import asyncio
import hashlib
import logging
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, TypedDict

import httpx
from arq.connections import RedisSettings
from qdrant_client import models as qdrant_models

from src.clients import DoclingClient, EmbedderClient, QdrantClient, WebhookClient
from src.config import load_config
from src.enums import JobState
from src.errors import EmptyDocumentError, QdrantError
from src.models import AppConfig, ChunkPayload, IngestResult
from src.setup_logging import configure_logging

logger = logging.getLogger(__name__)

_cfg = load_config()


class WorkerContext(TypedDict):
    """
    Partial view of the worker context, covering the keys this app populates and reads.
    """

    http: httpx.AsyncClient
    docling: DoclingClient
    embedder: EmbedderClient
    qdrant: QdrantClient
    webhook: WebhookClient
    cfg: AppConfig
    job_id: str


async def startup(ctx: WorkerContext) -> None:
    """
    Open the shared HTTP client and build the upstream clients once per worker.

    :param ctx: The worker context to populate.
    """
    cfg = load_config()
    configure_logging()
    http = httpx.AsyncClient()
    qdrant = QdrantClient(cfg.qdrant)
    try:
        ctx["http"] = http
        ctx["cfg"] = cfg
        ctx["qdrant"] = qdrant
        ctx["docling"] = DoclingClient(cfg.docling, cfg.embedder.model, http)
        ctx["embedder"] = EmbedderClient(cfg.embedder, http)
        ctx["webhook"] = WebhookClient(cfg.webhook, http)
    except Exception:
        await qdrant.close()
        await http.aclose()
        raise


async def shutdown(ctx: WorkerContext) -> None:
    """
    Close the shared HTTP client.

    :param ctx: The worker context.
    """
    await ctx["qdrant"].close()
    await ctx["http"].aclose()


async def _ingest_document(
    docling: DoclingClient,
    embedder: EmbedderClient,
    qdrant: QdrantClient,
    cfg: AppConfig,
    file_path: str,
    collection: str,
    metadata: dict[str, Any],
) -> IngestResult:
    """
    Ingest one document: chunk it, embed the chunks, and upsert them into the vector store.

    :param docling: Client for the docling chunking service.
    :param embedder: Client for the embedding service.
    :param qdrant: Client for the vector store.
    :param cfg: The service settings.
    :param file_path: Path to the spooled document on disk.
    :param collection: Target Qdrant collection.
    :param metadata: Caller-supplied metadata copied onto every point payload.
    :raises RagError: If the document yields no text, or chunking, embedding, or the upsert fails.
    :return: The ingestion result with the derived document id and produced chunk count.
    """
    chunks = await docling.chunk(file_path)
    if not chunks:
        raise EmptyDocumentError(f"no text extracted from {Path(file_path).name}")
    digest = hashlib.blake2b(digest_size=16)
    for chunk in chunks:
        digest.update(chunk.text.encode())
        digest.update(b"\n")
    document_id = str(uuid.UUID(bytes=digest.digest(), version=5))
    vectors = await embedder.embed(
        [chunk.text for chunk in chunks],
        prefix=cfg.embedder.passage_prefix,
    )
    await qdrant.ensure_collection(collection, vector_size=len(vectors[0]))
    points = [
        qdrant_models.PointStruct(
            id=str(uuid.uuid5(uuid.UUID(document_id), str(i))),
            vector={
                cfg.qdrant.dense_vector: vector,
                cfg.qdrant.sparse_vector: cfg.qdrant.bm25.document(chunk.text),
            },
            payload=ChunkPayload(
                document_id=document_id,
                text=chunk.text,
                headings=chunk.headings,
                metadata=metadata,
            ).model_dump(),
        )
        for i, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
    ]
    try:
        await qdrant.upsert(collection, points)
    except QdrantError:
        try:
            await qdrant.delete_document(collection, document_id)
        except QdrantError as cleanup_error:
            logger.warning("cleanup of document %s failed: %s", document_id, cleanup_error)
        raise
    return IngestResult(document_id=document_id, collection=collection, chunks=len(points))


async def ingest(
    ctx: WorkerContext,
    file_path: str,
    webhook_url: str | None,
    collection: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """
    Run the ingestion pipeline for one document and deliver a webhook callback.

    :param ctx: The worker context holding the shared clients.
    :param file_path: Path to the spooled document on disk.
    :param webhook_url: Optional URL to POST the terminal status to.
    :param collection: Target Qdrant collection.
    :param metadata: Caller-supplied metadata copied onto every point payload.
    :return: The ingestion result, serialized for the ARQ result store.
    """
    job_id = ctx["job_id"]
    webhook = ctx["webhook"]
    try:
        result = await _ingest_document(
            ctx["docling"], ctx["embedder"], ctx["qdrant"], ctx["cfg"], file_path, collection, metadata
        )
    except Exception as e:
        if webhook_url is not None:
            await webhook.notify(webhook_url, {"job_id": job_id, "status": JobState.FAILED, "error": str(e)})
        raise
    finally:
        await asyncio.to_thread(Path(file_path).unlink, missing_ok=True)

    payload = result.model_dump(mode="json")
    if webhook_url is not None:
        await webhook.notify(webhook_url, {"job_id": job_id, "status": JobState.SUCCESS, "result": payload})
    return payload


class WorkerSettings:
    """
    Worker configuration entrypoint for ARQ.
    """

    functions: ClassVar[list[Callable[..., Any]]] = [ingest]
    on_startup: ClassVar[Callable[..., Any]] = startup
    on_shutdown: ClassVar[Callable[..., Any]] = shutdown
    redis_settings: ClassVar[RedisSettings] = RedisSettings.from_dsn(_cfg.queue.redis_url)
    max_jobs: ClassVar[int] = _cfg.queue.concurrency
    job_timeout: ClassVar[int] = _cfg.queue.job_timeout
    allow_abort_jobs: ClassVar[bool] = True
