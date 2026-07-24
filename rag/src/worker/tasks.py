import asyncio
import logging
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import httpx
from arq.connections import RedisSettings

from clients.docling import DoclingClient
from clients.embedder import EmbedderClient
from clients.qdrant import QdrantClient
from clients.webhook import WebhookClient
from config import load_config
from enums import JobState
from pipeline import ingest as ingest_pipeline

from .context import WorkerContext

_cfg = load_config()


async def startup(ctx: WorkerContext) -> None:
    """
    Open the shared HTTP client and build the upstream clients once per worker.

    :param ctx: The worker context to populate.
    """
    cfg = load_config()
    logging.basicConfig(level=cfg.logging.level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    http = httpx.AsyncClient()
    qdrant = QdrantClient(cfg.qdrant, http)
    ctx["http"] = http
    ctx["cfg"] = cfg
    ctx["docling"] = DoclingClient(cfg.docling, http)
    ctx["embedder"] = EmbedderClient(cfg.embedder, http)
    ctx["qdrant"] = qdrant
    ctx["webhook"] = WebhookClient(cfg.webhook, http)


async def shutdown(ctx: WorkerContext) -> None:
    """
    Close the shared HTTP client.

    :param ctx: The worker context.
    """
    await ctx["http"].aclose()


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
    cfg = ctx["cfg"]
    job_id = ctx["job_id"]
    webhook = ctx["webhook"]
    document_id = str(uuid.uuid5(uuid.NAMESPACE_OID, job_id))
    try:
        result = await ingest_pipeline.run(
            file_path,
            collection,
            metadata,
            document_id=document_id,
            docling=ctx["docling"],
            embedder=ctx["embedder"],
            qdrant=ctx["qdrant"],
            chunk_size=cfg.ingest.chunk_size,
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
