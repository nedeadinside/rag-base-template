import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, TypedDict

import httpx
from arq.connections import RedisSettings

from src.clients import DoclingClient, EmbedderClient, LLMClient, QdrantClient, RerankerClient, WebhookClient
from src.config import load_config, load_prompts
from src.enums import JobState
from src.models import AppConfig
from src.pipeline import Pipeline

_cfg = load_config()


class WorkerContext(TypedDict):
    """
    Partial view of the worker context, covering the keys this app populates and reads.
    """

    http: httpx.AsyncClient
    qdrant: QdrantClient
    webhook: WebhookClient
    pipeline: Pipeline
    cfg: AppConfig
    job_id: str


async def startup(ctx: WorkerContext) -> None:
    """
    Open the shared HTTP client and build the pipeline once per worker.

    :param ctx: The worker context to populate.
    """
    cfg = load_config()
    logging.basicConfig(level=cfg.logging.level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    http = httpx.AsyncClient()
    qdrant = QdrantClient(cfg.qdrant)
    try:
        ctx["http"] = http
        ctx["cfg"] = cfg
        ctx["qdrant"] = qdrant
        ctx["webhook"] = WebhookClient(cfg.webhook, http)
        ctx["pipeline"] = Pipeline(
            cfg,
            load_prompts(),
            docling=DoclingClient(cfg.docling, http),
            embedder=EmbedderClient(cfg.embedder, http),
            qdrant=qdrant,
            reranker=RerankerClient(cfg.reranker, http),
            llm=LLMClient(cfg.llm),
        )
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
        result = await ctx["pipeline"].ingest(file_path, collection, metadata)
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
