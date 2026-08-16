import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import httpx
from arq import ArqRedis, create_pool
from arq.connections import RedisSettings
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate

from src.api.state import AppState
from src.clients import EmbedderClient, LLMClient, QdrantClient, RerankerClient
from src.config import load_config, load_prompts
from src.models import AppConfig
from src.pipeline import Pipeline

logger = logging.getLogger(__name__)


def build_state(
    config: AppConfig,
    prompts: dict[str, ChatPromptTemplate],
    http: httpx.AsyncClient,
    redis: ArqRedis,
) -> AppState:
    """
    Build the query-side dependency graph for the API process.

    :param config: The service settings.
    :param prompts: Prompt templates keyed by prompt name.
    :param http: Shared async HTTP client for upstream services.
    :param redis: Connected Arq Redis pool for enqueuing jobs.
    :return: The assembled application state.
    """
    embedder = EmbedderClient(config.embedder, http)
    qdrant = QdrantClient(config.qdrant)
    reranker = RerankerClient(config.reranker, http)
    llm = LLMClient(config.llm, http)

    pipeline = Pipeline(
        config,
        prompts,
        embedder=embedder,
        qdrant=qdrant,
        reranker=reranker,
        llm=llm,
    )

    return AppState(
        config=config,
        http=http,
        qdrant=qdrant,
        pipeline=pipeline,
        redis=redis,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """
    Manage the lifetime of the application state for the FastAPI app.

    :param app: The application whose state is managed.
    :return: An async generator yielding once the state is ready.
    """
    config = load_config()
    prompts = load_prompts()

    http = httpx.AsyncClient()
    redis = await create_pool(RedisSettings.from_dsn(config.queue.redis_url))

    app.state.deps = build_state(config, prompts, http, redis)
    logger.info("Application started.")
    try:
        yield
    finally:
        await app.state.deps.qdrant.close()
        await http.aclose()
        await redis.aclose()
        logger.info("Application has finished working.")
