from typing import Any, TypedDict

import httpx
from pydantic import BaseModel

from clients.docling import DoclingClient
from clients.embedder import EmbedderClient
from clients.qdrant import QdrantClient
from clients.webhook import WebhookClient
from enums import JobState

from .config import AppConfig


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


class JobStatusReport(BaseModel):
    """
    Observable state of one ingestion job.
    """

    job_id: str
    status: JobState
    result: Any | None = None
    error: str | None = None
