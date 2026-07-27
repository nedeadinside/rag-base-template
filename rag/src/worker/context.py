from typing import TypedDict

import httpx

from clients.docling import DoclingClient
from clients.embedder import EmbedderClient
from clients.qdrant import QdrantClient
from clients.webhook import WebhookClient
from config.models import AppConfig


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
