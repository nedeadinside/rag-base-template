from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    import httpx

    from clients.docling import DoclingClient
    from clients.embedder import EmbedderClient
    from clients.qdrant import QdrantClient
    from clients.webhook import WebhookClient
    from config.models import AppConfig


class WorkerContext(TypedDict):
    """
    Partial view of the worker context, covering the keys this app populates and reads.

    The runtime injects its own keys too; only what the app touches is declared here.
    """

    http: "httpx.AsyncClient"
    docling: "DoclingClient"
    embedder: "EmbedderClient"
    qdrant: "QdrantClient"
    webhook: "WebhookClient"
    cfg: "AppConfig"
    job_id: str
