from dataclasses import dataclass

from arq import ArqRedis

from src.clients import QdrantClient
from src.models import AppConfig
from src.pipeline import Pipeline


@dataclass(slots=True)
class AppState:
    """
    Application-wide dependencies shared across request handlers.
    """

    config: AppConfig
    qdrant: QdrantClient
    pipeline: Pipeline
    redis: ArqRedis
