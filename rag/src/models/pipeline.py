from typing import Any

from pydantic import BaseModel


class IngestResult(BaseModel):
    """
    Outcome of an ingestion run.
    """

    document_id: str
    collection: str
    chunks: int


class RetrievedChunk(BaseModel):
    """
    A single reranked chunk returned from retrieval.
    """

    text: str
    score: float
    document_id: str | None
    metadata: dict[str, Any]
