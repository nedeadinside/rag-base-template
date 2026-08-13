from typing import Any

from pydantic import BaseModel, Field


class ChunkPayload(BaseModel):
    """
    Payload stored on every Qdrant point and read back on retrieval.
    """

    document_id: str
    text: str
    headings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


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
    document_id: str
    metadata: dict[str, Any]


class Answer(BaseModel):
    """
    Generated answer with the chunks it was grounded on.
    """

    text: str
    sources: list[RetrievedChunk]


class VerifierOutput(BaseModel):
    """
    Verdict on whether the retrieved chunks answer the question.
    """

    can_answer: bool = Field(description="True if the fragments contain a direct and complete answer.")
