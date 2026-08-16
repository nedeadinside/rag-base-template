from typing import Any

from pydantic import BaseModel, Field


class ContextChunk(BaseModel):
    """
    HTTP projection of a retrieved chunk, returned as part of an answer.
    """

    text: str
    score: float
    document_id: str
    metadata: dict[str, Any]


class AskRequest(BaseModel):
    """
    Request body for asking a question against a collection.
    """

    query: str = Field(min_length=1, max_length=8192, description="User question to answer.")
    collection: str = Field(description="Name of the collection to search.")
    include_context: bool = Field(default=False, description="Whether to return the chunks the answer is grounded on.")


class AskResponse(BaseModel):
    """
    Response body for an answered question.
    """

    answer: str
    context: list[ContextChunk] | None = None


class IngestAccepted(BaseModel):
    """
    Response body confirming an ingestion job has been accepted.
    """

    job_id: str


class HealthReport(BaseModel):
    """
    Response body reporting the health of the service and its dependencies.
    """

    status: str
    checks: dict[str, bool]
