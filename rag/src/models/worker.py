from typing import Any

from pydantic import BaseModel

from src.enums import JobState


class JobStatusReport(BaseModel):
    """
    Observable state of one ingestion job.
    """

    job_id: str
    status: JobState
    result: Any | None = None
    error: str | None = None
