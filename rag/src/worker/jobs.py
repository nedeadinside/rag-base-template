from typing import Any

from arq.jobs import Job, JobStatus
from pydantic import BaseModel
from redis.asyncio import Redis

from enums import JobState

_PENDING_STATES = {
    JobStatus.deferred: JobState.QUEUED,
    JobStatus.queued: JobState.QUEUED,
    JobStatus.in_progress: JobState.IN_PROGRESS,
    JobStatus.not_found: JobState.NOT_FOUND,
}


class JobStatusReport(BaseModel):
    """
    Observable state of one ingestion job.
    """

    job_id: str
    status: JobState
    result: Any | None = None
    error: str | None = None


async def get_status(redis: Redis, job_id: str) -> JobStatusReport:
    """
    Read the current state of a job from the queue backend.

    :param redis: Redis connection used by the queue.
    :param job_id: Identifier the job was enqueued under.
    :return: The job state, with the result on success or the error text on failure.
    """
    job = Job(job_id, redis)
    status = await job.status()
    if status is not JobStatus.complete:
        return JobStatusReport(job_id=job_id, status=_PENDING_STATES.get(status, JobState.NOT_FOUND))

    info = await job.result_info()
    if info is None:
        return JobStatusReport(job_id=job_id, status=JobState.NOT_FOUND)
    if info.success:
        return JobStatusReport(job_id=job_id, status=JobState.SUCCESS, result=info.result)
    return JobStatusReport(job_id=job_id, status=JobState.FAILED, error=str(info.result))
