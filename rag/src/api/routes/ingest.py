import asyncio
import json
import logging
import shutil
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, File, Form, UploadFile

from src.api.deps import StateDep
from src.errors import ResourceError, ResourceTooLargeError, UnsupportedFormatError
from src.models import IngestAccepted, JobStatusReport
from src.worker import cancel_job, get_status

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest")

CHUNK_SIZE = 1024 * 1024


@router.post("", status_code=202)
async def submit(
    state: StateDep,
    file: Annotated[UploadFile, File()],
    collection: Annotated[str, Form()],
    webhook_url: Annotated[str | None, Form()] = None,
    metadata: Annotated[str, Form()] = "{}",
) -> IngestAccepted:
    """
    Accept an uploaded document, spool it to disk, and enqueue an ingestion job.

    :param state: Application-wide dependencies.
    :param file: The uploaded document.
    :param collection: Target Qdrant collection.
    :param webhook_url: Optional URL to POST the terminal status to.
    :param metadata: JSON-encoded object copied onto every point payload.
    :raises UnsupportedFormatError: If the file has no name or its extension is not allowed.
    :raises ResourceError: If the metadata field is not a JSON object.
    :raises ResourceTooLargeError: If the upload exceeds the configured byte cap.
    :return: The accepted job id.
    """
    suffix = Path(file.filename).suffix.lower() if file.filename else ""
    if not suffix or suffix not in state.config.ingest.allowed_extensions:
        raise UnsupportedFormatError(f"unsupported file extension: {suffix or '<none>'}")

    max_bytes = state.config.ingest.max_upload_bytes
    if file.size is None or file.size > max_bytes:
        raise ResourceTooLargeError(f"upload exceeds {max_bytes} bytes")

    try:
        metadata_dict = json.loads(metadata)
    except json.JSONDecodeError as e:
        raise ResourceError("metadata must be a JSON object") from e
    if not isinstance(metadata_dict, dict):
        raise ResourceError("metadata must be a JSON object")

    spool = Path(state.config.ingest.spool_dir)
    target = spool / f"{uuid4().hex}{suffix}"

    def _spill() -> None:
        spool.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as fh:
            shutil.copyfileobj(file.file, fh, CHUNK_SIZE)

    await asyncio.to_thread(_spill)

    job = await state.redis.enqueue_job("ingest", str(target), webhook_url, collection, metadata_dict)
    logger.info("Ingest job %s queued for %s", job.job_id, file.filename)
    return IngestAccepted(job_id=job.job_id)


@router.get("/{job_id}")
async def status(job_id: str, state: StateDep) -> JobStatusReport:
    """
    Report the current state of an ingestion job.

    :param job_id: Identifier the job was enqueued under.
    :param state: Application-wide dependencies.
    :return: The job's current status.
    """
    return await get_status(state.redis, job_id)


@router.post("/{job_id}/cancel", status_code=202)
async def cancel(job_id: str, state: StateDep) -> dict[str, str]:
    """
    Cancel a queued or running ingestion job.

    :param job_id: Identifier the job was enqueued under.
    :param state: Application-wide dependencies.
    :return: The job id and the outcome of the cancellation attempt.
    """
    success = await cancel_job(state.redis, job_id)
    return {"job_id": job_id, "status": "canceled" if success else "failed_to_cancel"}
