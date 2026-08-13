import asyncio
import time
from pathlib import Path

import httpx

from src.errors import DoclingError
from src.models import Chunk, DoclingConfig


class DoclingClient:
    """
    Client over the docling-serve chunking API.
    """

    def __init__(self, config: DoclingConfig, tokenizer: str, client: httpx.AsyncClient) -> None:
        """
        Store the docling settings, the embedder tokenizer, and the shared HTTP client.

        :param config: Docling service settings.
        :param tokenizer: Tokenizer name forwarded to the hybrid chunker.
        :param client: Shared async HTTP client.
        """
        self._config = config
        self._tokenizer = tokenizer
        self._client = client

    async def chunk(self, file_path: str) -> list[Chunk]:
        """
        Convert and chunk a document through docling-serve.

        :param file_path: Path to the spooled document on disk.
        :raises DoclingError: If the service is unreachable, times out, or returns an unusable response.
        :return: The parsed chunks.
        """
        path = Path(file_path)
        timeout_sec = self._config.timeout_sec
        deadline = time.monotonic() + timeout_sec if timeout_sec is not None else None

        task_id = await self._submit(path, deadline=deadline)
        await self._wait(task_id, deadline=deadline)
        return await self._fetch_result(task_id)

    def _headers(self) -> dict[str, str]:
        """
        Build the request headers carrying the optional api key.

        :return: Headers for a docling-serve request.
        """
        headers = {}
        if self._config.api_key is not None:
            headers["X-Api-Key"] = self._config.api_key.get_secret_value()
        return headers

    async def _submit(self, path: Path, *, deadline: float | None) -> str:
        """
        Submit a document for asynchronous conversion and chunking.

        :param path: Path to the spooled document on disk.
        :param deadline: Monotonic deadline for the whole chunk operation, or None for no deadline.
        :raises DoclingError: If the file cannot be read, the service is unreachable, or the response is unusable.
        :return: The docling task id.
        """
        url = f"{self._config.url}/v1/chunk/{self._config.chunker.value}/file/async"
        try:
            content = await asyncio.to_thread(path.read_bytes)
            response = await self._client.post(
                url,
                headers=self._headers(),
                data=self._config.form(tokenizer=self._tokenizer),
                files={"files": (path.name, content)},
                timeout=self._remaining(deadline),
            )
            response.raise_for_status()
            return str(response.json()["task_id"])
        except httpx.HTTPStatusError as e:
            raise DoclingError(f"submit request failed: HTTP {e.response.status_code}") from e
        except httpx.HTTPError as e:
            raise DoclingError(f"submit request failed: {type(e).__name__}") from e
        except (KeyError, ValueError) as e:
            raise DoclingError(f"unparseable submit response: {type(e).__name__}") from e
        except OSError as e:
            raise DoclingError(f"reading document failed: {e.strerror}") from e

    async def _wait(self, task_id: str, *, deadline: float | None) -> None:
        """
        Poll a docling task until it reaches a terminal, usable state.

        :param task_id: The docling task id to poll.
        :param deadline: Monotonic deadline for the whole chunk operation, or None for no deadline.
        :raises DoclingError: If the service is unreachable, the response is unusable, the task fails, or the
            deadline is exhausted before the task succeeds.
        """
        url = f"{self._config.url}/v1/status/poll/{task_id}"
        while True:
            try:
                response = await self._client.get(url, headers=self._headers(), timeout=self._config.poll_timeout_sec)
                response.raise_for_status()
                task = response.json()
            except httpx.HTTPStatusError as e:
                raise DoclingError(f"status poll failed: HTTP {e.response.status_code}") from e
            except httpx.HTTPError as e:
                raise DoclingError(f"status poll failed: {type(e).__name__}") from e
            except ValueError as e:
                raise DoclingError(f"unparseable status response: {type(e).__name__}") from e

            status = task.get("task_status") if isinstance(task, dict) else None
            if status in ("success", "partial_success"):
                return
            if status not in ("pending", "started"):
                reason = task.get("error_message") or "no reason reported"
                raise DoclingError(f"task ended with status {status!r}: {reason}")

            if deadline is None:
                await asyncio.sleep(self._config.poll_interval_sec)
            else:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise DoclingError(
                        f"task did not finish within {self._config.timeout_sec}s (last status: {status})"
                    )
                await asyncio.sleep(min(self._config.poll_interval_sec, left))

    async def _fetch_result(self, task_id: str) -> list[Chunk]:
        """
        Fetch the chunks produced by a completed docling task.

        :param task_id: The docling task id to fetch results for.
        :raises DoclingError: If the service is unreachable or returns an unusable response.
        :return: The parsed chunks.
        """
        url = f"{self._config.url}/v1/result/{task_id}"
        try:
            response = await self._client.get(url, headers=self._headers(), timeout=self._config.result_timeout_sec)
            response.raise_for_status()
            raw_chunks = response.json()["chunks"]
            return [Chunk.model_validate({**item, "headings": item.get("headings") or []}) for item in raw_chunks]
        except httpx.HTTPStatusError as e:
            raise DoclingError(f"result fetch failed: HTTP {e.response.status_code}") from e
        except httpx.HTTPError as e:
            raise DoclingError(f"result fetch failed: {type(e).__name__}") from e
        except (KeyError, TypeError, ValueError) as e:
            raise DoclingError(f"unparseable result response: {type(e).__name__}") from e

    def _remaining(self, deadline: float | None) -> float | None:
        """
        Compute the time budget left before the overall chunk deadline.

        :param deadline: Monotonic deadline for the whole chunk operation, or None for no deadline.
        :raises DoclingError: If the deadline has already passed.
        :return: Seconds remaining, or None if there is no deadline.
        """
        if deadline is None:
            return None
        left = deadline - time.monotonic()
        if left <= 0:
            raise DoclingError("timeout budget exhausted before the request was sent")
        return left
