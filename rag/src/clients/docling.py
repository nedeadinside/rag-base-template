import asyncio
from pathlib import Path

import httpx

from errors import DoclingError
from models import Chunk, DoclingConfig


class DoclingClient:
    """
    Thin client over the docling-serve chunking API.
    """

    def __init__(self, config: DoclingConfig, client: httpx.AsyncClient) -> None:
        """
        Store the docling settings and the shared HTTP client.

        :param config: Docling service settings.
        :param client: Shared async HTTP client.
        """
        self._config = config
        self._client = client

    async def chunk(self, file_path: str, *, max_tokens: int) -> list[Chunk]:
        """
        Convert and chunk a document through docling-serve.

        :param file_path: Path to the spooled document on disk.
        :param max_tokens: Target chunk size in tokens, forwarded to the chunker.
        :raises DoclingError: If the service is unreachable or returns an unusable response.
        :return: The parsed chunks.
        """
        path = Path(file_path)
        url = f"{self._config.url}/v1/chunk/{self._config.chunker}/file"
        headers = {}
        if self._config.api_key is not None:
            headers["X-Api-Key"] = self._config.api_key.get_secret_value()
        data = {"chunking_max_tokens": str(max_tokens), "chunking_merge_peers": "true"}
        try:
            content = await asyncio.to_thread(path.read_bytes)
            response = await self._client.post(
                url,
                headers=headers,
                data=data,
                files={"files": (path.name, content)},
                timeout=self._config.timeout_sec,
            )
            response.raise_for_status()
            payload = response.json()
            return [Chunk.model_validate(chunk) for chunk in payload["chunks"]]
        except httpx.HTTPStatusError as e:
            raise DoclingError(f"chunk request failed: HTTP {e.response.status_code}") from e
        except httpx.HTTPError as e:
            raise DoclingError(f"chunk request failed: {type(e).__name__}") from e
        except (KeyError, ValueError) as e:
            raise DoclingError(f"unparseable chunk response: {type(e).__name__}") from e
        except OSError as e:
            raise DoclingError(f"reading document failed: {e.strerror}") from e
