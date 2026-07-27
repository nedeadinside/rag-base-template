import logging
from typing import Any

import httpx

from models.config import WebhookConfig

_log = logging.getLogger(__name__)


class WebhookClient:
    """
    Deliverer of job-status callbacks.
    """

    def __init__(self, config: WebhookConfig, client: httpx.AsyncClient) -> None:
        """
        Store the webhook settings and the shared HTTP client.

        :param config: Webhook callback settings.
        :param client: Shared async HTTP client.
        """
        self._config = config
        self._client = client

    async def notify(self, url: str, payload: dict[str, Any]) -> None:
        """
        POST a payload to a caller-supplied URL, swallowing delivery failures.

        :param url: The caller endpoint to notify.
        :param payload: The JSON-serializable status payload.
        """
        try:
            response = await self._client.post(url, json=payload, timeout=self._config.timeout_sec)
            response.raise_for_status()
        except (httpx.HTTPError, httpx.InvalidURL) as e:
            _log.warning("webhook delivery to %s failed: %s", url, e)
