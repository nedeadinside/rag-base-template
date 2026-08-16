import logging
import uuid
from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.setup_logging import request_id_var

logger = logging.getLogger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware that tags each request with a request id for logging and correlation.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Attach a request id to the request context and to the response headers.

        :param request: The incoming request.
        :param call_next: The next handler in the middleware chain.
        :raises Exception: Whatever the downstream handler raised, after logging it.
        :return: The response, carrying the request id in its headers.
        """
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request_id_var.set(request_id)
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("%s - %s failed", request.method, request.url.path)
            raise
        response.headers["X-Request-ID"] = request_id
        return response
