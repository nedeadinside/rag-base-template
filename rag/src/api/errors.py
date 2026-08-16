import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.errors import ResourceError, ResourceTooLargeError, UpstreamError

logger = logging.getLogger(__name__)


async def on_resource_too_large(request: Request, exc: ResourceTooLargeError) -> JSONResponse:
    """
    Handle an input that exceeds a configured cap.

    :param request: The request that triggered the error.
    :param exc: The raised exception.
    :return: A 413 response with the error detail.
    """
    logger.warning("Resource too large on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=413, content={"detail": str(exc)})


async def on_resource_error(request: Request, exc: ResourceError) -> JSONResponse:
    """
    Handle an input that cannot be accepted.

    :param request: The request that triggered the error.
    :param exc: The raised exception.
    :return: A 400 response with the error detail.
    """
    logger.warning("Bad request on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=400, content={"detail": str(exc)})


async def on_upstream_error(request: Request, exc: UpstreamError) -> JSONResponse:
    """
    Handle a failure of an upstream service.

    :param request: The request that triggered the error.
    :param exc: The raised exception.
    :return: A 502 response without upstream details.
    """
    logger.error("Upstream failure on %s", request.url.path, exc_info=exc)
    return JSONResponse(status_code=502, content={"detail": "Upstream service unavailable"})


async def on_unhandled(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle any exception not covered by a more specific handler.

    :param request: The request that triggered the error.
    :param exc: The raised exception.
    :return: A 500 response without internal details.
    """
    logger.error("Unhandled error on %s", request.url.path, exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


def register(app: FastAPI) -> None:
    """
    Register exception handlers on the FastAPI application.

    :param app: The application to register handlers on.
    """
    app.add_exception_handler(ResourceTooLargeError, on_resource_too_large)
    app.add_exception_handler(ResourceError, on_resource_error)
    app.add_exception_handler(UpstreamError, on_upstream_error)
    app.add_exception_handler(Exception, on_unhandled)
