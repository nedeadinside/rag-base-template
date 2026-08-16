from fastapi import FastAPI

from src.api import errors
from src.api.lifespan import lifespan
from src.api.middleware import RequestIDMiddleware
from src.api.routes import ask_router, health_router, ingest_router


def create_app() -> FastAPI:
    """
    Build and configure the FastAPI application.

    :return: The configured application, with middleware, routers, and error handlers registered.
    """
    app = FastAPI(title="RAG Service API", lifespan=lifespan)
    app.add_middleware(RequestIDMiddleware)

    app.include_router(ask_router)
    app.include_router(ingest_router)
    app.include_router(health_router)

    errors.register(app)
    return app


app = create_app()
