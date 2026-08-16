import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.api.deps import StateDep
from src.models import HealthReport

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthReport, responses={503: {"model": HealthReport}})
async def health(state: StateDep) -> JSONResponse:
    """
    Report whether the service and its upstream dependencies are reachable.

    :param state: Application-wide dependencies.
    :return: A 200 response when every check passes, otherwise a 503.
    """
    checks: dict[str, bool] = {"qdrant": await state.qdrant.ping()}

    try:
        await state.redis.ping()
        checks["redis"] = True
    except Exception:
        logger.warning("Healthcheck: redis unreachable", exc_info=True)
        checks["redis"] = False

    ok = all(checks.values())
    return JSONResponse(
        status_code=200 if ok else 503,
        content={"status": "ok" if ok else "degraded", "checks": checks},
    )
