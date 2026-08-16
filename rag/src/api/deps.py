from typing import Annotated

from fastapi import Depends, Request

from src.api.state import AppState


def get_state(request: Request) -> AppState:
    """
    Retrieve the application state stored on the FastAPI app.

    :param request: The incoming request.
    :return: The application-wide dependencies.
    """
    return request.app.state.deps


StateDep = Annotated[AppState, Depends(get_state)]
