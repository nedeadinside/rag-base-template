import uvicorn

from src.config import load_config
from src.setup_logging import configure_logging


def main() -> None:
    """
    Configure logging and run the API server.
    """
    configure_logging()
    server = load_config().server
    uvicorn.run("src.api:app", host=server.host, port=server.port, log_config=None)
