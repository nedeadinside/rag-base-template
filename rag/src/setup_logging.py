import logging
import sys
from contextvars import ContextVar

from src.config import load_config

request_id_var: ContextVar[str] = ContextVar("request_id", default="")


class RequestIDFormatter(logging.Formatter):
    """
    Formatter for logs.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Attach the active request id to the record.

        :param record: The log record to format.
        :return: The formatted log line.
        """
        request_id = request_id_var.get("")
        record.request_id = f"[{request_id}]" if request_id else "[None]"
        return super().format(record)


def configure_logging() -> None:
    """
    Configure root logger once for the whole process, writing to stdout.
    """
    root = logging.getLogger()

    if root.handlers:
        return

    config = load_config().logging
    root.setLevel(config.level)

    for noisy_logger in ("httpx", "httpcore", "urllib3", "langchain_core", "openai", "qdrant_client"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    formatter = RequestIDFormatter(
        fmt="[%(levelname)s] %(request_id)s %(asctime)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(config.level)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)
