from functools import lru_cache

from models.config import AppConfig


@lru_cache
def load_config() -> AppConfig:
    """
    Return the cached settings instance for the service.

    :return: The validated settings.
    """
    return AppConfig()
