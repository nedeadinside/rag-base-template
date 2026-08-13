from functools import lru_cache
from pathlib import Path

import yaml
from langchain_core.prompts import ChatPromptTemplate

from src.models import AppConfig


@lru_cache
def load_config() -> AppConfig:
    """
    Return the cached settings instance for the service.

    :return: The validated settings.
    """
    return AppConfig()


@lru_cache
def load_prompts() -> dict[str, ChatPromptTemplate]:
    """
    Return the cached prompt templates for the service, keyed by prompt name.

    :return: The compiled prompt templates.
    """
    raw = yaml.safe_load(Path(load_config().prompts_path).read_text(encoding="utf-8"))
    return {name: ChatPromptTemplate.from_messages(messages) for name, messages in raw.items()}
