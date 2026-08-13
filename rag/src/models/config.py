from pydantic import BaseModel, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)
from qdrant_client.models import Document

from src.enums import ChunkerKind, LogLevel


class QueueConfig(BaseModel):
    """
    Task queue settings.
    """

    redis_url: str
    job_timeout: int
    concurrency: int


class LoggingConfig(BaseModel):
    """
    Logging settings.
    """

    level: LogLevel


class IngestConfig(BaseModel):
    """
    Ingestion pipeline settings.
    """

    max_upload_bytes: int
    allowed_extensions: list[str]
    chunk_size: int


class DoclingConfig(BaseModel):
    """
    Settings for the docling document conversion service.
    """

    url: str
    timeout_sec: int
    chunker: ChunkerKind
    api_key: SecretStr | None = None


class EmbedderConfig(BaseModel):
    """
    Settings for the embedding service.
    """

    url: str
    model: str
    timeout_sec: int
    batch_size: int
    max_concurrency: int
    query_prefix: str | None = None
    passage_prefix: str | None = None
    api_key: SecretStr | None = None


class Bm25Config(BaseModel):
    """
    Settings for the server-side BM25 sparse vector computation.
    """

    language: str
    tokenizer: str

    def document(self, text: str) -> Document:
        """
        Build a Qdrant BM25 document for the given text.

        :param text: Text to be scored by the server-side BM25 model.
        :return: A document referencing the ``Qdrant/bm25`` model with this config's options.
        """
        return Document(
            text=text,
            model="Qdrant/bm25",
            options={"language": self.language, "tokenizer": self.tokenizer},
        )


class QdrantConfig(BaseModel):
    """
    Settings for the Qdrant vector store.
    """

    url: str
    timeout_sec: int
    upsert_batch_size: int
    dense_vector: str
    sparse_vector: str
    bm25: Bm25Config
    api_key: SecretStr | None = None


class RerankerConfig(BaseModel):
    """
    Settings for the reranking service.
    """

    url: str
    model: str
    timeout_sec: int
    api_key: SecretStr | None = None


class LLMConfig(BaseModel):
    """
    Settings for the answer generation service.
    """

    url: str
    model: str
    timeout_sec: int
    temperature: float
    max_tokens: int
    api_key: SecretStr | None = None


class RetrieveConfig(BaseModel):
    """
    Retrieval pipeline settings.
    """

    top_k: int
    top_n: int
    prefetch_multiplier: int


class WebhookConfig(BaseModel):
    """
    Settings for outgoing webhook callbacks.
    """

    timeout_sec: int


class AppConfig(BaseSettings):
    """
    Root settings for the service, combining yaml and environment variables.
    """

    model_config = SettingsConfigDict(
        yaml_file="/app/config.yaml",
        yaml_config_section="rag",
        env_prefix="RAG_",
        env_nested_delimiter="__",
        extra="ignore",
        protected_namespaces=(),
    )

    prompts_path: str
    queue: QueueConfig
    logging: LoggingConfig
    ingest: IngestConfig
    docling: DoclingConfig
    embedder: EmbedderConfig
    qdrant: QdrantConfig
    reranker: RerankerConfig
    llm: LLMConfig
    retrieve: RetrieveConfig
    webhook: WebhookConfig

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """
        Add the yaml source and order it so environment variables win.

        :param settings_cls: The settings class being built.
        :param init_settings: Values passed to the constructor.
        :param env_settings: Environment variable source.
        :param dotenv_settings: Dotenv file source.
        :param file_secret_settings: Docker/file secret source.
        :return: Ordered tuple of settings sources, earliest wins.
        """
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )
