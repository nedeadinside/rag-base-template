from enum import StrEnum


class LogLevel(StrEnum):
    """
    Logging verbosity level.
    """

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class JobState(StrEnum):
    """
    Terminal outcome of an ingestion job.
    """

    SUCCESS = "success"
    FAILED = "failed"


class ChunkerKind(StrEnum):
    """
    Docling chunker strategy, matching the docling-serve path segment /v1/chunk/{kind}/file.
    """

    HYBRID = "hybrid"
    HIERARCHICAL = "hierarchical"
