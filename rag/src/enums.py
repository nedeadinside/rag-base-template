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
    Lifecycle state of an ingestion job.
    """

    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    NOT_FOUND = "not_found"


class ChunkerKind(StrEnum):
    """
    Docling chunker strategy, matching the docling-serve path segment /v1/chunk/{kind}/file.
    """

    HYBRID = "hybrid"
    HIERARCHICAL = "hierarchical"
