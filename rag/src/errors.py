class RagError(Exception):
    """
    Base for every error this service raises on purpose.
    """


class ResourceError(RagError):
    """
    An input that cannot be accepted.
    """


class ResourceTooLargeError(ResourceError):
    """
    An input that exceeds a configured cap.
    """


class UnsupportedFormatError(ResourceError):
    """
    An input whose format is not in the allowed set.
    """


class EmptyDocumentError(ResourceError):
    """
    An input the converter produced no text for.
    """


class UpstreamError(RagError):
    """
    An upstream service could not be reached, failed, or timed out.
    """


class DoclingError(UpstreamError):
    """
    The docling conversion service failed.
    """


class EmbedderError(UpstreamError):
    """
    The embedding service failed.
    """


class QdrantError(UpstreamError):
    """
    The Qdrant vector store failed.
    """


class RerankerError(UpstreamError):
    """
    The reranking service failed.
    """
