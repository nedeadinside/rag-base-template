import hashlib
import logging
import uuid
from pathlib import Path
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from qdrant_client import models as qdrant_models

from src.clients import DoclingClient, EmbedderClient, LLMClient, QdrantClient, RerankerClient
from src.errors import EmptyDocumentError, QdrantError
from src.models import Answer, AppConfig, ChunkPayload, IngestResult, RetrievedChunk

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Facade over the ingestion and question-answering steps.
    """

    def __init__(
        self,
        config: AppConfig,
        prompts: dict[str, ChatPromptTemplate],
        *,
        docling: DoclingClient,
        embedder: EmbedderClient,
        qdrant: QdrantClient,
        reranker: RerankerClient,
        llm: LLMClient,
    ) -> None:
        """
        Store the settings, prompts, and upstream clients the steps run against.

        :param config: The service settings.
        :param prompts: Prompt templates keyed by prompt name.
        :param docling: Client for the docling chunking service.
        :param embedder: Client for the embedding service.
        :param qdrant: Client for the vector store.
        :param reranker: Client for the reranking service.
        :param llm: Client for the answer generation service.
        """
        self._config = config
        self._prompts = prompts
        self._docling = docling
        self._embedder = embedder
        self._qdrant = qdrant
        self._reranker = reranker
        self._llm = llm

    async def ingest(self, file_path: str, collection: str, metadata: dict[str, Any]) -> IngestResult:
        """
        Ingest one document: chunk it, embed the chunks, and upsert them into the vector store.

        :param file_path: Path to the spooled document on disk.
        :param collection: Target Qdrant collection.
        :param metadata: Caller-supplied metadata copied onto every point payload.
        :raises RagError: If the document yields no text, or chunking, embedding, or the upsert fails.
        :return: The ingestion result with the derived document id and produced chunk count.
        """
        chunks = await self._docling.chunk(file_path)
        if not chunks:
            raise EmptyDocumentError(f"no text extracted from {Path(file_path).name}")
        digest = hashlib.blake2b(digest_size=16)
        for chunk in chunks:
            digest.update(chunk.text.encode())
            digest.update(b"\n")
        document_id = str(uuid.UUID(bytes=digest.digest(), version=5))
        vectors = await self._embedder.embed(
            [chunk.text for chunk in chunks],
            prefix=self._config.embedder.passage_prefix,
        )
        await self._qdrant.ensure_collection(collection, vector_size=len(vectors[0]))
        points = [
            qdrant_models.PointStruct(
                id=str(uuid.uuid5(uuid.UUID(document_id), str(i))),
                vector={
                    self._config.qdrant.dense_vector: vector,
                    self._config.qdrant.sparse_vector: self._config.qdrant.bm25.document(chunk.text),
                },
                payload=ChunkPayload(
                    document_id=document_id,
                    text=chunk.text,
                    headings=chunk.headings,
                    metadata=metadata,
                ).model_dump(),
            )
            for i, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
        ]
        try:
            await self._qdrant.upsert(collection, points)
        except QdrantError:
            try:
                await self._qdrant.delete_document(collection, document_id)
            except QdrantError as cleanup_error:
                logger.warning("cleanup of document %s failed: %s", document_id, cleanup_error)
            raise
        return IngestResult(document_id=document_id, collection=collection, chunks=len(points))

    async def retrieve(
        self,
        query: str,
        collection: str,
        *,
        query_filter: qdrant_models.Filter | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve chunks for a query: embed it, search the vector store, and rerank the candidates.

        :param query: The natural-language query.
        :param collection: Target Qdrant collection.
        :param query_filter: Optional Qdrant filter applied to the search.
        :raises RagError: If embedding, search, or reranking fails.
        :return: The reranked chunks, best first.
        """
        qvec = (await self._embedder.embed([query], prefix=self._config.embedder.query_prefix))[0]
        hits = await self._qdrant.search(
            collection,
            qvec,
            query,
            limit=self._config.retrieve.top_k,
            prefetch_limit=self._config.retrieve.top_k * self._config.retrieve.prefetch_multiplier,
            query_filter=query_filter,
        )
        if not hits:
            return []
        payloads = [ChunkPayload.model_validate(hit.payload or {}) for hit in hits]
        ranked = await self._reranker.rerank(
            query,
            [payload.text for payload in payloads],
            top_n=self._config.retrieve.top_n,
        )
        return [
            RetrievedChunk(
                text="\n".join([*payloads[index].headings, payloads[index].text]),
                score=score,
                document_id=payloads[index].document_id,
                metadata=payloads[index].metadata,
            )
            for index, score in ranked
        ]

    async def generate(self, query: str, chunks: list[RetrievedChunk]) -> Answer:
        """
        Generate an answer for a query from the chunks retrieved for it.

        :param query: The natural-language query.
        :param chunks: The retrieved chunks used as grounding context.
        :raises RagError: If generation fails.
        :return: The generated answer with the chunks it was grounded on.
        """
        context = "\n\n".join(f"[{number}] {chunk.text}" for number, chunk in enumerate(chunks, 1))
        text = await self._llm.complete(self._prompts["answer"], {"context": context, "question": query})
        return Answer(text=text, sources=chunks)

    async def answer(
        self,
        query: str,
        collection: str,
        *,
        query_filter: qdrant_models.Filter | None = None,
    ) -> Answer:
        """
        Answer a query end to end: retrieve the grounding chunks, then generate over them.

        :param query: The natural-language query.
        :param collection: Target Qdrant collection.
        :param query_filter: Optional Qdrant filter applied to the search.
        :raises RagError: If retrieval or generation fails.
        :return: The generated answer with the chunks it was grounded on.
        """
        chunks = await self.retrieve(query, collection, query_filter=query_filter)
        return await self.generate(query, chunks)
