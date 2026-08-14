from langchain_core.prompts import ChatPromptTemplate
from qdrant_client import models as qdrant_models

from src.clients import EmbedderClient, LLMClient, QdrantClient, RerankerClient
from src.models import Answer, AppConfig, ChunkPayload, RetrievedChunk, VerifierOutput

INSUFFICIENT_MESSAGE = "There is not enough information to answer this question."


class Pipeline:
    """
    Facade over the question-answering steps.
    """

    def __init__(
        self,
        config: AppConfig,
        prompts: dict[str, ChatPromptTemplate],
        *,
        embedder: EmbedderClient,
        qdrant: QdrantClient,
        reranker: RerankerClient,
        llm: LLMClient,
    ) -> None:
        """
        Store the settings, prompts, and upstream clients the steps run against.

        :param config: The service settings.
        :param prompts: Prompt templates keyed by prompt name.
        :param embedder: Client for the embedding service.
        :param qdrant: Client for the vector store.
        :param reranker: Client for the reranking service.
        :param llm: Client for the answer generation service.
        """
        self._config = config
        self._prompts = prompts
        self._embedder = embedder
        self._qdrant = qdrant
        self._reranker = reranker
        self._llm = llm

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

    async def verify(self, query: str, chunks: list[RetrievedChunk]) -> bool:
        """
        Ask the LLM whether the retrieved chunks contain enough information to answer the query.

        :param query: The natural-language query.
        :param chunks: The retrieved chunks to check.
        :raises RagError: If the verification request fails.
        :return: True if the chunks contain a direct and complete answer.
        """
        context = "\n---\n".join(chunk.text for chunk in chunks)
        result = await self._llm.complete_structured(
            self._prompts["verify"], {"query": query, "chunks": context}, VerifierOutput
        )
        return result.can_answer

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
        Answer a query end to end: retrieve, verify, then generate over the grounding chunks.

        Returns an honest "not enough information" answer instead of generating when no chunks
        are retrieved, or when verification is enabled and the chunks are judged insufficient.

        :param query: The natural-language query.
        :param collection: Target Qdrant collection.
        :param query_filter: Optional Qdrant filter applied to the search.
        :raises RagError: If retrieval, verification, or generation fails.
        :return: The generated answer with the chunks it was grounded on.
        """
        chunks = await self.retrieve(query, collection, query_filter=query_filter)
        if not chunks:
            return Answer(text=INSUFFICIENT_MESSAGE, sources=[])
        if self._config.retrieve.verify and not await self.verify(query, chunks):
            return Answer(text=INSUFFICIENT_MESSAGE, sources=[])
        return await self.generate(query, chunks)
