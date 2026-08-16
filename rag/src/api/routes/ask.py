import logging

from fastapi import APIRouter

from src.api.deps import StateDep
from src.models import AskRequest, AskResponse, ContextChunk

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/ask")
async def ask(req: AskRequest, state: StateDep) -> AskResponse:
    """
    Answer a question against a collection.

    :param req: The query, target collection, metadata filter, and context flag.
    :param state: Application-wide dependencies.
    :return: The generated answer, with grounding chunks when requested.
    """
    answer = await state.pipeline.answer(req.query, req.collection, metadata_filter=req.metadata_filter)
    context = (
        [
            ContextChunk(text=chunk.text, score=chunk.score, document_id=chunk.document_id, metadata=chunk.metadata)
            for chunk in answer.sources
        ]
        if req.include_context
        else None
    )
    return AskResponse(answer=answer.text, context=context)
