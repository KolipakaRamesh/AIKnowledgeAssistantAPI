from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional

from app.services.retrieval import answer_question
from app.utils.logger import logger

router = APIRouter()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000, description="The question to ask")
    document_id: Optional[str] = Field(
        None, description="Optional document ID to restrict search scope"
    )


@router.post(
    "/ask-question",
    summary="Ask a question using RAG over uploaded documents",
    response_description="Answer with source attribution and confidence score",
)
async def ask_question(body: AskRequest):
    """
    Embed the question, retrieve semantically relevant chunks from Chroma,
    and generate a grounded answer via LLM.

    The `source` field indicates:
    - `"vectordb"` — answer is grounded in retrieved document chunks
    - `"llm"` — no relevant context found; answer is from LLM's parametric knowledge
    """
    logger.info(f"Ask-question: '{body.question[:80]}'  doc_id={body.document_id}")

    try:
        result = await answer_question(body.question, body.document_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception("ask-question failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process question. Check server logs.",
        )

    return result
