from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.services.retrieval import summarize_document
from app.utils.logger import logger

router = APIRouter()


class SummarizeRequest(BaseModel):
    document_id: str = Field(..., description="ID returned from /upload-doc")


@router.post(
    "/summarize",
    summary="Generate a structured summary of an uploaded document",
    response_description="Summary, bullet points, and important concepts",
)
async def summarize(body: SummarizeRequest):
    """
    Retrieve document content from Chroma and produce a structured summary
    containing an overview, key bullet points, and important concepts.
    """
    logger.info(f"Summarize request for document_id={body.document_id}")

    try:
        result = await summarize_document(body.document_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception("summarize failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate summary. Check server logs.",
        )

    return result
