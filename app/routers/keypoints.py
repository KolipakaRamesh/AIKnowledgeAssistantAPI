from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.services.retrieval import extract_keypoints
from app.utils.logger import logger

router = APIRouter()


class KeypointsRequest(BaseModel):
    document_id: str = Field(..., description="ID returned from /upload-doc")


@router.post(
    "/extract-keypoints",
    summary="Extract major insights, terms, and action items from a document",
    response_description="Major insights, important terms, and action items",
)
async def keypoints(body: KeypointsRequest):
    """
    Retrieve document content from Chroma and extract:
    - Major insights
    - Important domain terms
    - Actionable items
    """
    logger.info(f"Extract-keypoints request for document_id={body.document_id}")

    try:
        result = await extract_keypoints(body.document_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception("extract-keypoints failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to extract key points. Check server logs.",
        )

    return result
