from fastapi import APIRouter, UploadFile, File, HTTPException, Request, status
from app.services.ingestion import ingest_document
from app.utils.logger import logger

router = APIRouter()

ALLOWED_EXTENSIONS = {"pdf", "txt", "md"}
MAX_FILE_SIZE_MB = 20


@router.post(
    "/upload-doc",
    summary="Upload a document for RAG indexing",
    response_description="Document ID and ingestion stats",
)
async def upload_document(request: Request, file: UploadFile = File(...)):
    """
    Accept a PDF, TXT, or Markdown file.
    Extract text, chunk it, generate embeddings, and store in Chroma.
    Returns a `document_id` to use with other endpoints.
    """
    # --- Validate file extension ---
    filename = file.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '.{ext}'. Allowed: pdf, txt, md",
        )

    # --- Read and validate file size ---
    file_bytes = await file.read()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large ({size_mb:.1f} MB). Max allowed: {MAX_FILE_SIZE_MB} MB",
        )

    logger.info(f"Received file: {filename} ({size_mb:.2f} MB)")

    try:
        result = await ingest_document(filename, file_bytes)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.exception(f"Ingestion failed for {filename}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {str(e)}",
        )

    return result
