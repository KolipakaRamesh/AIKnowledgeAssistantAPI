import uuid
from loguru import logger

from app.config import get_settings
from app.utils.file_parser import extract_text
from app.utils.chunker import chunk_text
from app.vectorstore.chroma_store import add_documents

settings = get_settings()


async def ingest_document(filename: str, file_bytes: bytes) -> dict:
    """
    Full ingestion pipeline:
    File bytes → Text extraction → Chunking → Embedding → Chroma storage

    Args:
        filename: Original uploaded filename (e.g. "report.pdf").
        file_bytes: Raw file content.

    Returns:
        dict with document_id, filename, chunks_indexed, status.
    """
    document_id = str(uuid.uuid4())
    log = logger.bind(document_id=document_id, filename=filename)

    log.info("Starting document ingestion")

    # 1. Extract text
    log.info("Extracting text from file")
    raw_text = extract_text(filename, file_bytes)

    if not raw_text.strip():
        raise ValueError("Could not extract any text from the uploaded file.")

    log.info(f"Extracted {len(raw_text)} characters")

    # 2. Chunk text
    log.info("Chunking text")
    chunks = chunk_text(raw_text, settings.chunk_size, settings.chunk_overlap)

    # Annotate each chunk with metadata for source attribution
    for i, chunk in enumerate(chunks):
        chunk.metadata.update(
            {
                "document_id": document_id,
                "filename": filename,
                "chunk_index": i,
            }
        )

    log.info(f"Created {len(chunks)} chunks")

    # 3. Embed + store
    log.info("Embedding and storing chunks in Chroma")
    count = add_documents(collection_name=document_id, documents=chunks)

    log.info(f"Successfully stored {count} chunks")

    return {
        "document_id": document_id,
        "filename": filename,
        "chunks_indexed": count,
        "status": "success",
    }
