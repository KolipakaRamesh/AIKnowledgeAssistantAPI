from app.config import get_settings
settings = get_settings()

from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
import os

from loguru import logger

# Shared embedding model (singleton)
_embeddings: OpenAIEmbeddings | None = None

def _get_embeddings() -> OpenAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        cache_dir = os.environ.get("TIKTOKEN_CACHE_DIR", "NOT SET")
        logger.info(f"Creating OpenAIEmbeddings instance (model: {settings.embedding_model}). Tiktoken cache: {cache_dir}")
        try:
            _embeddings = OpenAIEmbeddings(
                model=settings.embedding_model,
                openai_api_key=settings.openai_api_key if settings.openai_api_key else settings.openrouter_api_key,
                openai_api_base=settings.openrouter_base_url if not settings.openai_api_key else None,
            )
            logger.info("OpenAIEmbeddings instance created successfully")
        except Exception as e:
            logger.error(f"Failed to create OpenAIEmbeddings: {str(e)}")
            raise e
    return _embeddings

def get_vectorstore() -> PineconeVectorStore:
    """
    Return a LangChain Pinecone vectorstore.
    Note: Pinecone uses a single global index for this app. 
    We differentiate documents using metadata filters.
    """
    return PineconeVectorStore(
        index_name=settings.pinecone_index_name,
        embedding=_get_embeddings(),
        pinecone_api_key=settings.pinecone_api_key
    )

def add_documents(documents: list[Document]) -> int:
    """Embed and store documents in Pinecone."""
    log = logger.bind(chunk_count=len(documents))
    try:
        log.info("Initializing Pinecone vector store")
        store = get_vectorstore()
        
        log.info(f"Upserting {len(documents)} documents to Pinecone index: {settings.pinecone_index_name}")
        store.add_documents(documents)
        
        log.info("Successfully completed Pinecone upsert")
        return len(documents)
    except Exception as e:
        log.error(f"Failed to store documents in Pinecone: {str(e)}")
        # Provide a clean error message for user-facing diagnostics
        raise ValueError(f"Failed to store documents in Pinecone. This is likely a Vercel-specific issue. Detail: {str(e)}")

def similarity_search(
    query: str, document_id: str | None = None, k: int = 5
) -> list[tuple[Document, float]]:
    """
    Perform a similarity search in Pinecone.
    If document_id is provided, filters for that document.
    """
    store = get_vectorstore()
    
    filter = None
    if document_id:
        filter = {"document_id": document_id}
        
    return store.similarity_search_with_relevance_scores(query, k=k, filter=filter)

def list_all_document_ids() -> list[str]:
    """
    Note: Pinecone doesn't easily list 'collections' like Chroma.
    For this implementation, we return an empty list or rely on 
    other state management if needed. 
    For simple RAG, we usually query with specific document_id filters.
    """
    return []
