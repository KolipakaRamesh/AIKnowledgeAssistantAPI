import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from app.config import get_settings

settings = get_settings()

# Shared Chroma persistent client (singleton)
_chroma_client: chromadb.ClientAPI | None = None

# Shared embedding model (loaded once into memory — ~80 MB)
_embeddings: HuggingFaceEmbeddings | None = None


def _get_client() -> chromadb.ClientAPI:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=settings.chroma_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _chroma_client


def _get_embeddings() -> HuggingFaceEmbeddings:
    """
    Load the sentence-transformers model once and cache it.
    Uses all-MiniLM-L6-v2 by default — small (80 MB), fast, and free.
    No API key required.
    """
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_vectorstore(collection_name: str) -> Chroma:
    """
    Return a LangChain Chroma vectorstore for the given collection.
    Each document gets its own isolated Chroma collection (keyed by document_id).
    """
    return Chroma(
        client=_get_client(),
        collection_name=collection_name,
        embedding_function=_get_embeddings(),
    )


def add_documents(collection_name: str, documents: list[Document]) -> int:
    """Embed and store documents in the named Chroma collection."""
    store = get_vectorstore(collection_name)
    store.add_documents(documents)
    return len(documents)


def similarity_search(
    collection_name: str, query: str, k: int = 5
) -> list[tuple[Document, float]]:
    """
    Perform a similarity search and return (document, score) tuples.
    Score is cosine similarity (0–1, higher = more relevant).
    """
    store = get_vectorstore(collection_name)
    return store.similarity_search_with_relevance_scores(query, k=k)


def list_all_collections() -> list[str]:
    """Return names of all Chroma collections (i.e., all document IDs indexed so far)."""
    client = _get_client()
    return [col.name for col in client.list_collections()]
