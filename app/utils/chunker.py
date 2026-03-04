from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[Document]:
    """
    Split a plain-text string into overlapping LangChain Document chunks.

    Args:
        text: Raw extracted text.
        chunk_size: Max characters per chunk.
        chunk_overlap: Characters of overlap between adjacent chunks.

    Returns:
        List of LangChain Document objects with page_content set.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.create_documents([text])
