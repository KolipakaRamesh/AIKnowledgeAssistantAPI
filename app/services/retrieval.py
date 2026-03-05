import os
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from app.config import get_settings
from app.vectorstore.pinecone_store import similarity_search
from loguru import logger

settings = get_settings()

# ── Prompt templates ──────────────────────────────────────────────────────────

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a precise, helpful AI assistant. Use ONLY the context below to answer the question.
If the context does not contain enough information, say "I don't have enough information in the uploaded documents to answer this."

Context:
{context}

Question: {question}

Answer (be concise and cite specific facts from the context):""",
)

SUMMARY_PROMPT = PromptTemplate(
    input_variables=["context"],
    template="""Analyze the following document content and provide a structured summary.

Document Content:
{context}

Respond in this EXACT JSON format (no markdown, raw JSON only):
{{
  "summary": "A concise 2-3 sentence overview of the document",
  "bullet_points": ["key point 1", "key point 2", "key point 3"],
  "important_concepts": ["concept1", "concept2", "concept3"]
}}""",
)

KEYPOINTS_PROMPT = PromptTemplate(
    input_variables=["context"],
    template="""Extract structured insights from the following document content.

Document Content:
{context}

Respond in this EXACT JSON format (no markdown, raw JSON only):
{{
  "major_insights": ["insight 1", "insight 2", "insight 3"],
  "important_terms": ["term1", "term2", "term3"],
  "action_items": ["action 1", "action 2"]
}}""",
)


def _get_llm() -> ChatOpenAI:
    """
    Returns a ChatOpenAI instance pointed at OpenRouter's API.
    OpenRouter is fully OpenAI-compatible — just needs a different base_url and api_key.
    """
    return ChatOpenAI(
        model=settings.llm_model,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=0.2,
        # OpenRouter requires these headers for app identification
        default_headers={
            "HTTP-Referer": "https://github.com/KolipakaRamesh/AIKnowledgeAssistantAPI",
            "X-Title": "AI Knowledge Assistant API",
        },
    )


def _build_context_from_chunks(chunks: list[tuple[Document, float]]) -> str:
    parts = [
        f"[Chunk {i+1} — relevance: {score:.2f}]\n{doc.page_content}"
        for i, (doc, score) in enumerate(chunks)
    ]
    return "\n\n---\n\n".join(parts)


def _format_sources(chunks: list[tuple[Document, float]]) -> list[dict]:
    return [
        {
            "chunk_id": f"{doc.metadata.get('document_id', 'unknown')}-chunk-{doc.metadata.get('chunk_index', i)}",
            "text": doc.page_content[:300],
            "score": round(score, 4),
            "filename": doc.metadata.get("filename", "unknown"),
        }
        for i, (doc, score) in enumerate(chunks)
    ]


# ── Public service functions ──────────────────────────────────────────────────

async def answer_question(question: str, document_id: str | None = None) -> dict:
    """
    RAG pipeline: embed query → vector search in Pinecone → LLM answer via OpenRouter.
    Returns answer, source flag ('vectordb'|'llm'), confidence, and source chunks.
    """
    log = logger.bind(question=question[:80], document_id=document_id)
    log.info("Processing ask-question request")

    try:
        # If document_id is provided, we filter by it. 
        # If None, Pinecone searches the whole index.
        top_chunks = similarity_search(question, document_id=document_id, k=5)
    except Exception as e:
        log.warning(f"Vector search failed: {e}")
        return await _llm_only_answer(question)

    if not top_chunks:
        log.warning("No relevant chunks found — falling back to pure LLM")
        return await _llm_only_answer(question)

    avg_score = sum(s for _, s in top_chunks) / len(top_chunks)
    context = _build_context_from_chunks(top_chunks)
    
    llm = _get_llm()
    prompt_text = QA_PROMPT.format(context=context, question=question)
    response = llm.invoke(prompt_text)
    answer = response.content.strip()

    log.info(f"Answer generated from Pinecone; avg_score={avg_score:.3f}")

    return {
        "answer": answer,
        "source": "vectordb",
        "confidence": round(max(0.0, avg_score), 4),
        "sources": _format_sources(top_chunks),
    }


async def _llm_only_answer(question: str) -> dict:
    """Fallback: answer using only the LLM's parametric knowledge."""
    llm = _get_llm()
    response = llm.invoke(question)
    return {
        "answer": response.content.strip(),
        "source": "llm",
        "confidence": 0.0,
        "sources": [],
    }


async def summarize_document(document_id: str) -> dict:
    """Retrieve all chunks and produce a structured summary via OpenRouter LLM."""
    log = logger.bind(document_id=document_id)
    log.info("Generating document summary")

    chunks = similarity_search(
        "document summary overview introduction", 
        document_id=document_id, 
        k=20
    )
    if not chunks:
        raise ValueError(f"No content found for document_id: {document_id}")

    context = _build_context_from_chunks(chunks)
    llm = _get_llm()
    prompt_text = SUMMARY_PROMPT.format(context=context)
    response = llm.invoke(prompt_text)

    try:
        result = json.loads(response.content.strip())
    except json.JSONDecodeError:
        result = {
            "summary": response.content.strip(),
            "bullet_points": [],
            "important_concepts": [],
        }

    log.info("Summary generated successfully")
    return result


async def extract_keypoints(document_id: str) -> dict:
    """Retrieve all chunks and extract structured key insights via OpenRouter LLM."""
    log = logger.bind(document_id=document_id)
    log.info("Extracting key points")

    chunks = similarity_search(
        "key insights action items important terms results", 
        document_id=document_id, 
        k=20
    )
    if not chunks:
        raise ValueError(f"No content found for document_id: {document_id}")

    context = _build_context_from_chunks(chunks)
    llm = _get_llm()
    prompt_text = KEYPOINTS_PROMPT.format(context=context)
    response = llm.invoke(prompt_text)

    try:
        result = json.loads(response.content.strip())
    except json.JSONDecodeError:
        result = {
            "major_insights": [response.content.strip()],
            "important_terms": [],
            "action_items": [],
        }

    log.info("Key points extracted successfully")
    return result
