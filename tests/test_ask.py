import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.services.ingestion import ingest_document

HEADERS = {"X-API-Key": "test-secret", "Content-Type": "application/json"}

SAMPLE_TEXT = (
    "Artificial intelligence is transforming industries. "
    "Machine learning algorithms use data to make predictions. "
    "Natural language processing enables computers to understand text. "
    "Deep learning uses neural networks with multiple layers. "
    "Reinforcement learning trains agents through reward signals. " * 20
)


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest_asyncio.fixture
async def document_id():
    result = await ingest_document("test.txt", SAMPLE_TEXT.encode())
    return result["document_id"]


@pytest.mark.asyncio
async def test_ask_question_returns_answer(client, document_id):
    resp = await client.post(
        "/ask-question",
        headers=HEADERS,
        json={"question": "What is artificial intelligence?", "document_id": document_id},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert data["source"] in ("vectordb", "llm")
    assert "confidence" in data
    assert isinstance(data["sources"], list)


@pytest.mark.asyncio
async def test_ask_question_source_vectordb(client, document_id):
    """When doc is indexed, source should be 'vectordb'."""
    resp = await client.post(
        "/ask-question",
        headers=HEADERS,
        json={"question": "What does the document discuss?", "document_id": document_id},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["source"] == "vectordb"
    assert data["confidence"] >= 0


@pytest.mark.asyncio
async def test_ask_question_short_query_rejected(client):
    resp = await client.post(
        "/ask-question",
        headers=HEADERS,
        json={"question": "a"},
    )
    assert resp.status_code == 422
