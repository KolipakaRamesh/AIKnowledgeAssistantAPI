import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.services.ingestion import ingest_document

HEADERS = {"X-API-Key": "test-secret", "Content-Type": "application/json"}

SAMPLE_TEXT = (
    "This annual report covers the company's performance in 2024. "
    "Revenue increased by 25% year-over-year to reach $10 million. "
    "Key growth drivers include AI product expansion and new enterprise clients. "
    "Challenges include rising infrastructure costs and talent acquisition. "
    "The board recommends expanding into Southeast Asian markets in 2025. "
    "Action items: hire 10 engineers, launch v2.0 product, establish Singapore office. " * 15
)


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest_asyncio.fixture
async def document_id():
    result = await ingest_document("annual_report.txt", SAMPLE_TEXT.encode())
    return result["document_id"]


@pytest.mark.asyncio
async def test_summarize_returns_structure(client, document_id):
    resp = await client.post(
        "/summarize",
        headers=HEADERS,
        json={"document_id": document_id},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
    assert isinstance(data.get("bullet_points"), list)
    assert isinstance(data.get("important_concepts"), list)


@pytest.mark.asyncio
async def test_extract_keypoints_returns_structure(client, document_id):
    resp = await client.post(
        "/extract-keypoints",
        headers=HEADERS,
        json={"document_id": document_id},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data.get("major_insights"), list)
    assert isinstance(data.get("important_terms"), list)
    assert isinstance(data.get("action_items"), list)


@pytest.mark.asyncio
async def test_summarize_unknown_doc(client):
    resp = await client.post(
        "/summarize",
        headers=HEADERS,
        json={"document_id": "non-existent-doc-id"},
    )
    assert resp.status_code in (404, 500)
