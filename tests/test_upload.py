import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from app.main import app

HEADERS = {"X-API-Key": "test-secret"}


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


@pytest.mark.asyncio
async def test_missing_api_key(client):
    resp = await client.post("/upload-doc")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_upload_txt(client, tmp_path):
    sample = tmp_path / "sample.txt"
    sample.write_text(
        "This is a test document about artificial intelligence and machine learning. "
        "It discusses neural networks, deep learning, and natural language processing. "
        "The document covers supervised learning, unsupervised learning, and reinforcement learning. "
        "Key concepts include gradient descent, backpropagation, and transformer architectures. " * 10
    )
    with open(sample, "rb") as f:
        resp = await client.post(
            "/upload-doc",
            headers=HEADERS,
            files={"file": ("sample.txt", f, "text/plain")},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "document_id" in data
    assert data["status"] == "success"
    assert data["chunks_indexed"] > 0
    return data["document_id"]


@pytest.mark.asyncio
async def test_upload_unsupported_format(client):
    resp = await client.post(
        "/upload-doc",
        headers=HEADERS,
        files={"file": ("image.png", b"fake-png-bytes", "image/png")},
    )
    assert resp.status_code == 415
