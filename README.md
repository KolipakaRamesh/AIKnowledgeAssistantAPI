# 🧠 AI Knowledge Assistant API

A production-ready **Retrieval Augmented Generation (RAG)** API designed for high-performance document intelligence. Build, query, and extract insights from your documents with full source attribution.

> [!NOTE]
> Designed with production stability in mind, addressing common pitfalls highlighted in *"Most RAG Systems Fail in Production"* (e.g., robust chunking, metadata filtering, and cross-platform reliability).

---

## 🏗️ Architecture

```mermaid
graph TD
    Client["Client / Agent / MCP"] -- "HTTPS (X-API-Key)" --> FastAPI["FastAPI Server"]
    
    subgraph "Core Backend"
        FastAPI --> Middleware["Auth & Rate Limiting"]
        Middleware --> Router["API Routers"]
        Middleware --> Logger["Loguru (Structured JSON)"]
    end
    
    subgraph "Ingestion Pipeline"
        Router --> Parser["PyMuPDF Parser"]
        Parser --> Chunker["Recursive Chunker"]
        Chunker --> Embedder["OpenAI Embeddings"]
        Embedder --> Pinecone["Pinecone Vector DB"]
    end
    
    subgraph "Retrieval & RAG"
        Router --> Retriever["Vector Store Retriever"]
        Retriever --> LLM["LLM (OpenRouter / GPT-4o-mini)"]
        LLM --> Formatter["Response Formatter"]
    end
    
    subgraph "Observability"
        LLM -.-> LangSmith["LangSmith Tracing"]
        Retriever -.-> LangSmith
    end
    
    Pinecone -.-> Retriever
```

---

## ✨ Key Features

- **Document Ingestion**: Upload PDF, TXT, or Markdown files. Support for automatic chunking and indexing.
- **RAG-Powered Q&A**: Answers grounded in your documents with precise source attribution and confidence scoring.
- **AI-Driven Insights**:
    - **Summarization**: Structured summaries with key concepts and bullet points.
    - **Key Interest Points**: Extraction of major insights, important terms, and action items.
- **Developer First**:
    - **Swagger UI**: Interactive documentation at `/docs`.
    - **MCP Native**: Full support for Model Context Protocol via `mcp/mcp_manifest.json`.
    - **Production Ready**: Built-in rate limiting, security middleware, and structured JSON logging.
- **Enterprise Observability**: Native integration with LangSmith for deep tracing and evaluation.

---

## 🚀 Quick Start

### 1. Requirements
- Python 3.9+
- [OpenRouter API Key](https://openrouter.ai/keys) (for LLM)
- [Pinecone API Key](https://www.pinecone.io/) (for Vector Storage)

### 2. Setup
```bash
# Clone the repository
git clone https://github.com/your-repo/AIKnowledgeAssistantAPI.git
cd AIKnowledgeAssistantAPI

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and set your required API keys
```

### 3. Run Locally
```bash
uvicorn app.main:app --reload --port 8000
```
Visit **http://localhost:8000/docs** to explore the interactive API documentation.

---

## 🛡️ Production & Stability

- **Cross-Platform Compatibility**: Automatically handles environment-specific challenges (e.g., dynamically adjusting `TIKTOKEN_CACHE_DIR` for Vercel vs. Windows/Linux).
- **Pinecone Integration**: Uses Pinecone's serverless/pod-based architecture for scalable vector storage and fast retrieval.
- **Observability**: Built-in **LangSmith** tracing for debugging complex RAG chains.
- **Fail-Safe Startup**: Production-grade error handling that provides diagnostic hints even if initialization fails on Vercel.

---

## 🔌 API Summary

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload-doc` | Upload and index a document (PDF, TXT, MD). |
| `POST` | `/ask-question` | Retrieval-augmented Q&A from indexed content. |
| `POST` | `/summarize` | Generate structured summaries. |
| `POST` | `/extract-keypoints` | Extract insights, terms, and action items. |
| `GET` | `/health` | Check system status. |

---

## 🌐 RapidAPI Integration

Listing this API on [RapidAPI](https://rapidapi.com) is straightforward using the provided OpenAPI specification:

1. **Download the Spec**: Locate the [rapidapi_openapi.json](file:///d:/Development/AIKnowledgeAssistantAPI/rapidapi_openapi.json) file in this repository.
2. **Import to RapidAPI**:
   - Log in to the [RapidAPI Provider Dashboard](https://rapidapi.com/provider).
   - Click **Add New API**.
   - Choose **OpenAPI** as the definition type and upload `rapidapi_openapi.json`.
3. **Configure Authentication**:
   - Ensure the `X-API-Key` header is configured as the required authentication parameter for all protected endpoints.

---

## 🛡️ Security & Authentication

All protected endpoints require the **`X-API-Key`** header. This key is your internal secret defined in the `.env` file as `API_KEY`.

```bash
curl -X POST http://localhost:8000/ask-question \
  -H "X-API-Key: your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"question": "How does RAG work?"}'
```

---

## 🛠️ Tech Stack

- **API Framework**: [FastAPI](https://fastapi.tiangolo.com/)
- **Orchestration**: [LangChain](https://www.langchain.com/)
- **Vector Database**: [Pinecone](https://www.pinecone.io/)
- **Embeddings**: [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings) (`text-embedding-3-small`)
- **LLM Provider**: [OpenRouter](https://openrouter.ai/) (`gpt-4o-mini`)
- **Logging**: [Loguru](https://github.com/Delgan/loguru) (Structured JSON)
- **Tracing**: [LangSmith](https://smith.langchain.com/)
- **Deployment**: [Vercel](https://vercel.com/)
