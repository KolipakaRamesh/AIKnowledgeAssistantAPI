import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.config import get_settings
from app.middleware import APIKeyMiddleware
from app.routers import upload, ask, summarize, keypoints
from app.utils.logger import logger

settings = get_settings()

# ── Rate limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit_global])


# ── App Factory ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AI Knowledge Assistant API starting up")
    # Ensure Chroma persistence directory exists (writable in /tmp on Vercel)
    if settings.chroma_path:
        os.makedirs(settings.chroma_path, exist_ok=True)
    yield
    logger.info("AI Knowledge Assistant API shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Knowledge Assistant API",
        description=(
            "Upload documents and ask questions using Retrieval Augmented Generation (RAG). "
            "Responses are grounded in your uploaded content with source attribution.\n\n"
            "**Authentication**: Click **Authorize** and enter your `API_KEY` value from `.env` "
            "in the `X-API-Key` field to test protected endpoints."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── OpenAPI security scheme (makes Authorize button appear in Swagger UI) ──
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        schema["components"]["securitySchemes"] = {
            "ApiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
            }
        }
        # Apply security globally to all operations
        for path in schema.get("paths", {}).values():
            for operation in path.values():
                operation["security"] = [{"ApiKeyAuth": []}]
        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi

    # ── CORS ──────────────────────────────────────────────────────────────────
    origins = (
        [o.strip() for o in settings.allowed_origins.split(",")]
        if settings.allowed_origins != "*"
        else ["*"]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Rate limiting ─────────────────────────────────────────────────────────
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    # ── API Key auth ──────────────────────────────────────────────────────────
    app.add_middleware(APIKeyMiddleware)

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(upload.router, tags=["Document Ingestion"])
    app.include_router(ask.router, tags=["Q&A"])
    app.include_router(summarize.router, tags=["Summarization"])
    app.include_router(keypoints.router, tags=["Key Insights"])

    # ── Health check ──────────────────────────────────────────────────────────
    @app.get("/", tags=["Health"])
    async def root():
        return {"status": "ok", "service": "AI Knowledge Assistant API", "version": "1.0.0"}

    @app.get("/health", tags=["Health"])
    async def health():
        return {"status": "healthy"}

    return app


try:
    app = create_app()
except Exception as e:
    logger.critical(f"Failed to initialize FastAPI application: {e}")
    # Create a dummy app to respond with error for diagnostics
    app = FastAPI()
    @app.get("/{path:path}")
    async def caught_error(path: str):
        return {
            "error": "Initialization Failed", 
            "detail": str(e),
            "hint": "Check Vercel Environment Variables and Logs."
        }
