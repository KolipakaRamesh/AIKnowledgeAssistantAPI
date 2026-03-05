import os
import tempfile
import multiprocessing.pool
from concurrent.futures import ThreadPoolExecutor

# ── Multiprocessing Monkeypatch for Vercel ──────────────────────────────────
# Vercel's serverless environment does not support semaphores (required by 
# multiprocessing.synchronize.SemLock). Some libraries like pinecone-client 
# use multiprocessing.pool.ThreadPool, which triggers this. We replace it 
# with a ThreadPoolExecutor-based implementation that avoids semaphores.

class VercelResult:
    def __init__(self, future_or_futures, is_list=False):
        self._data = future_or_futures
        self._is_list = is_list

    def get(self, timeout=None):
        if self._is_list:
            return [f.result(timeout=timeout) for f in self._data]
        return self._data.result(timeout=timeout)

    def ready(self):
        if self._is_list:
            return all(f.done() for f in self._data)
        return self._data.done()

    def successful(self):
        if not self.ready():
            raise ValueError("Result not ready")
        if self._is_list:
            return all(f.exception() is None for f in self._data)
        return self._data.exception() is None

class VercelThreadPool:
    def __init__(self, processes=None, initializer=None, initargs=()):
        self._executor = ThreadPoolExecutor(max_workers=processes)

    def apply_async(self, func, args=(), kwds={}, callback=None, error_callback=None):
        future = self._executor.submit(func, *args, **kwds)
        if callback:
            future.add_done_callback(lambda f: callback(f.result()) if not f.exception() else None)
        if error_callback:
            future.add_done_callback(lambda f: error_callback(f.exception()) if f.exception() else None)
        return VercelResult(future)

    def map_async(self, func, iterable, chunksize=None, callback=None, error_callback=None):
        futures = [self._executor.submit(func, item) for item in iterable]
        if callback:
            # Simple combined callback for map_async
            def _done(fs):
                if all(f.done() for f in fs) and all(f.exception() is None for f in fs):
                    callback([f.result() for f in fs])
            for f in futures:
                f.add_done_callback(lambda _: _done(futures))
        return VercelResult(futures, is_list=True)

    def close(self):
        self._executor.shutdown(wait=False)

    def join(self):
        self._executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# Apply the monkeypatch
multiprocessing.pool.ThreadPool = VercelThreadPool

# ── Aggressive Environment Setup (Vercel Fix) ───────────────────────────────
# We set multiple environment variables to point to /tmp to ensure all libraries
# (like tiktoken, langchain, etc.) have a writable home and cache.
if os.getenv("VERCEL") == "1" or os.getenv("NOW_REGION"):
    tmp_base = tempfile.gettempdir()
    os.environ["HOME"] = tmp_base
    os.environ["XDG_CACHE_HOME"] = os.path.join(tmp_base, ".cache")
    
    tiktoken_cache = os.path.join(tmp_base, "tiktoken_cache")
    os.makedirs(tiktoken_cache, exist_ok=True)
    os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache
    
    print(f"--- INFO: Vercel detected. HOME={os.environ['HOME']}, TIKTOKEN_CACHE_DIR={os.environ['TIKTOKEN_CACHE_DIR']} ---")
    
    # Early test for tiktoken initialization
    try:
        import tiktoken
        # Try to get an encoder to trigger cache access/creation
        tiktoken.get_encoding("cl100k_base")
        print("--- INFO: Tiktoken initialization test: SUCCESS ---")
    except Exception as e:
        print(f"--- WARNING: Tiktoken initialization test: FAILED: {e} ---")

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
