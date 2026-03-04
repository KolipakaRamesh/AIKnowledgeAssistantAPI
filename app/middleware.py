from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import get_settings

settings = get_settings()

# Paths that bypass API key auth (health check, docs)
_PUBLIC_PATHS = {"/", "/health", "/docs", "/redoc", "/openapi.json"}


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Validates X-API-Key header on all non-public routes.
    Returns 401 if key is missing or invalid.
    """

    async def dispatch(self, request: Request, call_next):
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        # Strip whitespace from both the incoming key and the configured key
        incoming_key = request.headers.get("X-API-Key", "").strip()
        configured_key = settings.api_key.strip()

        if incoming_key != configured_key:
            from app.utils.logger import logger
            logger.warning(
                f"API Key mismatch for path: {request.url.path}. "
                f"Received: {incoming_key[:3]}...{incoming_key[-3:] if len(incoming_key) > 6 else ''}, "
                f"Expected: {configured_key[:3]}...{configured_key[-3:] if len(configured_key) > 6 else ''}"
            )
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "Invalid or missing API key.",
                    "hint": "Pass your API key in the X-API-Key header. Ensure no leading/trailing spaces.",
                },
            )
        return await call_next(request)
