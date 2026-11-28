"""FastAPI application factory.

Routers
-------
* ``routers.documents``  ``/health``, ``/extract``, ``/ask``, ``/samples``
* ``routers.review``     ``/review/items`` (human review queue; mutations need the token)
* ``routers.metrics``    ``/metrics/runs`` (run history for the dashboard)
* ``routers.pages``      ``/``, ``/dashboard`` (static UI under ``/static``), ``/docs``

Cross-cutting concerns are ASGI middleware (``docai.api.middleware``) and
exception handlers (``docai.api.errors``); see SECURITY.md for the rationale.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from docai import __version__
from docai.api.errors import install_error_handlers
from docai.api.middleware import (
    BodySizeLimitMiddleware,
    RateLimitMiddleware,
    RequestContextMiddleware,
)
from docai.api.ratelimit import TokenBucketLimiter
from docai.api.routers import documents, metrics, pages, review
from docai.observability import configure_logging
from docai.settings import get_settings

# Room for the multipart framing and the small form fields around the files.
MULTIPART_OVERHEAD = 64 * 1024


def create_app() -> FastAPI:
    """Build the application from the current settings."""
    settings = get_settings()
    configure_logging(
        settings.log_level, json_output=settings.log_json, secrets=settings.secret_values()
    )
    app = FastAPI(
        title="ai-guardrails",
        version=__version__,
        description="LLM extraction of Brazilian fiscal documents with deterministic guardrails. "
        "Synthetic data only.",
        docs_url=None,  # served by routers.pages with a strict CSP
        redoc_url=None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
    )
    install_error_handlers(app)
    if settings.web_dir.is_dir():
        app.mount("/static", StaticFiles(directory=settings.web_dir), name="static")
    app.include_router(documents.router)
    app.include_router(review.router)
    app.include_router(metrics.router)
    app.include_router(pages.router)

    # add_middleware prepends: the last one added is the outermost.
    max_body = settings.max_upload_bytes + settings.max_xml_bytes + MULTIPART_OVERHEAD
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=max_body)
    if settings.rate_limit_per_minute > 0:
        limiter = TokenBucketLimiter(settings.rate_limit_per_minute, settings.rate_limit_burst)
        app.add_middleware(RateLimitMiddleware, limiter=limiter)
    if settings.cors_origins:  # no CORS at all unless an explicit allowlist is configured
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
            allow_credentials=False,
        )
    app.add_middleware(RequestContextMiddleware)
    return app


app = create_app()
