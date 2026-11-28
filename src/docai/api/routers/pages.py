"""The single-page web UI and the interactive API docs."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from docai.settings import get_settings

router = APIRouter(include_in_schema=False)


def _page(name: str) -> FileResponse:
    page = get_settings().web_dir / name
    if not page.is_file():
        raise HTTPException(404, "page not found")
    return FileResponse(page, media_type="text/html")


@router.get("/")
def index() -> FileResponse:
    """Serve ``web/index.html``."""
    return _page("index.html")


@router.get("/dashboard")
def dashboard() -> FileResponse:
    """Cost & quality trends across recorded runs (``web/dashboard.html``)."""
    return _page("dashboard.html")


@router.get("/docs")
def docs() -> FileResponse:
    """Swagger UI (assets pinned by version + SRI); disable with ``DOCAI_DOCS_ENABLED=false``."""
    if not get_settings().docs_enabled:
        raise HTTPException(404, "API docs are disabled")
    return _page("docs.html")
