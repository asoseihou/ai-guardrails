"""Exception handlers: every error body is ``{"detail": ..., "request_id": ...}``.

Clients never see exception class names, stack traces or provider messages;
those are logged server-side under the same request id.
"""

from __future__ import annotations

import logging
from typing import Any, cast

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from docai.extract.pdf import PdfRejected
from docai.extract.providers import ProviderError
from docai.extract.xml_reader import XmlRejected
from docai.guardrails.validate import CorrectionRejected
from docai.observability import request_id_var
from docai.review import QueueFull, ReviewConflict, ReviewNotFound

log = logging.getLogger("docai.api")


def error_response(status: int, detail: Any, headers: dict[str, str] | None = None) -> JSONResponse:
    """JSON error body carrying the request id."""
    body = {"detail": detail, "request_id": request_id_var.get()}
    return JSONResponse(body, status_code=status, headers=headers)


async def _http_error(request: Request, exc: Exception) -> JSONResponse:
    http = cast(StarletteHTTPException, exc)  # registered for this type only
    return error_response(http.status_code, http.detail, dict(http.headers or {}))


async def _validation_error(request: Request, exc: Exception) -> JSONResponse:
    invalid = cast(RequestValidationError, exc)  # registered for this type only
    # Only where and what: no echo of the submitted value ("input") or internals ("ctx").
    errors = [
        {"loc": list(err.get("loc", ())), "msg": err.get("msg", "invalid value")}
        for err in invalid.errors()
    ]
    return error_response(422, errors)


async def _pdf_rejected(request: Request, exc: Exception) -> JSONResponse:
    return error_response(422, f"PDF rejected: {exc}")


async def _xml_rejected(request: Request, exc: Exception) -> JSONResponse:
    return error_response(422, f"XML rejected: {exc}")


async def _review_not_found(request: Request, exc: Exception) -> JSONResponse:
    return error_response(404, "unknown review item")


async def _review_conflict(request: Request, exc: Exception) -> JSONResponse:
    return error_response(409, str(exc))


async def _correction_rejected(request: Request, exc: Exception) -> JSONResponse:
    rejected = cast(CorrectionRejected, exc)
    return error_response(
        422, {"message": "rejected by the guardrail rules", "reasons": rejected.reasons}
    )


async def _queue_full(request: Request, exc: Exception) -> JSONResponse:
    return error_response(503, str(exc))


async def _provider_error(request: Request, exc: Exception) -> JSONResponse:
    log.error("LLM provider error: %s", exc)
    return error_response(502, "the LLM provider failed; see the server logs for this request id")


def install_error_handlers(app: FastAPI) -> None:
    """Register the handlers on ``app``."""
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(PdfRejected, _pdf_rejected)
    app.add_exception_handler(XmlRejected, _xml_rejected)
    app.add_exception_handler(ProviderError, _provider_error)
    app.add_exception_handler(ReviewNotFound, _review_not_found)
    app.add_exception_handler(ReviewConflict, _review_conflict)
    app.add_exception_handler(CorrectionRejected, _correction_rejected)
    app.add_exception_handler(QueueFull, _queue_full)
