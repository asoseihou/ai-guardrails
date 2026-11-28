"""Pure ASGI middleware: request ids, security headers, error shielding, body and rate limits.

Order (outermost first), as installed by :func:`docai.api.app.create_app`::

    RequestContextMiddleware   request id, security headers, access log, generic 500s
    RateLimitMiddleware        token bucket per client for mutating methods -> 429
    BodySizeLimitMiddleware    Content-Length check + streamed byte count -> 413

All three are plain ASGI callables (no ``BaseHTTPMiddleware``), so uploads are
streamed through them instead of being buffered.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from docai.api.ratelimit import TokenBucketLimiter
from docai.observability import request_id_var

log = logging.getLogger("docai.api")

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{8,64}")

# The UI is static files only: no inline script or style, no third-party origin.
UI_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "font-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; "
    "form-action 'self'; frame-ancestors 'none'"
)
# /docs loads Swagger UI from jsDelivr, pinned by version and Subresource Integrity.
DOCS_CSP = (
    "default-src 'self'; script-src 'self' https://cdn.jsdelivr.net; "
    "style-src 'self' https://cdn.jsdelivr.net; img-src 'self' data:; connect-src 'self'; "
    "object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
)
BASE_HEADERS = {
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    "referrer-policy": "no-referrer",
    "permissions-policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
    "cross-origin-opener-policy": "same-origin",
    "cross-origin-resource-policy": "same-origin",
}


def security_headers(path: str) -> dict[str, str]:
    """Headers added to every response (``Content-Security-Policy`` depends on the path)."""
    headers = dict(BASE_HEADERS)
    headers["content-security-policy"] = DOCS_CSP if path == "/docs" else UI_CSP
    if not path.startswith("/static/"):
        headers["cache-control"] = "no-store"
    return headers


async def send_json(
    send: Send, status: int, body: dict[str, Any], headers: dict[str, str] | None = None
) -> None:
    """Send a complete JSON response."""
    payload = json.dumps(body).encode()
    raw_headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(payload)).encode()),
    ]
    raw_headers += [(k.encode(), v.encode()) for k, v in (headers or {}).items()]
    await send({"type": "http.response.start", "status": status, "headers": raw_headers})
    await send({"type": "http.response.body", "body": payload})


def _header(scope: Scope, name: bytes) -> str | None:
    for key, value in scope.get("headers", []):
        if key.lower() == name:
            return str(value.decode("latin-1"))
    return None


class RequestContextMiddleware:
    """Correlation id, security headers, one access-log line, and no internals in 500s."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle one ASGI connection."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        incoming = _header(scope, b"x-request-id")
        request_id = incoming if incoming and _REQUEST_ID.fullmatch(incoming) else uuid.uuid4().hex
        token = request_id_var.set(request_id)
        extra = security_headers(scope["path"])
        extra["x-request-id"] = request_id
        status = {"code": 500, "started": False}
        started = time.perf_counter()

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                status["code"], status["started"] = message["status"], True
                present = {k.lower() for k, _ in message.get("headers", [])}
                headers = list(message.get("headers", []))
                headers += [
                    (k.encode(), v.encode()) for k, v in extra.items() if k.encode() not in present
                ]
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_headers)
        except Exception:
            log.exception("unhandled error on %s %s", scope["method"], scope["path"])
            if not status["started"]:
                body = {"detail": "internal error", "request_id": request_id}
                await send_json(send, 500, body, extra)
        finally:
            log.info(
                "%s %s -> %s in %.1f ms",
                scope["method"],
                scope["path"],
                status["code"],
                1000 * (time.perf_counter() - started),
            )
            request_id_var.reset(token)


class _BodyTooLarge(Exception):
    pass


class BodySizeLimitMiddleware:
    """Reject request bodies above ``max_bytes`` *before* they are fully read (413)."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle one ASGI connection."""
        if scope["type"] != "http" or scope["method"] not in MUTATING_METHODS:
            await self.app(scope, receive, send)
            return
        problem = self._declared_problem(_header(scope, b"content-length"))
        if problem is not None:
            await send_json(send, *problem)
            return
        received = 0
        exceeded = False

        async def limited_receive() -> Message:
            nonlocal received, exceeded
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    exceeded = True
                    raise _BodyTooLarge
            return message

        async def guarded_send(message: Message) -> None:
            if not exceeded:  # the app's own error response is replaced by the 413
                await send(message)

        with contextlib.suppress(_BodyTooLarge):
            await self.app(scope, limited_receive, guarded_send)
        if exceeded:
            await send_json(send, 413, self._too_large())

    def _declared_problem(self, declared: str | None) -> tuple[int, dict[str, str]] | None:
        """Status and body when the ``Content-Length`` header alone justifies a rejection."""
        if declared is None:
            return None
        if not declared.isdigit():
            return 400, self._body("invalid Content-Length header")
        if int(declared) > self.max_bytes:
            return 413, self._too_large()
        return None

    def _too_large(self) -> dict[str, str]:
        return self._body(f"request body larger than {self.max_bytes:,} bytes")

    @staticmethod
    def _body(detail: str) -> dict[str, str]:
        return {"detail": detail, "request_id": request_id_var.get()}


class RateLimitMiddleware:
    """Token bucket per client address for mutating methods (429 with ``Retry-After``)."""

    def __init__(self, app: ASGIApp, limiter: TokenBucketLimiter) -> None:
        self.app = app
        self.limiter = limiter

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle one ASGI connection."""
        if scope["type"] != "http" or scope["method"] not in MUTATING_METHODS:
            await self.app(scope, receive, send)
            return
        client = scope.get("client")
        key = client[0] if client else "unknown"
        wait = self.limiter.acquire(key)
        if wait > 0:
            body = {"detail": "too many requests", "request_id": request_id_var.get()}
            await send_json(send, 429, body, {"retry-after": str(max(1, round(wait)))})
            return
        await self.app(scope, receive, send)
