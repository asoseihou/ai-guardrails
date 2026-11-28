"""Logging with request correlation ids, optional JSON output and secret redaction.

* Every log record carries ``request_id`` (from a context variable set by the API
  middleware; ``-`` outside a request).
* ``configure_logging(json_output=True)`` emits one JSON object per line.
* :class:`RedactingFilter` masks configured secrets and bearer tokens in every
  message, as a last line of defence: the code never logs keys or document text
  on purpose, only short hashes (:func:`short_hash`).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from contextvars import ContextVar
from datetime import UTC, datetime

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

_BEARER = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+")
_RESERVED = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime", "request_id"}


def short_hash(data: bytes | str) -> str:
    """First 12 hex digits of the SHA-256: identifies a document without revealing it."""
    raw = data.encode() if isinstance(data, str) else data
    return hashlib.sha256(raw).hexdigest()[:12]


class RequestIdFilter(logging.Filter):
    """Attach the current request id to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add ``record.request_id``; never drops a record."""
        record.request_id = request_id_var.get()
        return True


class RedactingFilter(logging.Filter):
    """Replace secrets and ``Bearer <token>`` values in log messages with ``[REDACTED]``."""

    def __init__(self, secrets: list[str] | None = None) -> None:
        super().__init__()
        self.secrets = [s for s in (secrets or []) if len(s) >= 8]

    def redact(self, text: str) -> str:
        """Masked copy of ``text``."""
        for secret in self.secrets:
            text = text.replace(secret, "[REDACTED]")
        return _BEARER.sub(r"\1[REDACTED]", text)

    def filter(self, record: logging.LogRecord) -> bool:
        """Freeze the formatted message with secrets masked; never drops a record."""
        message = record.getMessage()
        redacted = self.redact(message)
        if redacted != message:
            record.msg, record.args = redacted, None
        if record.exc_info and not record.exc_text:
            record.exc_text = self.redact(logging.Formatter().formatException(record.exc_info))
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line: time, level, logger, request id, message and extras."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize ``record``."""
        payload: dict[str, object] = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_text:
            payload["exception"] = record.exc_text
        elif record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


TEXT_FORMAT = "%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s"


def configure_logging(
    level: str = "INFO", *, json_output: bool = False, secrets: list[str] | None = None
) -> None:
    """Install one handler on the root logger (idempotent: replaces a previous one)."""
    handler = logging.StreamHandler()
    handler.set_name("docai")
    handler.addFilter(RequestIdFilter())
    handler.addFilter(RedactingFilter(secrets))
    handler.setFormatter(JsonFormatter() if json_output else logging.Formatter(TEXT_FORMAT))
    root = logging.getLogger()
    for existing in list(root.handlers):
        if existing.get_name() == "docai":
            root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)
