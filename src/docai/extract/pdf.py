"""PDF → text, with limits for untrusted input and an on-disk cache for the trusted corpus.

Untrusted uploads go through :func:`read_pdf_text`, which enforces a size, page
and text-length limit and a wall-clock timeout. With ``isolate=True`` (the API
default) the parser runs in a separate Python process (``python -m
docai.extract.pdf``) that is killed on timeout, has capped CPU time and memory,
and does not inherit the parent's environment (no API keys). Encrypted and
malformed files are rejected with a clear, client-safe message
(:class:`PdfRejected`).
"""

from __future__ import annotations

import io
import json
import logging
import os

# subprocess (B404/B603): runs this package's own parser with a fixed argv, no shell.
import subprocess  # nosec B404
import sys
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

PDF_MAGIC = b"%PDF-"
PAGE_SEPARATOR = "\n\f\n"  # pages stay addressable (see docai.domain.layout.split_pages)
WORKER_MEMORY_BYTES = 1024 * 1024 * 1024  # address-space cap for the parser process
_ENCRYPTION_ERRORS = ("PDFPasswordIncorrect", "PDFEncryptionError", "PDFEncryptionWarning")


class PdfRejected(ValueError):
    """The PDF violates a limit or cannot be read. The message is safe to show to clients."""


@dataclass(frozen=True)
class PdfLimits:
    """Limits applied to untrusted PDFs."""

    max_bytes: int = 10 * 1024 * 1024
    max_pages: int = 20
    max_chars: int = 200_000
    timeout_s: float = 15.0
    isolate: bool = True


@dataclass(frozen=True)
class PdfText:
    """The text layer of a PDF and its page count."""

    text: str
    pages: int


def _extract(data: bytes, max_pages: int, max_chars: int) -> PdfText:
    """Parse ``data`` and return its text; raise :class:`PdfRejected` on any problem."""
    import pdfplumber

    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            pages = len(pdf.pages)
            if pages > max_pages:
                raise PdfRejected(f"the PDF has {pages} pages; the limit is {max_pages}")
            parts: list[str] = []
            total = 0
            for page in pdf.pages:
                chunk = page.extract_text() or ""
                total += len(chunk) + 1
                if total > max_chars:
                    raise PdfRejected(f"the text layer exceeds {max_chars:,} characters")
                parts.append(chunk)
    except PdfRejected:
        raise
    except Exception as exc:  # pdfminer raises many unrelated exception types
        names = {type(exc).__name__, type(exc.__cause__).__name__, type(exc.__context__).__name__}
        if names & set(_ENCRYPTION_ERRORS) or "encrypt" in str(exc).lower():
            raise PdfRejected("encrypted or password-protected PDFs are not supported") from exc
        raise PdfRejected("the file could not be parsed as a PDF") from exc
    return PdfText(PAGE_SEPARATOR.join(parts), pages)


def _worker_env() -> dict[str, str]:
    # The parser never sees the parent's environment (API keys, tokens): only the
    # import path it needs to find pdfplumber and this package.
    return {
        "PYTHONPATH": os.pathsep.join(p for p in sys.path if p),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONSAFEPATH": "1",
        "PYTHONIOENCODING": "utf-8",
    }


def _extract_isolated(data: bytes, limits: PdfLimits) -> PdfText:
    command = [
        sys.executable,
        "-m",
        __name__,
        str(limits.max_pages),
        str(limits.max_chars),
        str(int(limits.timeout_s) + 1),
    ]
    try:
        done = subprocess.run(  # noqa: S603  # nosec B603
            command,
            input=data,
            capture_output=True,
            timeout=limits.timeout_s,
            env=_worker_env(),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:  # the child is killed by subprocess.run
        raise PdfRejected(f"PDF processing timed out after {limits.timeout_s:g} s") from exc
    try:
        reply = json.loads(done.stdout)
    except ValueError:
        log.warning(
            "pdf worker exited with %s: %s",
            done.returncode,
            done.stderr[-300:].decode(errors="replace"),
        )
        raise PdfRejected("the file could not be parsed as a PDF") from None
    if reply.get("status") != "ok":
        raise PdfRejected(str(reply.get("reason", "the file could not be parsed as a PDF")))
    return PdfText(str(reply["text"]), int(reply["pages"]))


def _limit_resources(cpu_seconds: int) -> None:
    """Best-effort memory and CPU caps for the worker process (POSIX only)."""
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows
        return
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    memory = WORKER_MEMORY_BYTES
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))


def _worker_main(argv: list[str]) -> int:
    """Child-process entry point: PDF bytes on stdin, one JSON reply on stdout."""
    max_pages, max_chars, cpu_seconds = (int(a) for a in argv[:3])
    _limit_resources(cpu_seconds)
    data = sys.stdin.buffer.read()
    try:
        result = _extract(data, max_pages, max_chars)
        reply: dict[str, object] = {"status": "ok", "text": result.text, "pages": result.pages}
    except PdfRejected as exc:
        reply = {"status": "rejected", "reason": str(exc)}
    except MemoryError:
        reply = {"status": "rejected", "reason": "the PDF needs too much memory to parse"}
    sys.stdout.write(json.dumps(reply))
    return 0


def read_pdf_text(data: bytes, limits: PdfLimits | None = None) -> PdfText:
    """Text layer of an untrusted PDF, enforcing ``limits`` (see module docstring)."""
    limits = limits or PdfLimits()
    if len(data) > limits.max_bytes:
        raise PdfRejected(f"the file is larger than {limits.max_bytes:,} bytes")
    if not data.startswith(PDF_MAGIC):
        raise PdfRejected("the file is not a PDF (missing %PDF- header)")
    if limits.isolate:
        return _extract_isolated(data, limits)
    return _extract(data, limits.max_pages, limits.max_chars)


def pdf_to_text(source: Path | str | bytes) -> str:
    """Text layer of a *trusted* PDF (pages separated by form feeds); no limits applied."""
    import pdfplumber

    handle = io.BytesIO(source) if isinstance(source, bytes) else str(source)
    with pdfplumber.open(handle) as pdf:
        return PAGE_SEPARATOR.join(page.extract_text() or "" for page in pdf.pages)


def cached_pdf_text(pdf_path: Path, cache_dir: Path | None) -> str:
    """Extract text once and reuse it on later runs (keyed by file name and mtime)."""
    if cache_dir is None:
        return pdf_to_text(pdf_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stamp = int(pdf_path.stat().st_mtime)
    cache_file = cache_dir / f"{pdf_path.stem}.{stamp}.txt"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")
    text = pdf_to_text(pdf_path)
    cache_file.write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    sys.exit(_worker_main(sys.argv[1:]))
