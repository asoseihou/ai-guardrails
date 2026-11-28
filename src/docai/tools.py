"""Plain functions behind the MCP tools (and the ``docai validate`` command).

File-reading tools are sandboxed: only regular files under the configured root
(``DOCAI_MCP_ROOT``, default: the bundled samples folder), with the right suffix,
size and magic bytes, are read. See :func:`read_sandboxed_file`.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

from docai.domain.access_key import (
    DOC_TYPE_BY_MODEL,
    key_check_digit,
    normalize_key,
    parse_access_key,
)
from docai.domain.cnpj import cnpj_check_digits, format_cnpj, is_valid_cnpj, normalize_cnpj
from docai.extract.pdf import PDF_MAGIC, PdfRejected
from docai.extract.xml_reader import XmlRejected
from docai.settings import get_settings

MAX_QUERY_CHARS = 500


def validate_cnpj(cnpj: str) -> dict[str, Any]:
    """Check a CNPJ's shape and modulo-11 check digits (numeric or alphanumeric)."""
    value = normalize_cnpj(cnpj)
    result: dict[str, Any] = {"input": cnpj, "normalized": value, "valid": False}
    if len(value) != 14:
        result["reason"] = f"a CNPJ has 14 positions, got {len(value)}"
        return result
    try:
        expected = cnpj_check_digits(value[:12])
    except ValueError as exc:
        result["reason"] = str(exc)
        return result
    result.update(
        valid=is_valid_cnpj(value),
        formatted=format_cnpj(value),
        check_digits=value[12:],
        expected_check_digits=expected,
    )
    if not result["valid"]:
        result["reason"] = (
            "repeated-digit placeholder" if value[12:] == expected else "check digits do not match"
        )
    return result


def validate_access_key(access_key: str) -> dict[str, Any]:
    """Check a 44-digit NF-e/CT-e access key and decode its fields."""
    key = normalize_key(access_key)
    result: dict[str, Any] = {"input": access_key, "normalized": key, "valid": False}
    if len(key) != 44:
        result["reason"] = f"an access key has 44 digits, got {len(key)}"
        return result
    expected = key_check_digit(key[:43])
    parsed = parse_access_key(key)
    result.update(
        valid=key[-1] == expected,
        check_digit=key[-1],
        expected_check_digit=expected,
        doc_type=DOC_TYPE_BY_MODEL.get(parsed.model, f"model {parsed.model}"),
        fields={
            "cuf": parsed.cuf,
            "year_month": f"{parsed.year}-{parsed.month:02d}",
            "issuer_cnpj": parsed.cnpj,
            "issuer_cnpj_valid": is_valid_cnpj(parsed.cnpj),
            "model": parsed.model,
            "series": parsed.series,
            "number": parsed.number,
            "emission_type": parsed.emission_type,
            "code": parsed.code,
        },
    )
    if not result["valid"]:
        result["reason"] = "check digit (mod 11) does not match"
    return result


class ToolInputError(ValueError):
    """A tool argument was refused; the message is safe to return to the MCP client."""


MAX_PATH_CHARS = 1024


def read_sandboxed_file(
    path_arg: str, root: Path, *, suffix: str, max_bytes: int, magic: bytes
) -> bytes:
    """Read a regular ``suffix`` file under ``root`` (size and magic bytes are checked).

    The path is resolved first (``..`` and symlinks included) and must stay inside
    ``root``; the file is then opened without following symlinks and re-checked
    through its descriptor, so a swap between the check and the read is caught.
    """
    if not path_arg or len(path_arg) > MAX_PATH_CHARS or "\x00" in path_arg:
        raise ToolInputError("invalid path")
    root = root.resolve()
    candidate = Path(path_arg)
    candidate = candidate if candidate.is_absolute() else root / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        raise ToolInputError("file not found under the allowed root") from None
    if not resolved.is_relative_to(root):
        raise ToolInputError("path must be inside the allowed root (DOCAI_MCP_ROOT)")
    if resolved.suffix.lower() != suffix:
        raise ToolInputError(f"only {suffix} files are accepted")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(resolved, flags)
    except OSError:
        raise ToolInputError("file cannot be opened") from None
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_size > max_bytes:
        os.close(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ToolInputError("not a regular file")
        raise ToolInputError(f"file larger than {max_bytes:,} bytes")
    with os.fdopen(fd, "rb") as handle:
        data = handle.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ToolInputError(f"file larger than {max_bytes:,} bytes")
    if not data.startswith(magic):
        raise ToolInputError(f"file content is not a {suffix} document")
    return data


def extract_document(pdf_path: str, xml_path: str | None = None) -> dict[str, Any]:
    """Run the full pipeline on a PDF (and optional XML) inside ``DOCAI_MCP_ROOT``."""
    from docai.container import get_pipeline

    settings = get_settings()
    data = read_sandboxed_file(
        pdf_path,
        settings.mcp_root,
        suffix=".pdf",
        max_bytes=settings.max_upload_bytes,
        magic=PDF_MAGIC,
    )
    xml_data = None
    if xml_path:
        xml_data = read_sandboxed_file(
            xml_path, settings.mcp_root, suffix=".xml", max_bytes=settings.max_xml_bytes, magic=b"<"
        )
    pipeline = get_pipeline()
    try:
        result = pipeline.run_pdf(data, filename=Path(pdf_path).name, xml_data=xml_data)
    except (PdfRejected, XmlRejected) as exc:
        raise ToolInputError(str(exc)) from exc
    return result.to_dict()


def search_fiscal_rules(query: str, k: int = 3) -> list[dict[str, Any]]:
    """BM25 search over the fiscal-rules notes; returns cited passages."""
    from docai.container import get_knowledge_base

    if not query.strip() or len(query) > MAX_QUERY_CHARS:
        raise ToolInputError(f"query must have 1-{MAX_QUERY_CHARS} characters")
    return [p.to_dict() for p in get_knowledge_base().search(query, max(1, min(k, 10)))]
