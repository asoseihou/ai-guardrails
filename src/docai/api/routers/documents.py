"""Document extraction, the knowledge base and the bundled samples."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from docai import __version__, container
from docai.extract.pdf import PDF_MAGIC
from docai.extract.providers import detect_real_provider, get_provider
from docai.observability import short_hash
from docai.settings import get_settings

log = logging.getLogger(__name__)
router = APIRouter()


class AskRequest(BaseModel):
    """A question for the fiscal-rules knowledge base."""

    question: str = Field(min_length=3, max_length=500)
    k: int = Field(3, ge=1, le=10)
    use_llm: bool = Field(False, description="synthesize with a real provider when configured")


def _sample_files() -> dict[str, Path]:
    pdf_dir = get_settings().samples_dir / "pdf"
    return {p.name: p for p in sorted(pdf_dir.glob("*.pdf"))} if pdf_dir.is_dir() else {}


def _sample_xml(pdf_name: str) -> Path | None:
    xml = get_settings().samples_dir / "xml" / (pdf_name.removesuffix(".pdf") + ".xml")
    return xml if xml.is_file() else None


@router.get("/health")
def health() -> dict[str, Any]:
    """Liveness plus which provider, classifier and guardrail preset are loaded."""
    return {
        "status": "ok",
        "version": __version__,
        "provider": container.get_provider_instance().describe(),
        "classifier": container.get_classifier().name,
        "guardrails": get_settings().guardrails,
        "knowledge_chunks": len(container.get_knowledge_base().chunks),
        "ground_truth_documents": len(container.get_truth_store()),
    }


def safe_filename(name: str | None) -> str | None:
    """Client-supplied file name reduced to a short, printable base name (it reaches the prompt)."""
    if not name:
        return None
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    cleaned = "".join(c for c in base if c.isprintable() and c not in "<>\"'`")
    return cleaned[:128] or None


def read_pdf_upload(file: UploadFile, max_bytes: int) -> bytes:
    """Read an uploaded PDF up to ``max_bytes`` (413 above it, 415 when it is not a PDF)."""
    data = file.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(413, f"file larger than {max_bytes:,} bytes")
    if not data.startswith(PDF_MAGIC):
        raise HTTPException(415, "only PDF files are accepted")
    return data


def read_xml_upload(file: UploadFile | None, max_bytes: int) -> bytes | None:
    """Read the optional XML upload (413 above ``max_bytes``); parsing happens later, safely."""
    if file is None:
        return None
    data = file.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(413, f"XML larger than {max_bytes:,} bytes")
    return data or None


@router.post("/extract")
def extract(
    file: Annotated[UploadFile, File(description="a PDF document")],
    xml: Annotated[
        UploadFile | None, File(description="optional NF-e / NFS-e XML of the same document")
    ] = None,
    noise: Annotated[float | None, Form(description="simulated-LLM error-rate multiplier")] = None,
) -> dict[str, Any]:
    """Classify a PDF, extract its fields and run the guardrails (read-only: nothing is stored).

    With an XML, the XML-first policy applies: its values are used directly
    (no LLM call) or, if XML-first is disabled, they cross-check the LLM output.
    """
    settings = get_settings()
    data = read_pdf_upload(file, settings.max_upload_bytes)
    xml_data = read_xml_upload(xml, settings.max_xml_bytes)
    if noise is not None and not 0 <= noise <= 5:
        raise HTTPException(422, "noise must be between 0 and 5")
    log.info(
        "extract document=%s bytes=%d xml=%s",
        short_hash(data),
        len(data),
        short_hash(xml_data) if xml_data else "-",
    )
    result = container.get_pipeline(noise).run_pdf(
        data, filename=safe_filename(file.filename), xml_data=xml_data
    )
    return result.to_dict()


@router.post("/ask")
def ask(request: AskRequest) -> dict[str, Any]:
    """Answer from the knowledge base, citing the passages used."""
    settings = get_settings()
    provider = None
    if request.use_llm and (name := detect_real_provider(settings)):
        provider = get_provider(name, settings)
    return container.get_knowledge_base().ask(request.question, request.k, provider).to_dict()


@router.get("/samples")
def list_samples() -> list[dict[str, Any]]:
    """Synthetic sample PDFs bundled with the repository (and whether an XML comes with them)."""
    return [{"name": name, "has_xml": _sample_xml(name) is not None} for name in _sample_files()]


@router.get("/samples/{name}", response_class=FileResponse)
def get_sample(name: str) -> FileResponse:
    """Download one bundled sample (only names from the listing are served)."""
    files = _sample_files()
    if name not in files:
        raise HTTPException(404, "unknown sample")
    return FileResponse(files[name], media_type="application/pdf", filename=name)


@router.get("/samples/{name}/xml", response_class=FileResponse)
def get_sample_xml(name: str) -> FileResponse:
    """Download the XML that comes with a bundled sample, if there is one."""
    xml = _sample_xml(name) if name in _sample_files() else None
    if xml is None:
        raise HTTPException(404, "no XML for this sample")
    return FileResponse(xml, media_type="application/xml", filename=xml.name)
