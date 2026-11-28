"""Composition root: builds the shared, settings-driven components once per process.

Interfaces (API, MCP, CLI) ask this module for ready-made objects instead of
wiring adapters themselves. ``reset()`` forgets everything (tests, new settings).
"""

from __future__ import annotations

from functools import lru_cache

from docai.classify import DocumentClassifier, load_classifier
from docai.extract.pdf import PdfLimits
from docai.extract.providers import (
    ErrorRates,
    LLMProvider,
    SimulatedLLMProvider,
    TruthStore,
    get_provider,
)
from docai.pipeline import Pipeline
from docai.rag import KnowledgeBase
from docai.review import ReviewService
from docai.settings import Settings, get_settings
from docai.storage import ReviewRepository, RunRepository


@lru_cache(maxsize=1)
def get_truth_store() -> TruthStore:
    """Ground truth of the synthetic corpus and samples (feeds the simulated LLM)."""
    s = get_settings()
    return TruthStore([d for d in (s.data_dir, s.samples_dir) if d.exists()])


@lru_cache(maxsize=1)
def get_registry() -> dict[str, str]:
    """CNPJ -> registered company name (stands for supplier master data; synthetic here)."""
    return {party["cnpj"]: party["name"] for party in get_truth_store().parties()}


@lru_cache(maxsize=1)
def get_classifier() -> DocumentClassifier:
    """Trained PyTorch classifier, or the keyword fallback."""
    return load_classifier(get_settings().model_path)


@lru_cache(maxsize=1)
def get_provider_instance() -> LLMProvider:
    """The configured LLM provider."""
    settings = get_settings()
    if settings.provider == "simulated":
        return SimulatedLLMProvider(get_truth_store())
    return get_provider(settings.provider, settings)


def pdf_limits(settings: Settings) -> PdfLimits:
    """Limits applied to untrusted PDFs (uploads, MCP tool calls)."""
    return PdfLimits(
        max_bytes=settings.max_upload_bytes,
        max_pages=settings.max_pdf_pages,
        max_chars=settings.max_text_chars,
        timeout_s=settings.pdf_timeout_s,
        isolate=settings.pdf_isolation,
    )


def get_pipeline(noise: float | None = None) -> Pipeline:
    """Pipeline with the configured provider; ``noise`` rescales simulated error rates."""
    settings = get_settings()
    provider = get_provider_instance()
    if noise is not None and isinstance(provider, SimulatedLLMProvider):
        provider = SimulatedLLMProvider(get_truth_store(), ErrorRates().scaled(noise))
    return Pipeline(
        provider,
        get_classifier(),
        settings.guardrails,
        limits=pdf_limits(settings),
        xml_first=settings.xml_first,
        max_xml_bytes=settings.max_xml_bytes,
        registry=get_registry(),
    )


def get_review_service() -> ReviewService:
    """Review queue backed by the configured SQLite file."""
    settings = get_settings()
    return ReviewService(
        ReviewRepository(settings.db_path), max_pending=settings.review_max_pending
    )


def get_run_repository() -> RunRepository:
    """History of evaluation runs (dashboard)."""
    return RunRepository(get_settings().db_path)


@lru_cache(maxsize=1)
def get_knowledge_base() -> KnowledgeBase:
    """BM25 index over the markdown notes."""
    return KnowledgeBase.from_dir(get_settings().knowledge_dir)


def reset() -> None:
    """Forget cached components and settings (tests, settings changes)."""
    for fn in (
        get_settings,
        get_truth_store,
        get_registry,
        get_classifier,
        get_provider_instance,
        get_knowledge_base,
    ):
        fn.cache_clear()
