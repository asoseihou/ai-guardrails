"""LLM providers (adapters). Real SDKs are imported lazily, only when a provider is used."""

from __future__ import annotations

from pathlib import Path

from docai.extract.providers.base import (
    ExtractionContext,
    ExtractionResult,
    LLMProvider,
    Pricing,
    ProviderError,
    ProviderUnavailable,
    Usage,
)
from docai.extract.providers.simulated import (
    ErrorRates,
    SimulatedLLMProvider,
    TruthStore,
    doc_id_from_filename,
)
from docai.settings import Settings

PROVIDER_NAMES = ("simulated", "anthropic", "gemini", "bedrock", "auto")


def detect_real_provider(settings: Settings) -> str | None:
    """Name of the first real provider whose credentials are configured, if any."""
    if settings.anthropic_api_key is not None and settings.anthropic_model:
        return "anthropic"
    if settings.gemini_api_key is not None:
        return "gemini"
    if settings.bedrock_model_id:
        return "bedrock"
    return None


def get_provider(
    name: str,
    settings: Settings,
    *,
    truth_roots: list[Path] | None = None,
    rates: ErrorRates | None = None,
    seed: int = 7,
) -> LLMProvider:
    """Build a provider by name. ``auto`` = first real provider with credentials, else simulated.

    Real providers receive their model, key, output budget and timeout from ``settings``.
    """
    if name == "auto":
        name = detect_real_provider(settings) or "simulated"
    if name == "simulated":
        return SimulatedLLMProvider(TruthStore(truth_roots or []), rates, seed=seed)
    max_tokens, timeout_s = settings.llm_max_output_tokens, settings.llm_timeout_s
    if name == "anthropic":
        from docai.extract.providers.anthropic import AnthropicProvider

        key = settings.anthropic_api_key
        return AnthropicProvider(
            settings.anthropic_model,
            api_key=key.get_secret_value() if key else None,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )
    if name == "gemini":
        from docai.extract.providers.gemini import GeminiProvider

        key = settings.gemini_api_key
        return GeminiProvider(
            settings.gemini_model,
            api_key=key.get_secret_value() if key else None,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )
    if name == "bedrock":
        from docai.extract.providers.bedrock import BedrockProvider

        return BedrockProvider(
            settings.bedrock_model_id,
            region=settings.aws_region,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )
    raise ValueError(f"unknown provider {name!r}; choose one of {PROVIDER_NAMES}")


__all__ = [
    "PROVIDER_NAMES",
    "ErrorRates",
    "ExtractionContext",
    "ExtractionResult",
    "LLMProvider",
    "Pricing",
    "ProviderError",
    "ProviderUnavailable",
    "SimulatedLLMProvider",
    "TruthStore",
    "Usage",
    "detect_real_provider",
    "doc_id_from_filename",
    "get_provider",
]
