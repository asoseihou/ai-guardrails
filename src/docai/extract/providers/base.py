"""Provider interface shared by the simulated and the real LLM backends."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from docai.extract import prompts


class ProviderError(RuntimeError):
    """The provider call failed (transport, auth, quota) or its output is unusable."""


class ProviderUnavailable(ProviderError):
    """The provider's SDK or credentials are missing."""


@dataclass(frozen=True)
class Pricing:
    """USD per million tokens. Used only for cost *estimates*."""

    input_per_mtok: float
    output_per_mtok: float
    label: str = ""

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimated USD cost of one call."""
        return (input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok) / 1e6


ILLUSTRATIVE_PRICING = Pricing(3.0, 15.0, "illustrative: $3 / 1M input, $15 / 1M output tokens")


class FinishReason(StrEnum):
    """Why the model stopped writing, normalized across providers."""

    STOP = "stop"  # finished normally
    LENGTH = "length"  # hit the output-token budget: the answer is truncated
    OTHER = "other"


class PromptVariant(StrEnum):
    """``summary`` is the compact retry used after a truncated answer."""

    STANDARD = "standard"
    SUMMARY = "summary"


@dataclass
class Usage:
    """Tokens, latency and estimated cost of one or more calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    simulated: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Plain dict."""
        return asdict(self)

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            round(self.latency_s + other.latency_s, 3),
            self.cost_usd + other.cost_usd,
            self.simulated and other.simulated,
        )


@dataclass(frozen=True)
class ExtractionContext:
    """Metadata that travels with the text (never part of the document itself)."""

    doc_id: str | None = None
    filename: str | None = None


@dataclass
class ExtractionResult:
    """One provider call: the parsed fields (``None`` when unusable) and how it ended."""

    data: dict[str, Any] | None
    usage: Usage
    provider: str
    model: str
    meta: dict[str, Any] = field(default_factory=dict)
    finish_reason: FinishReason = FinishReason.STOP
    error: str | None = None

    @property
    def truncated(self) -> bool:
        """The answer was cut at the output limit (or is not valid JSON)."""
        return self.finish_reason == FinishReason.LENGTH or self.data is None

    @property
    def usable(self) -> bool:
        """A complete JSON object came back."""
        return not self.truncated


def build_user_prompt(
    text: str,
    doc_type: str,
    context: ExtractionContext | None,
    variant: PromptVariant = PromptVariant.STANDARD,
) -> str:
    """User message for one extraction (see :mod:`docai.extract.prompts`)."""
    filename = context.filename if context else None
    return prompts.build_user_prompt(
        text, doc_type, filename, summary=variant == PromptVariant.SUMMARY
    )


def inline_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve local ``$ref``/``$defs`` so providers with partial JSON-schema support work."""
    defs = schema.get("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                return resolve(defs[ref.split("/")[-1]])
            return {k: resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    resolved: dict[str, Any] = resolve(schema)
    return resolved


def estimate_tokens(text: str) -> int:
    """Rough token estimate (≈4 characters per token)."""
    return max(1, round(len(text) / 4))


class LLMProvider(ABC):
    """Structured extraction: ``extract(text, schema) -> dict``."""

    name: str = "base"
    simulated: bool = False

    def __init__(self, model: str, pricing: Pricing | None = None) -> None:
        self.model = model
        self.pricing = pricing or ILLUSTRATIVE_PRICING

    @abstractmethod
    def complete(
        self,
        text: str,
        schema: type[BaseModel],
        context: ExtractionContext | None,
        variant: PromptVariant = PromptVariant.STANDARD,
    ) -> ExtractionResult:
        """Run one extraction and report usage and the finish reason (never retries)."""

    def extract(
        self,
        text: str,
        schema: type[BaseModel],
        *,
        context: ExtractionContext | None = None,
    ) -> dict[str, Any]:
        """Parsed fields of one call; ``ProviderError`` when the answer is unusable."""
        result = self.complete(text, schema, context)
        if result.data is None or result.truncated:
            raise ProviderError(result.error or "the model's answer was truncated")
        return result.data

    def extract_with_usage(
        self,
        text: str,
        schema: type[BaseModel],
        *,
        context: ExtractionContext | None = None,
        variant: PromptVariant = PromptVariant.STANDARD,
    ) -> ExtractionResult:
        """One call with usage and finish reason (see ``docai.pipeline.recovery``)."""
        return self.complete(text, schema, context, variant)

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        """Free-text generation (used for optional RAG answer synthesis)."""
        raise NotImplementedError(f"{self.name} does not support free-text generation")

    def describe(self) -> dict[str, Any]:
        """Name, model and whether it is simulated (shown by ``/health``)."""
        return {"name": self.name, "model": self.model, "simulated": self.simulated}


def parse_json_object(raw: str | None) -> tuple[dict[str, Any] | None, str | None]:
    """``(object, None)`` for a JSON object, else ``(None, reason)``."""
    try:
        data: object = json.loads(raw or "")
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc.msg} at character {exc.pos}"
    if not isinstance(data, dict):
        return None, "JSON that is not an object"
    return data, None
