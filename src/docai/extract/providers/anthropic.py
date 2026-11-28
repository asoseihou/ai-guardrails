"""Anthropic provider (Messages API, forced tool use for structured output).

Optional dependency: ``pip install 'ai-guardrails[anthropic]'``. The API key and
model come from :class:`docai.settings.Settings` (``ANTHROPIC_API_KEY``,
``DOCAI_ANTHROPIC_MODEL``) through :func:`docai.extract.providers.get_provider`.
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel

from docai.domain.schemas import doc_type_of_schema
from docai.extract.prompts import SYSTEM_PROMPT
from docai.extract.providers.base import (
    ExtractionContext,
    ExtractionResult,
    FinishReason,
    LLMProvider,
    Pricing,
    PromptVariant,
    ProviderUnavailable,
    Usage,
    build_user_prompt,
    inline_json_schema,
)

TOOL_NAME = "record_extraction"


class AnthropicProvider(LLMProvider):
    """Anthropic models through the Messages API, structured output via a forced tool call."""

    name = "anthropic"

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        max_tokens: int = 2048,
        timeout_s: float = 60.0,
        client: Any = None,
        pricing: Pricing | None = None,
    ) -> None:
        if not model:
            raise ProviderUnavailable("set DOCAI_ANTHROPIC_MODEL to an Anthropic model id")
        super().__init__(model, pricing)
        self._api_key = api_key
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self._client = client

    @property
    def client(self) -> Any:
        """The SDK client, created on first use (the SDK is an optional dependency)."""
        if self._client is None:
            try:
                import anthropic  # lazy: optional dependency
            except ImportError as exc:
                raise ProviderUnavailable(
                    "install the extra: pip install 'ai-guardrails[anthropic]'"
                ) from exc
            if not self._api_key:
                raise ProviderUnavailable("ANTHROPIC_API_KEY is not set")
            self._client = anthropic.Anthropic(api_key=self._api_key, timeout=self.timeout_s)
        return self._client

    def complete(
        self,
        text: str,
        schema: type[BaseModel],
        context: ExtractionContext | None,
        variant: PromptVariant = PromptVariant.STANDARD,
    ) -> ExtractionResult:
        """One forced-tool call; ``stop_reason == "max_tokens"`` means truncated."""
        doc_type = doc_type_of_schema(schema)
        tool = {
            "name": TOOL_NAME,
            "description": "Record the fields extracted from the document.",
            "input_schema": inline_json_schema(schema.model_json_schema()),
        }
        started = time.perf_counter()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            tools=[tool],
            tool_choice={"type": "tool", "name": TOOL_NAME},
            messages=[
                {"role": "user", "content": build_user_prompt(text, doc_type, context, variant)}
            ],
        )
        latency = time.perf_counter() - started
        data = next(
            (b.input for b in response.content if getattr(b, "type", None) == "tool_use"), None
        )
        finish = (
            FinishReason.LENGTH
            if getattr(response, "stop_reason", None) == "max_tokens"
            else FinishReason.STOP
        )
        error = None
        if finish == FinishReason.LENGTH:
            data, error = None, "the answer was cut at max_tokens"
        elif not isinstance(data, dict):
            data, error = None, "the response has no tool_use block"
        usage = Usage(
            input_tokens=int(response.usage.input_tokens),
            output_tokens=int(response.usage.output_tokens),
            latency_s=round(latency, 3),
        )
        usage.cost_usd = self.pricing.cost(usage.input_tokens, usage.output_tokens)
        return ExtractionResult(
            data=data,
            usage=usage,
            provider=self.name,
            model=self.model,
            finish_reason=finish,
            error=error,
        )

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        """Free text for the optional RAG answer synthesis."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system or "You answer concisely and cite sources as [n].",
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(getattr(b, "text", "") for b in response.content).strip()
