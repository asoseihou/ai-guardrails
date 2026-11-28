"""Google Gemini provider (google-genai SDK, JSON-schema constrained output).

Optional dependency: ``pip install 'ai-guardrails[gemini]'``. The API key and model
come from :class:`docai.settings.Settings` (``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``,
``DOCAI_GEMINI_MODEL``) through :func:`docai.extract.providers.get_provider`.
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
    parse_json_object,
)

DEFAULT_MODEL = "gemini-2.5-flash"


def _finish_reason(response: Any) -> FinishReason:
    candidates = getattr(response, "candidates", None) or []
    reason = getattr(candidates[0], "finish_reason", None) if candidates else None
    name = str(getattr(reason, "name", reason) or "")
    return FinishReason.LENGTH if name.endswith("MAX_TOKENS") else FinishReason.STOP


class GeminiProvider(LLMProvider):
    """Gemini through ``google-genai`` with a JSON-schema constrained response."""

    name = "gemini"

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
        super().__init__(model or DEFAULT_MODEL, pricing)
        self._api_key = api_key
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self._client = client

    @property
    def client(self) -> Any:
        """The SDK client, created on first use (the SDK is an optional dependency)."""
        if self._client is None:
            try:
                from google import genai  # lazy: optional dependency
            except ImportError as exc:
                raise ProviderUnavailable(
                    "install the extra: pip install 'ai-guardrails[gemini]'"
                ) from exc
            if not self._api_key:
                raise ProviderUnavailable("GEMINI_API_KEY / GOOGLE_API_KEY is not set")
            self._client = genai.Client(
                api_key=self._api_key, http_options={"timeout": int(self.timeout_s * 1000)}
            )
        return self._client

    def complete(
        self,
        text: str,
        schema: type[BaseModel],
        context: ExtractionContext | None,
        variant: PromptVariant = PromptVariant.STANDARD,
    ) -> ExtractionResult:
        """One JSON-schema constrained call; ``MAX_TOKENS`` means truncated."""
        doc_type = doc_type_of_schema(schema)
        started = time.perf_counter()
        response = self.client.models.generate_content(
            model=self.model,
            contents=build_user_prompt(text, doc_type, context, variant),
            config={
                "system_instruction": SYSTEM_PROMPT,
                "temperature": 0,
                "max_output_tokens": self.max_tokens,
                "response_mime_type": "application/json",
                "response_json_schema": inline_json_schema(schema.model_json_schema()),
            },
        )
        latency = time.perf_counter() - started
        data, error = parse_json_object(response.text)
        finish = _finish_reason(response)
        if finish == FinishReason.LENGTH:
            data, error = None, "the answer was cut at max_output_tokens"
        meta = getattr(response, "usage_metadata", None)
        output_tokens = int(getattr(meta, "candidates_token_count", 0) or 0)
        output_tokens += int(getattr(meta, "thoughts_token_count", 0) or 0)  # billed as output
        usage = Usage(
            input_tokens=int(getattr(meta, "prompt_token_count", 0) or 0),
            output_tokens=output_tokens,
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
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={"system_instruction": system or "Answer concisely and cite sources as [n]."},
        )
        return (response.text or "").strip()
