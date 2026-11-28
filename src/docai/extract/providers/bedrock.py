"""AWS Bedrock provider (Converse API with a forced tool for structured output).

Optional dependency: ``pip install 'ai-guardrails[bedrock]'``. Credentials come from
the standard AWS chain (env vars, profile, role). The model id is account/region
specific, so it must be configured: ``DOCAI_BEDROCK_MODEL_ID``.
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


class BedrockProvider(LLMProvider):
    """AWS Bedrock through the Converse API, structured output via a forced tool."""

    name = "bedrock"

    def __init__(
        self,
        model: str | None = None,
        *,
        region: str | None = None,
        max_tokens: int = 2048,
        timeout_s: float = 60.0,
        client: Any = None,
        pricing: Pricing | None = None,
    ) -> None:
        if not model:
            raise ProviderUnavailable("set DOCAI_BEDROCK_MODEL_ID to a Bedrock model id")
        super().__init__(model, pricing)
        self.region = region
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self._client = client

    @property
    def client(self) -> Any:
        """The SDK client, created on first use (the SDK is an optional dependency)."""
        if self._client is None:
            try:
                import boto3  # lazy: optional dependency
            except ImportError as exc:
                raise ProviderUnavailable(
                    "install the extra: pip install 'ai-guardrails[bedrock]'"
                ) from exc
            from botocore.config import Config

            config = Config(read_timeout=self.timeout_s, retries={"max_attempts": 2})
            self._client = boto3.client("bedrock-runtime", region_name=self.region, config=config)
        return self._client

    def complete(
        self,
        text: str,
        schema: type[BaseModel],
        context: ExtractionContext | None,
        variant: PromptVariant = PromptVariant.STANDARD,
    ) -> ExtractionResult:
        """One Converse call with a forced tool; ``stopReason == "max_tokens"`` = truncated."""
        doc_type = doc_type_of_schema(schema)
        prompt = build_user_prompt(text, doc_type, context, variant)
        started = time.perf_counter()
        response = self.client.converse(
            modelId=self.model,
            system=[{"text": SYSTEM_PROMPT}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            toolConfig={
                "tools": [
                    {
                        "toolSpec": {
                            "name": TOOL_NAME,
                            "description": "Record the fields extracted from the document.",
                            "inputSchema": {"json": inline_json_schema(schema.model_json_schema())},
                        }
                    }
                ],
                "toolChoice": {"tool": {"name": TOOL_NAME}},
            },
            inferenceConfig={"maxTokens": self.max_tokens, "temperature": 0},
        )
        wall = time.perf_counter() - started
        blocks = response.get("output", {}).get("message", {}).get("content", [])
        data = next((b["toolUse"]["input"] for b in blocks if "toolUse" in b), None)
        finish = (
            FinishReason.LENGTH if response.get("stopReason") == "max_tokens" else FinishReason.STOP
        )
        error = None
        if finish == FinishReason.LENGTH:
            data, error = None, "the answer was cut at maxTokens"
        elif not isinstance(data, dict):
            data, error = None, "the response has no toolUse block"
        tokens = response.get("usage", {})
        latency_ms = response.get("metrics", {}).get("latencyMs")
        usage = Usage(
            input_tokens=int(tokens.get("inputTokens", 0)),
            output_tokens=int(tokens.get("outputTokens", 0)),
            latency_s=round(latency_ms / 1000 if latency_ms else wall, 3),
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
        response = self.client.converse(
            modelId=self.model,
            system=[{"text": system or "Answer concisely and cite sources as [n]."}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": self.max_tokens},
        )
        blocks = response.get("output", {}).get("message", {}).get("content", [])
        return "".join(b.get("text", "") for b in blocks).strip()
