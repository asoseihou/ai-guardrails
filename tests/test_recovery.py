"""Truncation recovery: one summary-mode retry, then review; never a blind retry loop."""

import dataclasses
import json
from types import SimpleNamespace

import pytest

from docai.classify import KeywordClassifier
from docai.domain.compare import same_value
from docai.domain.schemas import CTeExtraction, NFeExtraction
from docai.extract.prompts import SUMMARY_MODE
from docai.extract.providers import (
    ErrorRates,
    ExtractionContext,
    ProviderError,
    SimulatedLLMProvider,
    TruthStore,
)
from docai.extract.providers.anthropic import AnthropicProvider
from docai.extract.providers.base import FinishReason, PromptVariant, build_user_prompt
from docai.extract.providers.bedrock import BedrockProvider
from docai.extract.providers.gemini import GeminiProvider
from docai.pipeline import Pipeline
from docai.pipeline.recovery import (
    NAIVE_MAX_RETRIES,
    TRUNCATION_REASON,
    extract_with_recovery,
    truncation_summary,
)


def _provider(corpus_dir, truncation, persists):
    rates = dataclasses.replace(
        ErrorRates.none(), truncation=truncation, truncation_persists=persists
    )
    return SimulatedLLMProvider(TruthStore([corpus_dir]), rates)


def _doc(by_type, texts, doc_type="cte"):
    doc = by_type[doc_type][0]
    return doc, texts[doc["doc_id"]], ExtractionContext(doc_id=doc["doc_id"])


def test_summary_retry_recovers_a_truncated_answer(corpus_dir, by_type, texts):
    doc, text, ctx = _doc(by_type, texts)
    recovery = extract_with_recovery(_provider(corpus_dir, 1.0, 0.0), text, "cte", ctx)
    first, second = recovery.attempts
    assert first.finish_reason == FinishReason.LENGTH and first.data is None
    assert second.usable
    assert all(same_value(k, v, doc["fields"][k]) for k, v in second.data.items())
    assert recovery.truncated and recovery.retried and recovery.result is second
    meta = recovery.meta()
    assert meta["attempts"] == 2 and meta["recovered"] is True
    assert meta["finish_reasons"] == ["length", "stop"]
    assert recovery.usage.output_tokens == first.usage.output_tokens + second.usage.output_tokens
    assert second.usage.output_tokens < first.usage.output_tokens  # compact answer


def test_no_truncation_means_one_call(corpus_dir, by_type, texts):
    _, text, ctx = _doc(by_type, texts)
    recovery = extract_with_recovery(_provider(corpus_dir, 0.0, 0.0), text, "cte", ctx)
    assert len(recovery.attempts) == 1 and not recovery.truncated and not recovery.retried


def test_truncated_twice_goes_to_review(corpus_dir, by_type, texts):
    doc, text, _ = _doc(by_type, texts, "nfe")
    pipeline = Pipeline(_provider(corpus_dir, 1.0, 1.0), KeywordClassifier(), "full")
    result = pipeline.run_text(text, doc_id=doc["doc_id"])
    assert result.meta["attempts"] == 2 and result.meta["recovered"] is False
    assert result.guardrails.needs_review
    assert TRUNCATION_REASON in result.guardrails.review_reasons
    assert result.raw is None
    body = result.to_dict()
    assert body["needs_review"] and TRUNCATION_REASON in body["review_reasons"]


def test_extract_raises_on_an_unusable_answer(corpus_dir, by_type, texts):
    _, text, ctx = _doc(by_type, texts)
    with pytest.raises(ProviderError, match="output limit"):
        _provider(corpus_dir, 1.0, 1.0).extract(text, CTeExtraction, context=ctx)


def test_summary_prompt_keeps_the_rules():
    prompt = build_user_prompt("x", "cte", None, PromptVariant.SUMMARY)
    assert prompt.startswith(SUMMARY_MODE) and "TOMADOR" in prompt
    assert SUMMARY_MODE not in build_user_prompt("x", "cte", None)


def test_naive_loop_comparison():
    metas = [
        {"truncated": True, "recovered": True, "attempts": 2, "attempt_costs": [0.01, 0.004]},
        {"truncated": True, "recovered": False, "attempts": 2, "attempt_costs": [0.01, 0.01]},
        {"truncated": False, "attempts": 1, "attempt_costs": [0.003]},
    ]
    summary = truncation_summary(metas)
    assert summary["truncated"] == 2 and summary["recovered"] == 1
    assert summary["sent_to_review"] == 1 and summary["retries"] == 2
    assert summary["naive_extra_calls"] == 2 * NAIVE_MAX_RETRIES
    # naive: 2 docs x 4 calls x 0.01 = 0.08; ours: 2 x 0.01 + 0.014 = 0.034
    assert summary["cost_avoided_usd"] == pytest.approx(0.046)
    assert truncation_summary([])["truncated"] == 0


# ---------------------------------------------------------------- real providers (mocked)
def test_anthropic_reports_max_tokens_as_truncation():
    class Fake:
        messages = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            block = SimpleNamespace(type="tool_use", input={"number": "1"})
            return SimpleNamespace(
                content=[block],
                stop_reason="max_tokens",
                usage=SimpleNamespace(input_tokens=10, output_tokens=2048),
            )

    fake = Fake()
    fake.messages = fake
    result = AnthropicProvider("test-model", client=fake).complete(
        "t", NFeExtraction, None, PromptVariant.SUMMARY
    )
    assert result.truncated and result.data is None and "max_tokens" in result.error
    assert SUMMARY_MODE in fake.kwargs["messages"][0]["content"]


def test_gemini_reports_max_tokens_and_invalid_json():
    def client(text, reason):
        def generate_content(**kwargs):
            meta = SimpleNamespace(prompt_token_count=5, candidates_token_count=5)
            candidate = SimpleNamespace(finish_reason=SimpleNamespace(name=reason))
            return SimpleNamespace(text=text, usage_metadata=meta, candidates=[candidate])

        return SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))

    cut = GeminiProvider(client=client('{"number": "1', "MAX_TOKENS")).complete(
        "t", NFeExtraction, None
    )
    assert cut.finish_reason == FinishReason.LENGTH and cut.truncated
    broken = GeminiProvider(client=client("not json", "STOP")).complete("t", NFeExtraction, None)
    assert broken.finish_reason == FinishReason.STOP and broken.truncated
    assert "invalid JSON" in broken.error
    ok = GeminiProvider(client=client(json.dumps({"number": "1"}), "STOP"))
    assert ok.complete("t", NFeExtraction, None).usable


def test_bedrock_reports_max_tokens():
    class Runtime:
        def converse(self, **kwargs):
            return {
                "output": {"message": {"content": [{"toolUse": {"input": {"number": "1"}}}]}},
                "stopReason": "max_tokens",
                "usage": {"inputTokens": 1, "outputTokens": 2048},
            }

    result = BedrockProvider(model="m", client=Runtime()).complete("t", NFeExtraction, None)
    assert result.truncated and result.data is None
