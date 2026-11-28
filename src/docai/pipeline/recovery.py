"""Truncation recovery: one compact retry, then human review.

A structured answer cut at the output limit is invalid JSON. Retrying the same
request is pointless at temperature 0 (the answer is cut at the same place,
every time) and costs a full output budget per attempt. The policy here:

1. call once with the standard prompt;
2. if the answer is truncated or not valid JSON, retry **once** with the
   compact ``summary`` prompt variant (same rules, no whitespace, short lists);
3. if that fails too, give up and send the document to review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from docai.domain.schemas import schema_for
from docai.extract.providers import ExtractionContext, LLMProvider, Usage
from docai.extract.providers.base import ExtractionResult, PromptVariant

NAIVE_MAX_RETRIES = 3  # the "retry until it works" loop we compare against


@dataclass
class Recovery:
    """Every attempt of one extraction and the usable result, if any."""

    attempts: list[ExtractionResult] = field(default_factory=list)

    @property
    def result(self) -> ExtractionResult | None:
        """The last attempt when it is usable."""
        last = self.attempts[-1] if self.attempts else None
        return last if last is not None and last.usable else None

    @property
    def truncated(self) -> bool:
        """The first answer was truncated (or not valid JSON)."""
        return bool(self.attempts) and self.attempts[0].truncated

    @property
    def retried(self) -> bool:
        """A summary-mode retry was made."""
        return len(self.attempts) > 1

    @property
    def usage(self) -> Usage:
        """Tokens and cost of every attempt together."""
        total = Usage(simulated=all(a.usage.simulated for a in self.attempts))
        for attempt in self.attempts:
            total = total + attempt.usage
        return total

    def meta(self) -> dict[str, Any]:
        """What the pipeline reports about the attempts."""
        meta = dict(self.result.meta) if self.result else {}
        meta.update(
            attempts=len(self.attempts),
            truncated=self.truncated,
            retried=self.retried,
            recovered=self.retried and self.result is not None,
            finish_reasons=[str(a.finish_reason) for a in self.attempts],
            attempt_costs=[a.usage.cost_usd for a in self.attempts],
        )
        if self.result is None and self.attempts:
            meta["injected_errors"] = []
            meta["error"] = self.attempts[-1].error
        return meta


def extract_with_recovery(
    provider: LLMProvider, text: str, doc_type: str, context: ExtractionContext
) -> Recovery:
    """Standard call, then at most one compact retry when the answer was truncated."""
    recovery = Recovery()
    schema = schema_for(doc_type)
    for variant in (PromptVariant.STANDARD, PromptVariant.SUMMARY):
        attempt = provider.extract_with_usage(text, schema, context=context, variant=variant)
        recovery.attempts.append(attempt)
        if attempt.usable:
            break
    return recovery


TRUNCATION_REASON = (
    "the model's answer was cut at the output limit twice (standard and summary mode); "
    "the document goes to review instead of retrying again"
)


def truncation_summary(metas: list[dict[str, Any]]) -> dict[str, Any]:
    """Retries and the cost avoided versus a naive "retry until it works" loop.

    At temperature 0 a truncated answer is truncated again on an identical
    retry, so the naive loop spends ``NAIVE_MAX_RETRIES`` more full-budget calls
    per truncated document and recovers none of them.
    """
    truncated = [m for m in metas if m.get("truncated")]
    recovered = sum(1 for m in truncated if m.get("recovered"))
    retries = sum(m.get("attempts", 1) - 1 for m in truncated)
    first_costs = [m["attempt_costs"][0] for m in truncated if m.get("attempt_costs")]
    retry_cost = sum(sum(m.get("attempt_costs", [])[1:]) for m in truncated)
    cut_call = sum(first_costs) / len(first_costs) if first_costs else 0.0
    naive_cost = len(truncated) * (1 + NAIVE_MAX_RETRIES) * cut_call
    our_cost = len(truncated) * cut_call + retry_cost
    return {
        "calls": len(metas),
        "truncated": len(truncated),
        "recovered": recovered,
        "sent_to_review": len(truncated) - recovered,
        "retries": retries,
        "retry_cost_usd": round(retry_cost, 5),
        "naive_max_retries": NAIVE_MAX_RETRIES,
        "naive_extra_calls": len(truncated) * NAIVE_MAX_RETRIES,
        "naive_recovered": 0,
        "cost_avoided_usd": round(naive_cost - our_cost, 5),
    }
