# 2. Never invent: correct only with proof, otherwise null and review

- Status: accepted
- Date: 2026-11

## Context

A guardrail that "fixes" a wrong value with a plausible guess turns a visible error into an invisible one. In finance, a missing field costs a reviewer a minute; a wrong field published silently costs a payment or a tax filing.

## Decision

A guardrail may replace a value only with a value it can prove from the document:

1. the authoritative XML (ADR 0003);
2. the verified access key (check digit valid, model right, literally printed), for the fields it encodes;
3. a unique, self-consistent re-read of the box that owns the value (exactly one candidate, and where possible an arithmetic confirmation such as components adding up to the printed total).

Otherwise the value is nulled and the document goes to review. The day of issue, for example, is not in the access key, so a wrong date is nulled, never corrected. Every published value carries a provenance (`model`, `xml`, `access_key`, `document_text`, `reviewer`). Reviewer corrections go through the same rules (`guardrails/validate.py`): a value that is not printed in the document, or breaks a check digit or a sum, is rejected with the reasons.

## Consequences

- False corrections are measured on every run and are zero on the evaluation corpus. A randomized test (12 seeds, error rates scaled ×6) asserts that every published value is printed in the document or derived from the verified key, and that no correct value is ever changed.
- Some documents that a human would fix in seconds go to review. That is the intended trade: review is visible, a silent error is not.
- "Silent-error documents" (wrong data, no review) is the headline metric, not field accuracy.
