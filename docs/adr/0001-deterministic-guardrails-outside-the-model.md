# 1. Deterministic guardrails outside the model

- Status: accepted
- Date: 2026-11

## Context

The model's mistakes on fiscal documents are not random noise. It copies a valid CNPJ from the wrong box, takes a number from the file name, reads the weight of an attached DANFE, or completes a company name from a neighbour it has seen. The output is well-formed, so schema validation passes, and at `temperature=0` the same input reproduces the same mistake. Asking the model (or a second model) to double-check tends to repeat the error, costs tokens on every document and is hard to audit.

Brazilian fiscal documents carry strong anchors: modulo-11 check digits on CNPJs and access keys, an access key that encodes the issuer, model, series, number and month, totals that must equal the sum of printed lines, and fixed boxes for each party.

## Decision

All validation and correction happens in deterministic code that runs after the model: `docai.guardrails`, a pure package (input record plus the document text, output record plus findings) with no network, model or database access. Checks are grouped into presets (`off`, `dv-only`, `text`, `full`) so each level can be measured, and every change to a value is recorded as a `Finding` with the check, the reason, the value before and after.

## Consequences

- Checks are cheap (microseconds), explainable to an auditor and testable, including under heavy randomized corruption.
- A rule change can be replayed on stored model outputs (`docai replay`) to measure its blast radius before it ships.
- The guardrails only know what the schema and layout readers know: a new field or layout needs code, not a prompt tweak.
- Layout anchors depend on the printed layout; on an unknown layout they do nothing and the layout-independent checks still apply.
