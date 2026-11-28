# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-11-26

### Added

- **XML-first extraction.** The generator emits NF-e and NFS-e XML; the XML is parsed with `defusedxml`, wins over the LLM field by field, and replaces the LLM call when complete. The report compares LLM-only, cross-check and XML-first.
- **Human review queue** in SQLite: list, inspect, correct (validated by the guardrail rules) and approve, with an append-only audit trail; `/review` endpoints, a CLI command and a UI tab.
- **Truncation recovery**: providers report a finish reason; a cut answer gets one summary-mode retry, then goes to review. Retries and the cost avoided versus a naive retry loop are reported.
- **Cost & quality dashboard** at `/dashboard` and `GET /metrics/runs`, backed by a run history stored on every evaluation.
- **Layout-anchor guardrails (v3)**: role blocks (sender, recipient, payer trap), CT-e main page for weight and due date (attachments ignored), referenced NF-e keys, labelled service-order references, NFS-e services table, name follows a verified CNPJ (registry). New `text` preset (v2) between `dv-only` and `full`.
- Per-type anchored prompts, new fields (CT-e weight, due date, referenced keys, service order; NFS-e services), boleto routing rule, and simulator error modes for each new failure (name from a neighbour, reference from an address, weight and due date from an attached DANFE, payer into recipient, referenced-key typo, truncation, prompt injection).
- Central typed `Settings` (pydantic-settings), layering enforced by a test, structured JSON logging with request ids.
- Documentation: architecture, processing guide, ADRs 0001-0005, plan, security policy, contributing guide, screenshots.

### Changed

- The Anthropic model id is required configuration (`DOCAI_ANTHROPIC_MODEL`), like the Bedrock model id; `auto` picks that provider only when both the key and the model are set.
- Renamed the project from `doc-ai-guardrails` to `ai-guardrails` (the Python package is still `docai`).
- Package split into `domain`, `guardrails`, `extract`, `pipeline`, `storage`, `review`, `eval` and interface layers; long functions split.
- Python 3.12+ (CI matrix 3.12 and 3.13).
- `make demo` also refreshes the results table in the README.

### Security

- Streamed request-body limit (413 before the body is read), PDF parsing in an isolated child process with page, text, CPU, memory and time limits (422 for encrypted or malformed files).
- Security headers and a strict CSP with no inline script or style; CORS only from an explicit allowlist.
- Per-client token-bucket rate limit on `POST`; generic errors with `X-Request-ID`; logs without document content or secrets.
- Bearer token (`DOCAI_API_TOKEN`, constant-time comparison) for the review queue, reads included, 503 when unset. `DOCAI_REVIEW_PUBLIC_READ=true` opens reads for a demo.
- MCP file tools sandboxed to `DOCAI_MCP_ROOT`.
- The web UI keeps the review token only in memory; it is never written to browser storage.
- Prompt-injection fencing of the document text, with a test that the guardrails catch an obeyed instruction.
- Supply chain: hash-pinned locks, base image pinned by digest, non-root read-only container with a health check, Actions pinned to commit SHAs, CodeQL, bandit, pip-audit, Dependabot.

## [0.1.0] - 2026-11-26

### Added

- Synthetic NF-e, CT-e, NFS-e, boleto and other PDFs with ground truth; CNPJ and access-key arithmetic.
- Seeded simulated LLM and providers for Anthropic, Gemini and AWS Bedrock.
- Deterministic guardrails (check digits, access-key cross-checks, literal presence, consistency) with `off`, `dv-only` and `full` presets.
- Evaluation harness with silent-error rate, replay (blast radius) and Markdown/JSON reports.
- BM25 RAG over fiscal-rule notes and a PyTorch document-type classifier.
- FastAPI service with a web UI, MCP server, Docker image and CI.

[0.2.0]: https://github.com/asoseihou/ai-guardrails/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/asoseihou/ai-guardrails/releases/tag/v0.1.0
