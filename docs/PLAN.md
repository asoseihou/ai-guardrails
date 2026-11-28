# Plan

## Delivered (0.2.0)

| Area | What exists |
|---|---|
| Guardrails | Four measured levels: check digits (v1); access-key cross-checks, literal presence and consistency (v2); layout anchors and a name registry (v3). Never-invent policy with provenance and structured findings. Replay with blast radius before every rule change. |
| Extraction | Per-type anchored prompts, document fenced as untrusted data, XML-first with `defusedxml`, one summary-mode retry on truncation, four providers (simulated, Anthropic, Gemini, Bedrock). |
| Review | SQLite queue, corrections validated by the guardrail rules, append-only audit trail, API and UI. |
| Measurement | Eval harness with silent-error rate, per error mode and per field; XML-first vs LLM-only; truncation recovery and cost avoided; run history and dashboard. |
| Security | Streamed body limit, isolated PDF parsing with limits, security headers and strict CSP, rate limit, bearer token for mutations, MCP sandbox, structured logs without document content, pinned supply chain, CodeQL, bandit, pip-audit. |
| Engineering | Layered architecture enforced by a test, typed settings, mypy strict, ruff, 85% coverage gate on the core (95% today), CI matrix 3.12/3.13, Docker image pinned by digest, Dependabot. |

## Milestones

### M1. Real-provider benchmark

Run the same corpus through the Anthropic, Gemini and Bedrock models and publish what the simulator cannot tell: real error modes, real cost and latency.

Acceptance criteria:
- `python -m docai eval --provider <name>` on the full corpus for at least two providers, with the report committed under `reports/providers/`.
- Per provider: field accuracy and silent-error rate per guardrail level, false corrections, cost per 1,000 documents from billed usage, p50/p95 latency.
- Error modes observed in real output are categorized; any new mode gets a simulator counterpart and, where possible, a guardrail with a replay showing zero regressions.
- README states which numbers are simulated and which are measured.

### M2. OCR path for scanned PDFs

Acceptance criteria:
- A PDF with no text layer is detected and routed to OCR (for example Tesseract with Portuguese data) inside the same isolated worker and limits.
- The synthetic generator can emit rasterized pages with noise; the eval reports accuracy and silent errors on scanned vs native PDFs.
- Presence checks tolerate OCR confusions only through explicit, tested normalization (no fuzzy acceptance of CNPJs or keys).

### M3. MLflow tracking

Acceptance criteria:
- Classifier training logs parameters, metrics and the model artifact; each evaluation run logs the guardrail preset, provider, metrics, cost and the report as an artifact.
- The SQLite run history stays the source for `/dashboard`; MLflow is optional (`DOCAI_MLFLOW_URI`), off in CI.
- A guardrail change can be compared run to run in the MLflow UI, linked to the replay report.

### M4. Airflow DAG for ingestion

Acceptance criteria:
- DAG: mailbox or bucket → deduplicate by content hash → pipeline → publish or enqueue for review, with retries that never re-publish a document already published.
- Idempotent by document hash; failed tasks are visible in the review queue, not only in Airflow logs.
- Integration test with a local Airflow (docker compose) on the synthetic corpus.

### M5. Kubernetes deployment

Acceptance criteria:
- Helm chart with a read-only root filesystem, non-root user, resource limits, liveness and readiness on `/health`, and the token from a Secret.
- Ingress with TLS and rate limiting at the proxy; the in-process limiter documented as a fallback only.
- The review and run repositories moved to PostgreSQL behind the same interfaces (ADR 0005), so the API can scale to more than one replica.

## Domain items not implemented

The domain guidance asked for these where they fit the synthetic schema. They are skipped for now, with the reason:

| Item | Why it is not here yet |
|---|---|
| Multi-document PDF segmentation (several documents per file, continuation pages, merging segments with the same number and total, statement/mirror pages that repeat an invoice's amount) | The generator produces one document per PDF plus attachments; segmentation needs a corpus with mixed files and its own evaluation. Today attachments are handled by reading the CT-e main page only. |
| Summary-mode retry for the segmentation step (an identifying excerpt of at most ~1,500 characters per segment) | Depends on segmentation; truncation recovery exists for extraction. |
| Correction letters inheriting the type of the document they correct | No correction letters in the corpus. |
| Payment receipts, house/AWB documents and bill-of-lading references | Not in the schema; only the service-order reference is modelled. |
| NFS-e retained taxes (PIS, COFINS, CSLL, IRPJ, ISS, INSS; 0.0 when absent) | Not in the synthetic NFS-e; the ISS rate × value check covers the one tax that is generated. |
| NF-e nature of operation, entry/exit flag, model and cancelled flag | Not generated; the model number is already enforced through the access key. |
| A separate CT-e payer field and the "recipient equals payer while the sender is also the payer" flag | The TOMADOR box is rendered and the role anchors stop it from leaking into sender or recipient; two roles with the same CNPJ are flagged. A payer field would need its own schema change. |
| XML signature verification | The XML is trusted as an input; checking the signature needs certificate handling (planned with M1-M2). |
| Per-user authentication (OIDC) for reviewers | Out of scope for a single-token demo service; reviewer names are self-declared (see SECURITY.md). |
| Splitting `render_nfe`, `render_nfse` and `make_nfe` (synthetic-data builders, 62 to 94 lines) | Straight-line layout and record construction with no branches worth testing separately; `render_cte` was split because it has branches. Every other function is under 60 lines and complexity 10 (ruff C90). |
