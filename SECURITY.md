# Security

## Reporting a vulnerability

Please use GitHub's **private vulnerability reporting** (Security tab → "Report a vulnerability") on this repository. Do not open a public issue for a security problem. Include the version or commit, the steps to reproduce and the impact you expect. You should get an answer within a week.

Only the latest release on `main` is supported.

## Scope and assets

The service receives untrusted PDFs and XML, sends document text to an LLM provider when one is configured, and stores queued documents and review actions in SQLite. The assets worth protecting are the document contents (fiscal data, company identifiers), the integrity of published records and of the audit trail, provider API keys, and the availability of the service.

## Threat model

| Threat | Entry point | Controls | Residual risk |
|---|---|---|---|
| Memory or disk exhaustion by a huge upload | `POST /extract`, `POST /review/items` | ASGI middleware rejects by `Content-Length` and counts streamed bytes, answering **413** before the body is read; per-endpoint size check; XML size limit | A proxy should still cap body size before traffic reaches Python |
| Malicious or pathological PDF (decompression bombs, parser bugs, very long text) | PDF parsing | `%PDF-` magic check; parsing in a child process with page, character, CPU-time and memory limits and a wall-clock timeout; the child gets a scrubbed environment (no API keys); encrypted or malformed files → **422** | A parser vulnerability inside the child is contained to that process, not eliminated |
| XML attacks (entity expansion, external entities) | optional `xml` upload, MCP | `defusedxml` only, size limit, type and access-key consistency checks before use | — |
| Prompt injection planted in the document | LLM extraction | the text is fenced as data and look-alike delimiters are neutralized; the system prompt forbids following embedded instructions; **guardrails do not trust the model**: values must be printed, pass check digits, match the access key or add up | An attacker who prints a *plausible* value (for example a second total) on the document can still steer the model to a value that is printed; check digits, the XML and sums catch most of it, not all. Keep humans on high-value documents |
| Path traversal and arbitrary file read | MCP `extract_document`, `/samples/{name}` | MCP reads only regular files inside `DOCAI_MCP_ROOT` (resolved path, no escape through `..` or symlinks), with the right suffix, size and magic bytes; sample names are matched against a fixed list | — |
| Unauthorized changes to the review queue | `POST /review/*` | bearer token (`DOCAI_API_TOKEN`, ≥ 24 characters) compared with `hmac.compare_digest`; **503** when no token is configured, so a forgotten token disables mutations instead of opening them | One shared token: the reviewer name in the audit trail is declared by the client and recorded as `"<name> (bearer token)"`, not authenticated per user |
| Tampering with the audit trail | SQLite | append-only by triggers (no `UPDATE`/`DELETE`), parameterized SQL only, corrections validated by the guardrail rules | Anyone with write access to the database file can bypass triggers; protect the volume |
| Abuse and brute force | all `POST` endpoints | in-memory token bucket per client address (429 with `Retry-After`), bounded number of tracked clients | Per process and keyed on the socket address: behind a proxy every client looks the same. Rate limit at the proxy |
| Cross-site scripting and clickjacking | web UI, dashboard | static files only, no inline script or style, strict CSP (`default-src 'self'`, `frame-ancestors 'none'`), all dynamic text set with `textContent`; `nosniff`, `X-Frame-Options: DENY`, `no-referrer`, COOP/CORP | `/docs` loads Swagger UI from jsDelivr, pinned by version and Subresource Integrity; set `DOCAI_DOCS_ENABLED=false` to remove that third-party origin |
| Cross-origin requests | API | no CORS unless `DOCAI_CORS_ORIGINS` lists explicit origins (`*` is rejected) | — |
| Information leakage in errors and logs | all | generic error bodies with an `X-Request-ID`; validation errors never echo input; logs carry the request id and a 12-character hash of the document, never its content, keys or tokens; secrets are `SecretStr` | Queued documents (text and extracted fields) are stored in SQLite in clear text |
| Data sent to a third party | LLM provider | the simulated provider is the default and sends nothing; real providers are explicit opt-in; XML-first avoids the call when the XML is complete | Document text leaves the machine when a real provider is used: check your provider's data-retention terms |
| Supply chain | dependencies, image, CI | hash-pinned locks (`--require-hashes`), base image pinned by tag and digest, non-root user, read-only root filesystem, no shell entry point; GitHub Actions pinned to commit SHAs with least-privilege permissions; CodeQL, bandit, pip-audit and Dependabot | The CPU PyTorch wheel is installed from the PyTorch index by exact version without hashes (it is not on PyPI in the CPU variant) |

## Deployment guidance

- **Set a token**: `DOCAI_API_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(32))")`. Without it the review endpoints answer 503.
- **Keep the queue private**: `GET /review/*` needs the token by default, because the queue contains document data. `DOCAI_REVIEW_PUBLIC_READ=true` opens reads; use it only for a demo with synthetic documents.
- **Run behind a reverse proxy** (nginx, Caddy, a cloud load balancer) that terminates **TLS**, caps the request body, rate-limits per real client address and forwards only the paths you use. Keep the app bound to `127.0.0.1` (as `docker-compose.yml` does) or to a private network.
- **Keep the container hardened**: read-only root filesystem, `/tmp` as tmpfs, `cap_drop: ALL`, `no-new-privileges`, the database on a dedicated volume with restricted permissions and backups.
- **Logs**: `DOCAI_LOG_JSON=true` for structured logs; ship them with the request id so a user report can be traced.
- **Real providers**: pass keys through the environment or a secret store, never through files in the image; prefer the XML-first path to reduce what is sent.
- **Disable what you do not need**: `DOCAI_DOCS_ENABLED=false` removes `/docs` and its CDN dependency.
