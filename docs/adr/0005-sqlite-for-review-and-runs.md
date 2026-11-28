# 5. SQLite for the review queue and the run history

- Status: accepted
- Date: 2026-11

## Context

The review queue needs durable items, an audit trail that cannot be rewritten, and one pending item per document. The dashboard needs the history of evaluation runs. The service is a single process that should start with `make serve` or one container, with no external database to operate.

## Decision

Use SQLite through the standard library (`storage/`), with WAL, foreign keys, parameterized SQL only, a partial unique index for one pending item per document, and triggers that reject `UPDATE` and `DELETE` on `review_audit`. Repositories (`ReviewRepository`, `RunRepository`) hide SQL from the services. The file lives at `DOCAI_DB_PATH` on a volume, since the container root filesystem is read-only.

## Consequences

- Zero operational cost and transactional writes; the audit trail is append-only by construction.
- One writer at a time: fine for a review team, not for many API replicas. Scaling out means a server database (PostgreSQL) behind the same repository interfaces.
- The database holds the text of queued documents; protect the volume and back it up (see SECURITY.md).
