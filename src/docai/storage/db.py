"""SQLite connection and schema (stdlib ``sqlite3``, parameterized SQL only).

One small database file holds the review queue (with its append-only audit
trail, enforced by triggers) and the run history used by the dashboard.
Every connection enables foreign keys and WAL; the schema is created
idempotently on first use.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS review_items (
    id                  TEXT PRIMARY KEY,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL,
    status              TEXT NOT NULL CHECK (status IN ('pending', 'approved')),
    doc_type            TEXT NOT NULL,
    filename            TEXT,
    document_sha256     TEXT NOT NULL,
    source              TEXT NOT NULL,
    record_json         TEXT NOT NULL,
    field_status_json   TEXT NOT NULL,
    findings_json       TEXT NOT NULL,
    review_reasons_json TEXT NOT NULL,
    text                TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS review_items_by_status ON review_items (status, created_at);
CREATE UNIQUE INDEX IF NOT EXISTS review_items_one_pending_per_document
    ON review_items (document_sha256) WHERE status = 'pending';

CREATE TABLE IF NOT EXISTS review_audit (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id     TEXT NOT NULL REFERENCES review_items (id),
    at          TEXT NOT NULL,
    actor       TEXT NOT NULL,
    action      TEXT NOT NULL CHECK (action IN ('created', 'corrected', 'approved')),
    field       TEXT,
    before_json TEXT,
    after_json  TEXT,
    reason      TEXT
);
CREATE INDEX IF NOT EXISTS review_audit_by_item ON review_audit (item_id, seq);
CREATE TRIGGER IF NOT EXISTS review_audit_no_update BEFORE UPDATE ON review_audit
BEGIN SELECT RAISE(ABORT, 'the audit trail is append-only'); END;
CREATE TRIGGER IF NOT EXISTS review_audit_no_delete BEFORE DELETE ON review_audit
BEGIN SELECT RAISE(ABORT, 'the audit trail is append-only'); END;

CREATE TABLE IF NOT EXISTS runs (
    id                 TEXT PRIMARY KEY,
    created_at         TEXT NOT NULL,
    command            TEXT NOT NULL,
    config_json        TEXT NOT NULL,
    n_docs             INTEGER NOT NULL,
    field_accuracy     REAL NOT NULL,
    review_rate        REAL NOT NULL,
    silent_error_rate  REAL NOT NULL,
    straight_through   REAL NOT NULL,
    cost_per_doc       REAL NOT NULL,
    latency_p50        REAL NOT NULL,
    latency_p95        REAL NOT NULL,
    retries            INTEGER NOT NULL,
    per_field_json     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS runs_by_time ON runs (created_at);
"""


@contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Open the database (creating it and its schema if needed), commit or roll back, close."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.executescript(SCHEMA)
        with connection:  # one transaction: commit on success, roll back on error
            yield connection
    finally:
        connection.close()
