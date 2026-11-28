"""Repository for the human review queue and its append-only audit trail."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from docai.storage.db import connect

STATUSES = ("pending", "approved")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


@dataclass(frozen=True)
class AuditEntry:
    """One immutable line of the audit trail: who did what to which field, and why."""

    seq: int
    at: str
    actor: str
    action: str
    field: str | None
    before: Any
    after: Any
    reason: str | None

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation."""
        return dict(self.__dict__)


@dataclass
class ReviewItem:
    """A document waiting for (or done with) human review."""

    id: str
    created_at: str
    updated_at: str
    status: str
    doc_type: str
    filename: str | None
    document_sha256: str
    source: str
    record: dict[str, Any]
    field_status: dict[str, str]
    findings: list[dict[str, Any]]
    review_reasons: list[str]
    text: str = field(repr=False)
    audit: list[AuditEntry] = field(default_factory=list)

    @property
    def pending_fields(self) -> list[str]:
        """Fields still nulled or flagged (not yet corrected or confirmed by a reviewer)."""
        return [
            name for name, status in self.field_status.items() if status in ("nulled", "flagged")
        ]

    def summary(self) -> dict[str, Any]:
        """What the queue listing shows (never the document text)."""
        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "doc_type": self.doc_type,
            "filename": self.filename,
            "source": self.source,
            "pending_fields": self.pending_fields,
            "review_reasons": self.review_reasons,
        }

    def to_dict(self) -> dict[str, Any]:
        """Full detail for the API (the document text stays server-side)."""
        return {
            **self.summary(),
            "document_sha256": self.document_sha256,
            "record": self.record,
            "field_status": self.field_status,
            "findings": self.findings,
            "audit": [entry.to_dict() for entry in self.audit],
        }


class ReviewRepository:
    """Persistence for :class:`ReviewItem` (one SQLite file, parameterized SQL only)."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def add(
        self,
        *,
        doc_type: str,
        filename: str | None,
        document_sha256: str,
        source: str,
        record: dict[str, Any],
        field_status: dict[str, str],
        findings: list[dict[str, Any]],
        review_reasons: list[str],
        text: str,
        actor: str = "pipeline",
    ) -> tuple[str, bool]:
        """Insert a pending item (or return the pending one for the same document).

        Returns ``(item_id, created)``.
        """
        with connect(self.db_path) as db:
            row = db.execute(
                "SELECT id FROM review_items WHERE document_sha256 = ? AND status = 'pending'",
                (document_sha256,),
            ).fetchone()
            if row is not None:
                return str(row["id"]), False
            item_id, now = uuid.uuid4().hex, _now()
            db.execute(
                "INSERT INTO review_items (id, created_at, updated_at, status, doc_type,"
                " filename, document_sha256, source, record_json, field_status_json,"
                " findings_json, review_reasons_json, text)"
                " VALUES (?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    item_id,
                    now,
                    now,
                    doc_type,
                    filename,
                    document_sha256,
                    source,
                    _dump(record),
                    _dump(field_status),
                    _dump(findings),
                    _dump(review_reasons),
                    text,
                ),
            )
            self._audit(db, item_id, actor, "created", None, None, None, "sent to review")
        return item_id, True

    def count_pending(self) -> int:
        """Number of items waiting for review."""
        with connect(self.db_path) as db:
            row = db.execute("SELECT COUNT(*) AS n FROM review_items WHERE status = 'pending'")
            return int(row.fetchone()["n"])

    def list(
        self, status: str | None = "pending", limit: int = 50, offset: int = 0
    ) -> list[ReviewItem]:
        """Items by creation time (newest first), optionally filtered by status."""
        query = "SELECT * FROM review_items"
        params: list[Any] = []
        if status is not None:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC, id LIMIT ? OFFSET ?"
        params += [limit, offset]
        with connect(self.db_path) as db:
            return [self._item(row) for row in db.execute(query, params).fetchall()]

    def get(self, item_id: str) -> ReviewItem | None:
        """One item with its audit trail, or ``None``."""
        with connect(self.db_path) as db:
            row = db.execute("SELECT * FROM review_items WHERE id = ?", (item_id,)).fetchone()
            if row is None:
                return None
            item = self._item(row)
            item.audit = [
                AuditEntry(
                    seq=a["seq"],
                    at=a["at"],
                    actor=a["actor"],
                    action=a["action"],
                    field=a["field"],
                    before=json.loads(a["before_json"]) if a["before_json"] else None,
                    after=json.loads(a["after_json"]) if a["after_json"] else None,
                    reason=a["reason"],
                )
                for a in db.execute(
                    "SELECT * FROM review_audit WHERE item_id = ? ORDER BY seq", (item_id,)
                ).fetchall()
            ]
            return item

    def correct(
        self, item_id: str, field_name: str, value: Any, *, actor: str, reason: str
    ) -> None:
        """Set one field (already validated by the caller) and append to the audit trail."""
        with connect(self.db_path) as db:
            row = self._pending_row(db, item_id)
            record = json.loads(row["record_json"])
            status = json.loads(row["field_status_json"])
            before = record.get(field_name)
            record[field_name] = value
            status[field_name] = "reviewed"
            db.execute(
                "UPDATE review_items SET record_json = ?, field_status_json = ?, updated_at = ?"
                " WHERE id = ?",
                (_dump(record), _dump(status), _now(), item_id),
            )
            self._audit(db, item_id, actor, "corrected", field_name, before, value, reason)

    def approve(self, item_id: str, *, actor: str, reason: str) -> None:
        """Mark the item approved (publishable) and append to the audit trail."""
        with connect(self.db_path) as db:
            self._pending_row(db, item_id)
            db.execute(
                "UPDATE review_items SET status = 'approved', updated_at = ? WHERE id = ?",
                (_now(), item_id),
            )
            self._audit(db, item_id, actor, "approved", None, None, None, reason)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _pending_row(db: sqlite3.Connection, item_id: str) -> sqlite3.Row:
        row = db.execute("SELECT * FROM review_items WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            raise KeyError(item_id)
        if row["status"] != "pending":
            raise ValueError("the item is already approved")
        return row  # type: ignore[no-any-return]

    @staticmethod
    def _audit(
        db: sqlite3.Connection,
        item_id: str,
        actor: str,
        action: str,
        field_name: str | None,
        before: Any,
        after: Any,
        reason: str | None,
    ) -> None:
        db.execute(
            "INSERT INTO review_audit (item_id, at, actor, action, field, before_json, after_json,"
            " reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                item_id,
                _now(),
                actor,
                action,
                field_name,
                None if before is None else _dump(before),
                None if after is None else _dump(after),
                reason,
            ),
        )

    @staticmethod
    def _item(row: sqlite3.Row) -> ReviewItem:
        return ReviewItem(
            id=row["id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            status=row["status"],
            doc_type=row["doc_type"],
            filename=row["filename"],
            document_sha256=row["document_sha256"],
            source=row["source"],
            record=json.loads(row["record_json"]),
            field_status=json.loads(row["field_status_json"]),
            findings=json.loads(row["findings_json"]),
            review_reasons=json.loads(row["review_reasons_json"]),
            text=row["text"],
        )
