"""Repository for evaluation runs (the dashboard's time series)."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from docai.storage.db import connect


@dataclass(frozen=True)
class RunRecord:
    """Headline numbers of one evaluation run (full guardrails)."""

    command: str
    config: dict[str, Any]
    n_docs: int
    field_accuracy: float
    review_rate: float
    silent_error_rate: float
    straight_through: float
    cost_per_doc: float
    latency_p50: float
    latency_p95: float
    retries: int
    per_field: dict[str, dict[str, float]]
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation."""
        return asdict(self)


class RunRepository:
    """Append and read runs (one SQLite file, parameterized SQL only)."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def add(self, run: RunRecord) -> str:
        """Store a run and return its id."""
        with connect(self.db_path) as db:
            db.execute(
                "INSERT INTO runs (id, created_at, command, config_json, n_docs, field_accuracy,"
                " review_rate, silent_error_rate, straight_through, cost_per_doc, latency_p50,"
                " latency_p95, retries, per_field_json)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run.id,
                    run.created_at,
                    run.command,
                    json.dumps(run.config),
                    run.n_docs,
                    run.field_accuracy,
                    run.review_rate,
                    run.silent_error_rate,
                    run.straight_through,
                    run.cost_per_doc,
                    run.latency_p50,
                    run.latency_p95,
                    run.retries,
                    json.dumps(run.per_field),
                ),
            )
        return run.id

    def list(self, limit: int = 100) -> list[RunRecord]:
        """The latest ``limit`` runs, oldest first (ready to plot)."""
        with connect(self.db_path) as db:
            rows = db.execute(
                "SELECT * FROM runs ORDER BY created_at DESC, rowid DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            RunRecord(
                id=row["id"],
                created_at=row["created_at"],
                command=row["command"],
                config=json.loads(row["config_json"]),
                n_docs=row["n_docs"],
                field_accuracy=row["field_accuracy"],
                review_rate=row["review_rate"],
                silent_error_rate=row["silent_error_rate"],
                straight_through=row["straight_through"],
                cost_per_doc=row["cost_per_doc"],
                latency_p50=row["latency_p50"],
                latency_p95=row["latency_p95"],
                retries=row["retries"],
                per_field=json.loads(row["per_field_json"]),
            )
            for row in reversed(rows)
        ]
