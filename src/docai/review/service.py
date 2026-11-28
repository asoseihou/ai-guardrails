"""Review use cases on top of the repository: enqueue, correct (validated), approve.

Every mutation is recorded in the append-only audit trail with who, when,
before, after and why. Corrections go through
:func:`docai.guardrails.validate.validate_correction`, so a reviewer cannot
publish a value the guardrails would reject from the model.
"""

from __future__ import annotations

from typing import Any

from docai.domain.schemas import REQUIRED_FIELDS
from docai.guardrails.validate import CorrectionRejected, validate_correction
from docai.pipeline import PipelineResult
from docai.storage.review import ReviewItem, ReviewRepository


class ReviewNotFound(LookupError):
    """No review item with that id."""


class ReviewConflict(ValueError):
    """The item cannot change in its current state (e.g. already approved)."""


class QueueFull(RuntimeError):
    """Too many pending items; resolve some before adding more."""


def reviewer_actor(reviewer: str) -> str:
    """How a reviewer appears in the audit trail.

    One shared API token authenticates *a* trusted operator; the name is
    self-declared, so the trail says so.
    """
    return f"{reviewer} (bearer token)"


class ReviewService:
    """Use cases of the review queue."""

    def __init__(self, repository: ReviewRepository, *, max_pending: int = 1000) -> None:
        self.repository = repository
        self.max_pending = max_pending

    def enqueue(
        self, result: PipelineResult, *, text: str, document_sha256: str, actor: str
    ) -> tuple[str | None, bool]:
        """Store a result that needs review: ``(item_id, created)``, or ``(None, False)``."""
        g = result.guardrails
        if g is None or not g.needs_review:
            return None, False
        if self.repository.count_pending() >= self.max_pending:
            raise QueueFull(f"the review queue already has {self.max_pending} pending items")
        return self.repository.add(
            doc_type=g.doc_type,
            filename=result.filename,
            document_sha256=document_sha256,
            source=g.source,
            record=g.record,
            field_status={k: str(v) for k, v in g.field_status.items()},
            findings=[f.to_dict() for f in g.findings],
            review_reasons=list(g.review_reasons),
            text=text,
            actor=actor,
        )

    def get(self, item_id: str) -> ReviewItem:
        """One item with its audit trail."""
        item = self.repository.get(item_id)
        if item is None:
            raise ReviewNotFound(item_id)
        return item

    def correct(
        self, item_id: str, field: str, value: Any, *, reviewer: str, reason: str
    ) -> ReviewItem:
        """Validate ``value`` with the guardrail rules and store it (``CorrectionRejected``)."""
        item = self._pending(item_id)
        normalized = validate_correction(item.doc_type, field, value, item.record, item.text)
        self.repository.correct(
            item_id, field, normalized, actor=reviewer_actor(reviewer), reason=reason
        )
        return self.get(item_id)

    def approve(self, item_id: str, *, reviewer: str, reason: str) -> ReviewItem:
        """Approve the record as it stands; every required field must have a value."""
        item = self._pending(item_id)
        missing = [f for f in REQUIRED_FIELDS[item.doc_type] if item.record.get(f) is None]
        if missing:
            raise CorrectionRejected([f"correct the empty required fields first: {missing}"])
        self.repository.approve(item_id, actor=reviewer_actor(reviewer), reason=reason)
        return self.get(item_id)

    def _pending(self, item_id: str) -> ReviewItem:
        item = self.get(item_id)
        if item.status != "pending":
            raise ReviewConflict("the item is already approved")
        return item
