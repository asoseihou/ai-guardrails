"""Structured guardrail output: what was done to each field, and why."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Action(StrEnum):
    """Final disposition of a field after the guardrails ran."""

    OK = "ok"
    CORRECTED = "corrected"  # replaced by a value read/derived deterministically from the document
    NULLED = "nulled"  # value removed: wrong and not recoverable without guessing
    FLAGGED = "flagged"  # value kept but routed to human review
    REVIEWED = "reviewed"  # set or confirmed by a human reviewer (review queue)


class Provenance(StrEnum):
    """Where the published value of a field comes from."""

    MODEL = "model"  # the LLM output, not independently verified
    ACCESS_KEY = "access_key"  # derived from the verified 44-digit access key
    DOCUMENT_TEXT = "document_text"  # literally printed in the document text layer
    XML = "xml"  # the authoritative fiscal XML
    REVIEWER = "reviewer"  # a human correction (validated by the same rules)


VERIFIED_PROVENANCE = frozenset(
    {Provenance.ACCESS_KEY, Provenance.DOCUMENT_TEXT, Provenance.XML, Provenance.REVIEWER}
)
"""Provenances that do not need a further literal-presence check."""


@dataclass
class Finding:
    """One guardrail decision on one field (or on the whole document, ``field='*'``)."""

    field: str
    action: Action
    check: str
    reason: str
    before: Any = None
    after: Any = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation."""
        return {
            "field": self.field,
            "action": str(self.action),
            "check": self.check,
            "reason": self.reason,
            "before": self.before,
            "after": self.after,
        }


@dataclass
class GuardrailResult:
    """The record that would be published, plus every finding that shaped it."""

    doc_type: str
    record: dict[str, Any]
    findings: list[Finding] = field(default_factory=list)
    provenance: dict[str, str] = field(default_factory=dict)
    config: str = "full"
    source: str = "llm"  # "llm", "xml" or "llm+xml"
    review_reasons: list[str] = field(default_factory=list)  # document-level reasons

    def status(self, field_name: str) -> Action:
        """Final status of a field: the action of its last finding (``ok`` if none)."""
        for finding in reversed(self.findings):
            if finding.field == field_name:
                return finding.action
        return Action.OK

    @property
    def field_status(self) -> dict[str, Action]:
        """Final status of every field of the record."""
        return {name: self.status(name) for name in self.record}

    @property
    def needs_review(self) -> bool:
        """True when a human must look at the document before it is published."""
        if self.review_reasons:
            return True
        return any(s in (Action.NULLED, Action.FLAGGED) for s in self.field_status.values())

    def count(self, action: Action) -> int:
        """Number of fields whose *final* status is ``action``."""
        return sum(1 for s in self.field_status.values() if s == action)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation."""
        return {
            "doc_type": self.doc_type,
            "config": self.config,
            "source": self.source,
            "record": self.record,
            "needs_review": self.needs_review,
            "review_reasons": list(self.review_reasons),
            "field_status": {k: str(v) for k, v in self.field_status.items()},
            "provenance": self.provenance,
            "findings": [f.to_dict() for f in self.findings],
        }
