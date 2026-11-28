"""Document classes handled by the project."""

from __future__ import annotations

DOC_TYPES: tuple[str, ...] = ("nfe", "cte", "nfse", "boleto", "other")
"""Document classes known to the classifier."""

EXTRACTABLE_TYPES: tuple[str, ...] = ("nfe", "cte", "nfse")
"""Document classes that go through extraction + guardrails."""

XML_TYPES: tuple[str, ...] = ("nfe", "nfse")
"""Document classes for which an authoritative XML may accompany the PDF."""
