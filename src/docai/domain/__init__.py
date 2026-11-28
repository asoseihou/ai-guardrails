"""Domain layer: document schemas, findings and Brazilian fiscal rules.

Pure Python with no I/O. Nothing in here imports from other ``docai`` packages;
``tests/test_architecture.py`` enforces it.
"""

from docai.domain.doc_types import DOC_TYPES, EXTRACTABLE_TYPES, XML_TYPES
from docai.domain.findings import Action, Finding, GuardrailResult, Provenance

__all__ = [
    "DOC_TYPES",
    "EXTRACTABLE_TYPES",
    "XML_TYPES",
    "Action",
    "Finding",
    "GuardrailResult",
    "Provenance",
]
