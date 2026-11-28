"""Validate a human correction with the same rules the guardrails apply to the model.

The never-invent policy applies to reviewers too: a corrected value must be a
valid instance of its kind, pass its check digits, agree with the verified
access key and be printed in the document (labelled, for references; on the
main page, for CT-e weight and due date).
"""

from __future__ import annotations

from typing import Any

from docai.domain.access_key import MODEL_BY_DOC_TYPE, is_valid_access_key, parse_access_key
from docai.domain.cnpj import is_valid_cnpj
from docai.domain.layout import first_page, read_labelled_references
from docai.domain.normalize import normalize_value
from docai.domain.schemas import FIELD_KINDS, POSITIVE_FIELDS, field_names
from docai.guardrails.checks import is_printed
from docai.guardrails.text_index import DocumentText

MAIN_PAGE_FIELDS = frozenset({"gross_weight", "due_date"})


class CorrectionRejected(ValueError):
    """The corrected value breaks a guardrail rule; ``reasons`` explain which."""

    def __init__(self, reasons: list[str]) -> None:
        super().__init__("; ".join(reasons))
        self.reasons = reasons


def _key_problems(doc_type: str, field: str, value: Any, record: dict[str, Any]) -> list[str]:
    key = record.get("access_key")
    if field == "access_key" or not key or not is_valid_access_key(key):
        return []
    parsed = parse_access_key(key)
    derived = {
        "issuer_cnpj": parsed.cnpj,
        "number": str(parsed.number),
        "series": str(parsed.series),
    }
    if field in derived and value != derived[field]:
        return [f"contradicts the access key, which says {derived[field]!r}"]
    if field == "issue_date" and (int(value[2:4]), int(value[5:7])) != (
        int(parsed.aamm[:2]),
        int(parsed.aamm[2:]),
    ):
        return [f"year/month contradict AAMM={parsed.aamm} in the access key"]
    return []


def _kind_problems(
    doc_type: str, field: str, kind: str, value: Any, doc: DocumentText
) -> list[str]:
    if kind == "cnpj" and not is_valid_cnpj(value):
        return ["check digits (mod 11) do not match"]
    if kind == "access_key":
        if not is_valid_access_key(value):
            return ["check digit (mod 11) does not match"]
        model = MODEL_BY_DOC_TYPE.get(doc_type)
        if model and value[20:22] != model:
            return [f"model {value[20:22]} is not a {doc_type} (expects {model})"]
        return [] if doc.contains_key(value) else ["key not printed in the document"]
    if kind == "reference" and value not in read_labelled_references(doc.text):
        return ["not printed after a reference label (OS, Ordem de Serviço, Ref.)"]
    if (
        field in MAIN_PAGE_FIELDS
        and is_printed(DocumentText(first_page(doc.text)), kind, value) is False
    ):
        return ["not printed on the main page of the document"]
    if is_printed(doc, kind, value) is False:
        return ["not printed in the document"]
    return []


def validate_correction(
    doc_type: str, field: str, value: Any, record: dict[str, Any], text: str
) -> Any:
    """Normalized ``value`` for ``field``, or :class:`CorrectionRejected` with the reasons."""
    if field not in field_names(doc_type):
        raise CorrectionRejected([f"{field!r} is not a field of a {doc_type}"])
    kind = FIELD_KINDS[field]
    try:
        normalized = normalize_value(kind, value)
    except (ValueError, TypeError) as exc:
        raise CorrectionRejected([f"invalid {kind}: {exc}"]) from None
    if normalized is None:
        raise CorrectionRejected(["a correction needs a value"])
    if field in POSITIVE_FIELDS and normalized <= 0:
        raise CorrectionRejected(["must be positive"])
    problems = _kind_problems(doc_type, field, kind, normalized, DocumentText(text))
    problems += _key_problems(doc_type, field, normalized, record)
    if problems:
        raise CorrectionRejected(problems)
    return normalized
