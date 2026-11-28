"""Deterministic checks that run outside the model.

Policy — **never invent**. A value is published only if it is

* literally present in the document text, or
* derived from the access key, which is itself printed in the document and
  self-verifying (modulo-11 check digit).

When a value is wrong and the access key (or a unique, self-consistent
re-read of the document) gives the right one, the value is **corrected** and
the finding says where it came from. Otherwise it is **nulled** and the
document goes to human review. Checks that can only raise doubt **flag**.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from docai.domain.access_key import (
    MODEL_BY_DOC_TYPE,
    AccessKey,
    is_valid_access_key,
    parse_access_key,
)
from docai.domain.cnpj import is_valid_cnpj
from docai.domain.compare import same_value
from docai.domain.findings import VERIFIED_PROVENANCE, Action, Finding, Provenance
from docai.domain.layout import read_freight_components
from docai.domain.normalize import normalize_value
from docai.domain.schemas import FIELD_KINDS, POSITIVE_FIELDS, REQUIRED_FIELDS
from docai.guardrails.text_index import DocumentText

CNPJ_FIELDS = ("issuer_cnpj", "sender_cnpj", "recipient_cnpj")
PARTIES = (
    ("issuer_name", "issuer_cnpj"),
    ("sender_name", "sender_cnpj"),
    ("recipient_name", "recipient_cnpj"),
)

FROM_MODEL = Provenance.MODEL
FROM_KEY = Provenance.ACCESS_KEY
FROM_TEXT = Provenance.DOCUMENT_TEXT
FROM_XML = Provenance.XML


@dataclass
class State:
    """Mutable working state shared by the checks of one document."""

    doc_type: str
    record: dict[str, Any]
    doc: DocumentText
    findings: list[Finding] = field(default_factory=list)
    provenance: dict[str, str] = field(default_factory=dict)
    verified_key: AccessKey | None = None

    def change(self, name: str, value: Any, action: Action, check: str, reason: str) -> None:
        """Set ``name`` to ``value`` and record why."""
        before = self.record.get(name)
        self.record[name] = value
        self.findings.append(Finding(name, action, check, reason, before, value))

    def flag(self, name: str, check: str, reason: str) -> None:
        """Keep the value but route the document to human review."""
        value = self.record.get(name)
        self.findings.append(Finding(name, Action.FLAGGED, check, reason, value, value))

    def has_finding(self, name: str) -> bool:
        """Whether any check already acted on ``name``."""
        return any(f.field == name for f in self.findings)


# ------------------------------------------------------------------- schema
def check_schema(state: State, field_names: list[str]) -> None:
    """Coerce each field to its kind; values that cannot be valid are nulled."""
    raw = state.record
    state.record = {}
    for name in field_names:
        state.record[name] = raw.get(name)
        state.provenance[name] = FROM_MODEL
        try:
            value = normalize_value(FIELD_KINDS[name], raw.get(name))
        except (ValueError, TypeError) as exc:
            state.change(name, None, Action.NULLED, "schema", f"invalid {FIELD_KINDS[name]}: {exc}")
            continue
        state.record[name] = value
        if name in POSITIVE_FIELDS and value is not None and value <= 0:
            state.change(name, None, Action.NULLED, "schema", "a document total must be positive")


# ------------------------------------------------------------------- XML
def check_xml_cross(state: State, xml_fields: dict[str, Any]) -> None:
    """The authoritative XML wins: every field that differs from it is corrected."""
    for name in list(state.record):
        expected = xml_fields.get(name)
        if expected is None:
            continue
        current = state.record.get(name)
        if not same_value(name, current, expected):
            reason = (
                "missing; read from the fiscal XML"
                if current is None
                else f"{current!r} differs from the fiscal XML (the XML wins)"
            )
            state.change(name, expected, Action.CORRECTED, "xml_cross", reason)
        state.provenance[name] = FROM_XML


def check_xml_printed(state: State) -> None:
    """XML values should be printed on the PDF; when not, the printout may not match the XML."""
    for name, value in state.record.items():
        if value is None or state.provenance.get(name) != FROM_XML:
            continue
        kind = FIELD_KINDS[name]
        printed = (
            state.doc.contains_key(value)
            if kind == "access_key"
            else is_printed(state.doc, kind, value)
        )
        if printed is False:
            state.flag(name, "xml_printed", "the XML value is not printed on the PDF")


# ------------------------------------------------------------------- access key
def _key_problem(state: State, key: str | None, *, model_check: bool, presence: bool) -> str | None:
    """Why the extracted key cannot be trusted (``None`` when it can)."""
    model = MODEL_BY_DOC_TYPE[state.doc_type]
    if key is None:
        return "missing"
    if not is_valid_access_key(key):
        return "check digit (mod 11) does not match"
    if model_check and key[20:22] != model:
        return f"model {key[20:22]} does not match a {state.doc_type} (expects {model})"
    if presence and not state.doc.contains_key(key):
        return "key not printed in the document"
    return None


def _reread_key(state: State, problem: str) -> tuple[bool, str]:
    """Replace the key by the only valid key of the right model printed in the document."""
    model = MODEL_BY_DOC_TYPE[state.doc_type]
    candidates = state.doc.valid_keys(model=model)
    if len(candidates) != 1:
        return (
            False,
            f"{problem}; {len(candidates)} valid model-{model} keys in the text, cannot re-read",
        )
    state.change(
        "access_key",
        candidates[0],
        Action.CORRECTED,
        "access_key",
        f"{problem}; re-read the only valid model-{model} key printed in the document",
    )
    state.verified_key = parse_access_key(candidates[0])
    state.provenance["access_key"] = FROM_TEXT
    return True, problem


def check_access_key(state: State, *, model_check: bool, presence: bool, reread: bool) -> None:
    """Validate the key (DV, model, literal presence); re-read it from the page if needed."""
    if state.doc_type not in MODEL_BY_DOC_TYPE or "access_key" not in state.record:
        return
    key = state.record.get("access_key")
    from_xml = state.provenance.get("access_key") == FROM_XML
    if from_xml:  # printed-or-not is checked by check_xml_printed; no re-read over the XML
        presence = reread = False
    problem = _key_problem(state, key, model_check=model_check, presence=presence)
    if problem is None:
        state.verified_key = parse_access_key(key)
        if presence and not from_xml:
            state.provenance["access_key"] = FROM_TEXT
        return
    if reread:
        fixed, problem = _reread_key(state, problem)
        if fixed:
            return
    if key is not None:
        state.change("access_key", None, Action.NULLED, "access_key", problem)


def check_key_cross(state: State) -> None:
    """Fields encoded in a verified key must agree with it (key ↔ CNPJ, number, series, date)."""
    key = state.verified_key
    if key is None:
        return
    derived = {
        "issuer_cnpj": (key.cnpj, "positions 7-20"),
        "number": (str(key.number), "positions 26-34"),
        "series": (str(key.series), "positions 23-25"),
    }
    for name, (expected, where) in derived.items():
        if name not in state.record:
            continue
        current = state.record.get(name)
        if state.provenance.get(name) == FROM_XML:
            if current != expected:
                state.flag(name, "key_cross", f"XML value contradicts the access key ({where})")
            continue
        if current != expected:
            reason = (
                f"missing; recovered from the access key ({where})"
                if current is None
                else f"{current!r} contradicts the access key ({where})"
            )
            state.change(name, expected, Action.CORRECTED, "key_cross", reason)
        state.provenance[name] = FROM_KEY
    _check_issue_month(state, key)


def _check_issue_month(state: State, key: AccessKey) -> None:
    issued = state.record.get("issue_date")
    if not issued:
        return
    year, month = int(issued[:4]), int(issued[5:7])
    if (year % 100, month) == (int(key.aamm[:2]), int(key.aamm[2:])):
        return
    reason = f"{month:02d}/{year} contradicts AAMM={key.aamm} in the access key"
    if state.provenance.get("issue_date") == FROM_XML:
        state.flag("issue_date", "key_cross", reason)
    else:
        state.change(
            "issue_date",
            None,
            Action.NULLED,
            "key_cross",
            f"{reason}; the day cannot be recovered",
        )


# ------------------------------------------------------------------- CNPJ
def check_cnpj_dv(state: State) -> None:
    """A CNPJ whose modulo-11 check digits fail cannot be right: null it."""
    for name in CNPJ_FIELDS:
        value = state.record.get(name)
        if value is None or state.provenance.get(name) == FROM_KEY:
            continue
        if not is_valid_cnpj(value):
            state.change(name, None, Action.NULLED, "cnpj_dv", "check digits (mod 11) do not match")


# ------------------------------------------------------------------- presence
def is_printed(doc: DocumentText, kind: str, value: Any) -> bool | None:
    """Whether ``value`` is literally printed (``None``: this kind is checked elsewhere)."""
    if kind == "doc_number":
        return doc.contains_number(value)
    if kind == "cnpj":
        return doc.contains_cnpj(value)
    if kind == "date":
        return doc.contains_date(value)
    if kind in ("money", "rate"):
        return doc.contains_money(value)
    if kind == "name":
        return doc.contains_phrase(value)
    if kind == "cfop":
        return doc.contains_digits(value)
    if kind in ("code", "reference"):
        return doc.contains_code(value)
    if kind == "weight":
        return doc.contains_weight(value)
    if kind == "key_list":
        return all(doc.contains_key(key) for key in value)
    return None  # kinds verified elsewhere (access key, components, services)


def check_presence(state: State) -> None:
    """Every value the model produced must literally appear in the document."""
    for name, value in list(state.record.items()):
        if value is None or state.provenance.get(name) in VERIFIED_PROVENANCE:
            continue
        present = is_printed(state.doc, FIELD_KINDS[name], value)
        if present is None:
            continue
        if present:
            state.provenance[name] = FROM_TEXT
        else:
            state.change(
                name, None, Action.NULLED, "presence", f"{value!r} does not appear in the document"
            )


# ------------------------------------------------------------------- consistency
def _components_key(components: Any) -> list[tuple[str, float]] | None:
    if not components:
        return None
    return sorted((c["name"], round(c["value"], 2)) for c in components)


def check_cte_components(state: State) -> None:
    """Re-read the components table; trust it only when it adds up to a printed total."""
    if state.doc_type != "cte":
        return
    parsed = read_freight_components(state.doc.text)
    current = state.record.get("freight_components")
    total = state.record.get("total_value")
    if parsed:
        parsed_sum = round(sum(c["value"] for c in parsed), 2)
        if not state.doc.contains_money(parsed_sum):
            state.flag(
                "freight_components",
                "cte_components",
                f"components table sums to {parsed_sum:.2f}, which is not printed as a total",
            )
            return
        if _components_key(current) != _components_key(parsed):
            what = "missing" if current is None else "labels/values differ from the printed table"
            state.change(
                "freight_components",
                parsed,
                Action.CORRECTED,
                "cte_components",
                f"{what}; re-read from the components table (sum {parsed_sum:.2f} is printed)",
            )
        state.provenance["freight_components"] = FROM_TEXT
        if total != parsed_sum:
            what = "missing" if total is None else f"{total!r} is not the sum of the components"
            state.change(
                "total_value",
                parsed_sum,
                Action.CORRECTED,
                "cte_components",
                f"{what}; the printed components add up to {parsed_sum:.2f}, also printed",
            )
            state.provenance["total_value"] = FROM_TEXT
    elif current and total is not None and abs(sum(c["value"] for c in current) - total) > 0.07:
        state.flag("freight_components", "cte_components", "components do not add up to the total")


def check_nfse_iss(state: State) -> None:
    """ISS must equal total x rate (R$ 0.07 tolerance); flag otherwise."""
    if state.doc_type != "nfse":
        return
    total, rate, iss = (state.record.get(k) for k in ("total_value", "iss_rate", "iss_value"))
    if total is None or rate is None or iss is None:
        return
    expected = round(total * rate / 100, 2)
    if abs(expected - iss) > 0.07:
        state.flag("iss_value", "nfse_iss", f"ISS {iss:.2f} != total x rate = {expected:.2f}")


def check_parties(state: State) -> None:
    """Different roles must be different establishments, and each name must be bound to its CNPJ."""
    present = [(n, c) for n, c in PARTIES if c in state.record]
    for i, (_, cnpj_a) in enumerate(present):
        for _, cnpj_b in present[i + 1 :]:
            a, b = state.record.get(cnpj_a), state.record.get(cnpj_b)
            if a and a == b:
                state.flag(cnpj_b, "parties", f"same CNPJ as {cnpj_a}")
    for name_field, cnpj_field in present:
        name, cnpj = state.record.get(name_field), state.record.get(cnpj_field)
        if not name or not cnpj:
            continue
        bound = state.doc.cnpj_after_phrase(name)
        if bound and cnpj not in bound:
            state.flag(
                name_field,
                "parties",
                f"name is printed next to CNPJ {bound[0]}, not {cnpj}",
            )


# ------------------------------------------------------------------- required
def check_required(state: State) -> None:
    """A required field that is still empty sends the document to review."""
    for name in REQUIRED_FIELDS.get(state.doc_type, ()):
        if state.record.get(name) is None and not state.has_finding(name):
            state.flag(name, "required", "required field missing; route to human review")
