"""Anchor checks: a value is only right if it was read from the box that owns it.

Literal presence is not enough when the same kind of value is printed several
times on the page: the payer box repeats a party, an attached DANFE prints its
own weight and due dates, every address has a number. These checks re-read the
value from the labelled block that owns it (the DANFE / DACTE / NFS-e layouts
fix those labels) and correct the field when the extraction took it from
somewhere else. Each one is deterministic and records a finding.

They are layout-dependent by design; when the block is not found they do
nothing, so an unknown layout degrades to the generic checks.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from docai.domain.cnpj import is_valid_cnpj
from docai.domain.compare import same_value
from docai.domain.findings import VERIFIED_PROVENANCE, Action
from docai.domain.layout import (
    ROLE_LABELS,
    read_cte_due_date,
    read_cte_weight,
    read_labelled_references,
    read_nfse_services,
    read_referenced_keys,
    read_role_party,
)
from docai.guardrails.checks import FROM_TEXT, FROM_XML, State

MONEY_TOLERANCE = 0.07  # BRL: rounding differences between printed lines and totals


def _correct(state: State, name: str, value: Any, check: str, where: str) -> None:
    """Replace ``name`` with the value re-read from ``where`` (unless the XML owns it)."""
    if name not in state.record:
        return
    current = state.record.get(name)
    if state.provenance.get(name) == FROM_XML:
        if not same_value(name, current, value):
            state.flag(name, check, f"XML value differs from the {where}")
        return
    if not same_value(name, current, value):
        reason = (
            f"missing; re-read from the {where}"
            if current is None
            else f"{current!r} is not what the {where} says; re-read from it"
        )
        state.change(name, value, Action.CORRECTED, check, reason)
    state.provenance[name] = FROM_TEXT


def check_role_anchors(state: State) -> None:
    """Each party is the one printed inside its own role block (payer/sender/recipient traps)."""
    for role, label in ROLE_LABELS.get(state.doc_type, {}).items():
        printed = read_role_party(state.doc.text, label)
        if printed is None or not is_valid_cnpj(printed["cnpj"]):
            continue
        _correct(state, f"{role}_cnpj", printed["cnpj"], "role_anchor", f"{label} block")
        _correct(state, f"{role}_name", printed["name"], "role_anchor", f"{label} block")


def check_cte_main_page(state: State) -> None:
    """CT-e weight and due date come from the CT-e page, never from an attached DANFE."""
    if state.doc_type != "cte":
        return
    readers = (
        ("gross_weight", read_cte_weight, "CT-e cargo box (PESO BRUTO (KG))"),
        ("due_date", read_cte_due_date, "CT-e VENCIMENTO box"),
    )
    for name, reader, where in readers:
        printed = reader(state.doc.text)
        if printed is not None:
            _correct(state, name, printed, "cte_main_page", where)


def check_referenced_keys(state: State) -> None:
    """Referenced NF-e keys: valid check digit, model 55, printed in DOCUMENTOS ORIGINÁRIOS."""
    if "referenced_keys" not in state.record:
        return
    printed = read_referenced_keys(state.doc.text)
    valid = set(state.doc.valid_keys(model="55"))
    current = state.record.get("referenced_keys")
    problems = [] if current is None else _key_problems(state, current)
    if printed and all(k in valid for k in printed) and (current is None or problems):
        _correct(
            state, "referenced_keys", printed, "referenced_keys", "DOCUMENTOS ORIGINÁRIOS list"
        )
    elif problems:
        state.change("referenced_keys", None, Action.NULLED, "referenced_keys", "; ".join(problems))
    elif current is not None:
        state.provenance["referenced_keys"] = FROM_TEXT


def _key_problems(state: State, keys: list[str]) -> list[str]:
    valid = set(state.doc.valid_keys(model="55"))
    problems = []
    for key in keys:
        if key not in valid:
            reason = (
                "is not printed"
                if not state.doc.contains_key(key)
                else "is not a valid model-55 key"
            )
            problems.append(f"referenced key ...{key[-6:]} {reason}")
    return problems


def check_references(state: State) -> None:
    """A service-order reference must be printed right after its label (OS, Ordem de Serviço)."""
    if "service_order" not in state.record or state.provenance.get("service_order") == FROM_XML:
        return
    labelled = read_labelled_references(state.doc.text)
    current = state.record.get("service_order")
    if current is not None and current in labelled:
        state.provenance["service_order"] = FROM_TEXT
        return
    if len(labelled) == 1:
        _correct(state, "service_order", labelled[0], "reference_label", "labelled reference")
    elif current is not None:
        state.change(
            "service_order",
            None,
            Action.NULLED,
            "reference_label",
            f"{current!r} is not printed after a reference label (OS, Ordem de Serviço, Ref.)",
        )


def check_nfse_services(state: State) -> None:
    """NFS-e services must add up to the total (R$ 0.07); re-read the table when it does."""
    if state.doc_type != "nfse" or "services" not in state.record:
        return
    total = state.record.get("total_value")
    printed = read_nfse_services(state.doc.text)
    if printed and state.doc.contains_money(round(sum(s["value"] for s in printed), 2)):
        _correct(state, "services", printed, "nfse_services", "services table")
        return
    current = state.record.get("services")
    off = current and total is not None and abs(sum(s["value"] for s in current) - total)
    if off and off > MONEY_TOLERANCE:
        state.flag("services", "nfse_services", "service lines do not add up to the total")


def check_name_registry(state: State, registry: Mapping[str, str]) -> None:
    """The issuer name follows a *verified* issuer CNPJ (company master data), if it is printed."""
    cnpj = state.record.get("issuer_cnpj")
    if cnpj is None or state.provenance.get("issuer_cnpj") not in VERIFIED_PROVENANCE:
        return
    registered = registry.get(cnpj)
    if registered is None or not state.doc.contains_phrase(registered):
        return
    if state.provenance.get("issuer_name") == FROM_XML:
        return
    current = state.record.get("issuer_name")
    if not same_value("issuer_name", current, registered):
        reason = "missing" if current is None else f"{current!r} is not the registered name"
        state.change(
            "issuer_name",
            registered,
            Action.CORRECTED,
            "name_registry",
            f"{reason}; the name follows the verified issuer CNPJ (registry, printed)",
        )
    state.provenance["issuer_name"] = FROM_TEXT
