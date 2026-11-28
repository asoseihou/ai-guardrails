"""Regex baseline extractor for the synthetic layouts.

Used by the simulated provider when a document has no ground truth (for
example a PDF uploaded through the API). It is intentionally simple: it knows
the label vocabulary of DANFE/DACTE/NFS-e pages, not the documents themselves.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from docai.domain.access_key import MODEL_BY_DOC_TYPE, find_keys_in_text
from docai.domain.layout import (
    MONEY_PATTERN,
    read_cte_due_date,
    read_cte_weight,
    read_freight_components,
    read_labelled_references,
    read_nfse_services,
    read_referenced_keys,
)
from docai.domain.schemas import field_names
from docai.domain.text import digits_only, parse_brl

_CNPJ_RE = re.compile(r"\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}")
_DATE_RE = r"(\d{2}/\d{2}/\d{4})"
_NAME_RE = re.compile(r"NOME / RAZÃO SOCIAL:\s*(.+)")
_LEGAL_SUFFIX = re.compile(r"(Ltda|S\.A\.|ME|LTDA|EIRELI)$")


def _search(pattern: str, text: str, group: int = 1) -> str | None:
    match = re.search(pattern, text)
    return match.group(group) if match else None


def _iso(br_date: str | None) -> str | None:
    if not br_date:
        return None
    return datetime.strptime(br_date, "%d/%m/%Y").date().isoformat()


def _after(text: str, label: str) -> str:
    index = text.find(label)
    return text[index + len(label) :] if index >= 0 else ""


def _cnpj_after(text: str, label: str) -> str | None:
    match = _CNPJ_RE.search(_after(text, label))
    return digits_only(match.group(0)) if match else None


def _name_after(text: str, label: str) -> str | None:
    match = _NAME_RE.search(_after(text, label))
    if not match:
        return None
    name = re.sub(r"(\s+[\d/.,:\-]+)+$", "", match.group(1)).strip()
    return name or None


def _first_company_line(text: str) -> str | None:
    for line in text.splitlines()[:8]:
        line = line.strip()
        if ":" not in line and _LEGAL_SUFFIX.search(line):
            return line
    return None


def _money_after(text: str, label: str, index: int = 0) -> float | None:
    """Money value on the label's line (after it) or, failing that, on the next line."""
    tail = _after(text, label)
    for line in tail.split("\n")[:2]:
        values = re.findall(MONEY_PATTERN, line)
        if len(values) > abs(index) or (index < 0 and values):
            return parse_brl(values[index])
    return None


def _nfe(text: str) -> dict[str, Any]:
    number = _search(r"Nº\s*([\d.]+)", text)
    series = _search(r"SÉRIE\s+(\d+)", text)
    return {
        "number": str(int(digits_only(number))) if number else None,
        "series": str(int(series)) if series else None,
        "issue_date": _iso(_search(r"DATA DA EMISSÃO[\s\S]{0,160}?" + _DATE_RE, text)),
        "total_value": _money_after(text, "VALOR TOTAL DA NOTA", index=-1),
        "cfop": _search(r"\b([1-7]\d{3})\s+UN\b", text),
    }


def _cte(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    match = re.search(r"^57\s+(\d+)\s+(\d+)", text, flags=re.MULTILINE)
    if match:
        out["series"], out["number"] = str(int(match.group(1))), str(int(match.group(2)))
    out["issue_date"] = _iso(_search(r"DATA E HORA DE EMISSÃO[\s\S]{0,120}?" + _DATE_RE, text))
    out["sender_cnpj"] = _cnpj_after(text, "\nREMETENTE")
    out["sender_name"] = _name_after(text, "\nREMETENTE")
    out["total_value"] = _money_after(text, "VALOR TOTAL DA PRESTAÇÃO DO SERVIÇO")
    out["cfop"] = _search(r"\b([1-7]\d{3}) - PRESTAÇÃO", text)
    out["freight_components"] = read_freight_components(text)
    out["gross_weight"] = read_cte_weight(text)
    out["due_date"] = read_cte_due_date(text)
    out["referenced_keys"] = read_referenced_keys(text)
    out["service_order"] = next(iter(read_labelled_references(text)), None)
    return out


def _nfse(text: str) -> dict[str, Any]:
    number = _search(r"Número da Nota[\s\S]{0,120}?\b(\d{8})\b", text)
    return {
        "number": str(int(number)) if number else None,
        "issue_date": _iso(_search(r"Data e Hora de Emissão[\s\S]{0,120}?" + _DATE_RE, text)),
        "issuer_name": _name_after(text, "PRESTADOR"),
        "recipient_cnpj": _cnpj_after(text, "TOMADOR"),
        "recipient_name": _name_after(text, "TOMADOR"),
        "total_value": _money_after(text, "VALOR TOTAL DO SERVIÇO"),
        "iss_rate": _money_after(text, "Valor do ISS (R$)", index=3),
        "iss_value": _money_after(text, "Valor do ISS (R$)", index=4),
        "verification_code": _search(
            r"Código de Verificação\s*\n\s*([A-Z0-9]{4}-[A-Z0-9]{4})", text
        ),
        "services": read_nfse_services(text),
        "service_order": next(iter(read_labelled_references(text)), None),
    }


_BY_TYPE = {"nfe": _nfe, "cte": _cte, "nfse": _nfse}


def heuristic_extract(text: str, doc_type: str) -> dict[str, Any]:
    """Fill the schema of ``doc_type`` with regexes over the synthetic layouts."""
    out: dict[str, Any] = dict.fromkeys(field_names(doc_type))
    first_cnpj = _CNPJ_RE.search(text)
    out["issuer_cnpj"] = digits_only(first_cnpj.group(0)) if first_cnpj else None
    if doc_type in MODEL_BY_DOC_TYPE:
        model = MODEL_BY_DOC_TYPE[doc_type]
        keys = [k for k in find_keys_in_text(text) if k[20:22] == model]
        out["access_key"] = keys[0] if keys else None
        out["issuer_name"] = _first_company_line(text)
        out["recipient_cnpj"] = _cnpj_after(text, "DESTINATÁRIO")
        out["recipient_name"] = _name_after(text, "DESTINATÁRIO")
    out.update(_BY_TYPE[doc_type](text))
    return out
