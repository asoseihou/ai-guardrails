"""Deterministic readers for well-known blocks of the printed documents.

Shared by the regex baseline (``heuristic``) and by the guardrails, which use
them to *re-read* a value from the document instead of trusting the model.
"""

from __future__ import annotations

import re
from typing import TypedDict

from docai.domain.text import parse_brl

MONEY_PATTERN = r"\d{1,3}(?:\.\d{3})*,\d{2}"
_COMPONENT_LINE = re.compile(
    rf"^(?P<name>[A-ZÀ-Ü][A-ZÀ-Ü0-9 /.\-]*?)\s+(?P<value>{MONEY_PATTERN})$"
)
_COMPONENTS_HEADER = "COMPONENTES DO VALOR DA PRESTAÇÃO"
_COMPONENTS_END = ("VALOR TOTAL DA PRESTAÇÃO", "VALOR A RECEBER", "DOCUMENTOS ORIGINÁRIOS")


class Component(TypedDict):
    """One line of the CT-e components table."""

    name: str
    value: float


def read_freight_components(text: str) -> list[Component] | None:
    """Read the CT-e components table (one ``NAME VALUE`` per line).

    Returns ``None`` when the block is not found, so callers can tell
    "layout not recognised" apart from "no components".
    """
    lines = [line.strip() for line in text.splitlines()]
    try:
        start = next(i for i, line in enumerate(lines) if line.startswith(_COMPONENTS_HEADER))
    except StopIteration:
        return None
    components: list[Component] = []
    for line in lines[start + 1 :]:
        if line.startswith(_COMPONENTS_END):
            break
        match = _COMPONENT_LINE.match(line)
        if match:
            components.append(
                {"name": match.group("name").strip(), "value": parse_brl(match.group("value"))}
            )
    return components or None


# --------------------------------------------------------------------- pages
PAGE_BREAK = "\f"


def split_pages(text: str) -> list[str]:
    """Pages of a text layer produced by ``docai.extract.pdf`` (separated by form feeds)."""
    return text.split(PAGE_BREAK)


def first_page(text: str) -> str:
    """The main document page. Later pages may be attachments (e.g. a DANFE behind a CT-e)."""
    return split_pages(text)[0]


def _next_line(lines: list[str], label: str) -> str | None:
    for i, line in enumerate(lines):
        if label in line and i + 1 < len(lines):
            return lines[i + 1]
    return None


# --------------------------------------------------------------------- CT-e boxes
_WEIGHT = re.compile(r"(?<![\d.,])\d{1,3}(?:\.\d{3})+,\d{3}(?![\d,])|(?<![\d.,])\d+,\d{3}(?![\d,])")
_DATE = re.compile(r"\b(\d{2})/(\d{2})/(\d{4})\b")


def parse_weight(text: str) -> float:
    """'17.742,292' or '17742,292' -> 17742.292 (kg, three decimals)."""
    return round(float(text.replace(".", "").replace(",", ".")), 3)


def read_cte_weight(text: str) -> float | None:
    """Gross weight from the CT-e cargo box (first page, under ``PESO BRUTO (KG)``)."""
    line = _next_line(first_page(text).splitlines(), "PESO BRUTO (KG)")
    values = _WEIGHT.findall(line or "")
    return parse_weight(values[-1]) if values else None


def read_cte_due_date(text: str) -> str | None:
    """Freight due date (ISO) from the CT-e's own ``VENCIMENTO`` box, never from attachments."""
    line = _next_line(first_page(text).splitlines(), "VENCIMENTO DO FRETE")
    match = _DATE.search(line or "")
    return f"{match.group(3)}-{match.group(2)}-{match.group(1)}" if match else None


def read_referenced_keys(text: str) -> list[str] | None:
    """44-digit keys listed under ``DOCUMENTOS ORIGINÁRIOS`` on the CT-e page."""
    page = first_page(text)
    index = page.find("DOCUMENTOS ORIGINÁRIOS")
    if index < 0:
        return None
    keys = [
        re.sub(r"\D", "", m) for m in re.findall(r"(?<!\d)\d{4}(?: ?\d{4}){10}(?!\d)", page[index:])
    ]
    return keys or None


# --------------------------------------------------------------------- references
REFERENCE_LABEL = re.compile(
    r"\b(?:OS|O\.S\.|ORDEM DE SERVIÇO|ORDEM DE SERVICO|REF\.?|REFERÊNCIA)"
    r"\s*(?:N[º°O]\.?)?\s*:?\s*(?P<value>\d[\dA-Z./-]{2,19})",
    re.IGNORECASE,
)


def read_labelled_references(text: str) -> list[str]:
    """Reference numbers printed right after a label (``OS``, ``Ordem de Serviço``, ``Ref.``)."""
    found: list[str] = []
    for match in REFERENCE_LABEL.finditer(first_page(text)):
        value = match.group("value").rstrip(".-/").upper()
        if value not in found:
            found.append(value)
    return found


# --------------------------------------------------------------------- NFS-e services
class ServiceLine(TypedDict):
    """One line of the NFS-e services table."""

    description: str
    quantity: int
    value: float


_SERVICES_HEADER = "SERVIÇOS QTD VALOR"
_SERVICE_LINE = re.compile(
    rf"^(?P<description>.+?)\s+(?P<quantity>\d+)\s+(?P<value>{MONEY_PATTERN})$"
)


def read_nfse_services(text: str) -> list[ServiceLine] | None:
    """Services table of the NFS-e (``None`` when the block is not found)."""
    lines = [line.strip() for line in first_page(text).splitlines()]
    try:
        start = next(i for i, line in enumerate(lines) if line.startswith(_SERVICES_HEADER))
    except StopIteration:
        return None
    services: list[ServiceLine] = []
    for line in lines[start + 1 :]:
        match = _SERVICE_LINE.match(line)
        if not match:
            break
        services.append(
            {
                "description": match.group("description").strip(),
                "quantity": int(match.group("quantity")),
                "value": parse_brl(match.group("value")),
            }
        )
    return services or None


# --------------------------------------------------------------------- party roles
# The printed role labels are fixed by the DANFE / DACTE / NFS-e layouts.
ROLE_LABELS: dict[str, dict[str, str]] = {
    "nfe": {"recipient": "DESTINATÁRIO / REMETENTE"},
    "cte": {"sender": "REMETENTE", "recipient": "DESTINATÁRIO"},
    "nfse": {"issuer": "PRESTADOR DE SERVIÇOS", "recipient": "TOMADOR DE SERVIÇOS"},
}
_NAME_LINE = re.compile(r"NOME / RAZÃO SOCIAL:\s*(?P<name>.+)")
_CNPJ_LINE = re.compile(
    r"CNPJ:\s*(?P<cnpj>[0-9A-Z]{2}\.[0-9A-Z]{3}\.[0-9A-Z]{3}/[0-9A-Z]{4}-\d{2})"
)


class Party(TypedDict):
    """Name and CNPJ printed inside one role block."""

    name: str
    cnpj: str


def read_role_party(text: str, label: str, window: int = 8) -> Party | None:
    """Name + CNPJ printed in the block that starts with ``label`` (first page only)."""
    lines = [line.strip() for line in first_page(text).splitlines()]
    for i, line in enumerate(lines):
        if not line.startswith(label):
            continue
        name = cnpj = None
        for candidate in lines[i + 1 : i + 1 + window]:
            if name is None and (m := _NAME_LINE.search(candidate)):
                name = re.sub(r"(\s+[\d/.,:\-]+)+$", "", m.group("name")).strip()
            if cnpj is None and (c := _CNPJ_LINE.search(candidate)):
                cnpj = re.sub(r"[^0-9A-Z]", "", c.group("cnpj"))
            if name and cnpj:
                return {"name": name, "cnpj": cnpj}
        return None
    return None


# --------------------------------------------------------------------- routing
_DIGITABLE_LINE = re.compile(r"\b\d{5}\.\d{5} \d{5}\.\d{6} \d{5}\.\d{6} \d \d{14}\b")
_FISCAL_MARKERS = ("CHAVE DE ACESSO", "NFS-E", "DANFE", "DACTE", "NOTA FISCAL DE SERVIÇO")


def looks_like_boleto(text: str) -> bool:
    """A bank slip: the 47-digit digitable line and no fiscal-document marker on the page."""
    page = first_page(text)
    upper = page.upper()
    return bool(_DIGITABLE_LINE.search(page)) and not any(m in upper for m in _FISCAL_MARKERS)
