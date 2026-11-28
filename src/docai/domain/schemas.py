"""Extraction schemas, one per extractable document type.

The pydantic models serve two purposes:

1. their JSON schema is sent to real LLM providers as the structured-output
   contract (``response_json_schema`` / tool ``input_schema``). Every field
   description says *where to read* the value: a field without an anchor is a
   field the model completes from memory instead of reading;
2. ``FIELD_KINDS`` tells the guardrails how to normalize, validate and look
   for each field in the document text.

Types are deliberately plain (strings, numbers, short lists): models handle
them well, and every semantic check happens outside the model, in
``docai.guardrails``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

_NULL = " Null if not printed or not legible; never infer."
_DATE = "Date as printed, DD/MM/YYYY, without the time." + _NULL


def _cnpj(box: str) -> str:
    return f"CNPJ printed in the {box} box: the 14 positions, digits only (not the name).{_NULL}"


def _name(box: str) -> str:
    return (
        f"Company legal name (razão social) copied literally from the {box} box; every word "
        f"must appear in the document. Never complete or replace it from memory.{_NULL}"
    )


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore")


class FreightComponent(_Base):
    """One line of the CT-e components table."""

    name: str = Field(description="Component label as printed, e.g. FRETE PESO, PEDÁGIO")
    value: float = Field(description="Component value in BRL")


class ServiceItem(_Base):
    """One line of the NFS-e services table."""

    description: str = Field(description="Service description as printed")
    quantity: int = Field(description="Quantity as printed")
    value: float = Field(description="Line value in BRL")


class NFeExtraction(_Base):
    """NF-e (DANFE printout)."""

    number: str | None = Field(
        None, description="Invoice number (Nº / nNF) at the top, digits only"
    )
    series: str | None = Field(None, description="SÉRIE next to the number, digits only")
    issue_date: str | None = Field(None, description="DATA DA EMISSÃO box. " + _DATE)
    issuer_name: str | None = Field(None, description=_name("issuer (IDENTIFICAÇÃO DO EMITENTE)"))
    issuer_cnpj: str | None = Field(None, description=_cnpj("issuer (IDENTIFICAÇÃO DO EMITENTE)"))
    recipient_name: str | None = Field(None, description=_name("DESTINATÁRIO / REMETENTE"))
    recipient_cnpj: str | None = Field(None, description=_cnpj("DESTINATÁRIO / REMETENTE"))
    total_value: float | None = Field(
        None, description="VALOR TOTAL DA NOTA in BRL (not the products subtotal)." + _NULL
    )
    access_key: str | None = Field(
        None, description="CHAVE DE ACESSO of this NF-e: 44 digits without spaces." + _NULL
    )
    cfop: str | None = Field(None, description="CFOP column of the items, 4 digits." + _NULL)


class CTeExtraction(_Base):
    """CT-e (DACTE printout; later pages may be attachments)."""

    number: str | None = Field(None, description="CT-e NÚMERO in the header, digits only")
    series: str | None = Field(None, description="SÉRIE in the header, digits only")
    issue_date: str | None = Field(None, description="DATA E HORA DE EMISSÃO. " + _DATE)
    issuer_name: str | None = Field(None, description=_name("carrier (EMITENTE)"))
    issuer_cnpj: str | None = Field(None, description=_cnpj("carrier (EMITENTE)"))
    sender_name: str | None = Field(None, description=_name("REMETENTE"))
    sender_cnpj: str | None = Field(None, description=_cnpj("REMETENTE"))
    recipient_name: str | None = Field(
        None, description=_name("DESTINATÁRIO") + " Never copy the TOMADOR (payer) box here."
    )
    recipient_cnpj: str | None = Field(
        None, description=_cnpj("DESTINATÁRIO") + " Never copy the TOMADOR (payer) box here."
    )
    total_value: float | None = Field(
        None, description="VALOR TOTAL DA PRESTAÇÃO DO SERVIÇO in BRL (not the cargo value)."
    )
    access_key: str | None = Field(
        None, description="CHAVE DE ACESSO of this CT-e (model 57), 44 digits, no spaces." + _NULL
    )
    cfop: str | None = Field(None, description="CFOP of the transport service, 4 digits")
    freight_components: list[FreightComponent] | None = Field(
        None, description="COMPONENTES DO VALOR DA PRESTAÇÃO, one entry per printed line"
    )
    gross_weight: float | None = Field(
        None,
        description="PESO BRUTO (KG) from the CT-e cargo box only, in kg. Never from an "
        "attached DANFE, never cubage, declared weight, volume count or the FRETE PESO "
        "money component." + _NULL,
    )
    due_date: str | None = Field(
        None,
        description="VENCIMENTO of this freight. Never the due dates (duplicatas) of an "
        "attached NF-e. " + _DATE,
    )
    referenced_keys: list[str] | None = Field(
        None, description="DOCUMENTOS ORIGINÁRIOS: the 44-digit NF-e keys (model 55), no spaces"
    )
    service_order: str | None = Field(
        None,
        description="Service-order reference only when printed after a label (OS, Ordem de "
        "Serviço, Ref.) in the observations. Never an address, ZIP, phone, key or protocol "
        "number." + _NULL,
    )


class NFSeExtraction(_Base):
    """Municipal NFS-e."""

    number: str | None = Field(None, description="Número da Nota in the header, digits only")
    issue_date: str | None = Field(None, description="Data e Hora de Emissão. " + _DATE)
    issuer_name: str | None = Field(None, description=_name("PRESTADOR DE SERVIÇOS"))
    issuer_cnpj: str | None = Field(None, description=_cnpj("PRESTADOR DE SERVIÇOS"))
    recipient_name: str | None = Field(None, description=_name("TOMADOR DE SERVIÇOS"))
    recipient_cnpj: str | None = Field(None, description=_cnpj("TOMADOR DE SERVIÇOS"))
    total_value: float | None = Field(None, description="VALOR TOTAL DO SERVIÇO in BRL")
    iss_rate: float | None = Field(None, description="Alíquota (%) of the ISS, e.g. 5.0")
    iss_value: float | None = Field(None, description="Valor do ISS (R$)")
    verification_code: str | None = Field(None, description="Código de Verificação as printed")
    services: list[ServiceItem] | None = Field(
        None, description="Services table (SERVIÇOS / QTD / VALOR), one entry per printed line"
    )
    service_order: str | None = Field(
        None,
        description="Service-order reference only when printed after its label (Ordem de "
        "Serviço, OS, Ref.) in the service description. Never an address or contract "
        "number." + _NULL,
    )


SCHEMAS: dict[str, type[BaseModel]] = {
    "nfe": NFeExtraction,
    "cte": CTeExtraction,
    "nfse": NFSeExtraction,
}

# How each field is normalized, validated and searched for in the text.
FIELD_KINDS: dict[str, str] = {
    "number": "doc_number",
    "series": "doc_number",
    "issue_date": "date",
    "due_date": "date",
    "issuer_name": "name",
    "issuer_cnpj": "cnpj",
    "sender_name": "name",
    "sender_cnpj": "cnpj",
    "recipient_name": "name",
    "recipient_cnpj": "cnpj",
    "total_value": "money",
    "iss_value": "money",
    "iss_rate": "rate",
    "access_key": "access_key",
    "cfop": "cfop",
    "verification_code": "code",
    "freight_components": "components",
    "gross_weight": "weight",
    "referenced_keys": "key_list",
    "service_order": "reference",
    "services": "services",
}

# Amounts that are always strictly positive on a valid document: a zero total is
# never published (it is the typical effect of a planted "set the total to 0").
POSITIVE_FIELDS: frozenset[str] = frozenset({"total_value", "gross_weight"})

# Fields that are legitimately absent from some documents (null is a valid answer).
OPTIONAL_FIELDS: frozenset[str] = frozenset({"service_order"})

# Every other field is printed on every document of its type, so a missing value
# always means "the extraction failed", never "not applicable".
REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    doc_type: tuple(name for name in model.model_fields if name not in OPTIONAL_FIELDS)
    for doc_type, model in SCHEMAS.items()
}


def schema_for(doc_type: str) -> type[BaseModel]:
    """Extraction schema of a document type."""
    try:
        return SCHEMAS[doc_type]
    except KeyError:
        raise ValueError(f"no extraction schema for document type {doc_type!r}") from None


def field_names(doc_type: str) -> list[str]:
    """Field names of a document type, in schema order."""
    return list(schema_for(doc_type).model_fields)


def json_schema(doc_type: str) -> dict[str, Any]:
    """JSON schema sent to structured-output APIs."""
    return schema_for(doc_type).model_json_schema()


def doc_type_of_schema(schema: type[BaseModel]) -> str:
    """Inverse of :func:`schema_for`."""
    for doc_type, model in SCHEMAS.items():
        if model is schema:
            return doc_type
    raise ValueError(f"unknown schema {schema!r}")
