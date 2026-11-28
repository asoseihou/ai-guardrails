"""Deterministic reader for fiscal XMLs (NF-e ``nfeProc`` and ABRASF-style NFS-e).

The XML is the legally valid document (the PDF is only its printout), so when
it is available its values are taken as they are, without an LLM. Parsing uses
``defusedxml`` with DTDs, entities and external references forbidden (no XXE,
no billion-laughs), after a size check. Tags are matched by local name, so the
namespace prefix does not matter.

What is *not* done: verifying the XML digital signature or querying SEFAZ for
the authorisation status. Those would be the next steps for real documents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# xml.etree (B405) is imported for the Element type only; parsing uses defusedxml.
from xml.etree.ElementTree import Element  # nosec B405

from defusedxml import DefusedXmlException
from defusedxml.ElementTree import ParseError, fromstring

from docai.domain.access_key import is_valid_access_key
from docai.domain.normalize import normalize_value
from docai.domain.schemas import FIELD_KINDS, REQUIRED_FIELDS, field_names

DEFAULT_MAX_XML_BYTES = 1024 * 1024


class XmlRejected(ValueError):
    """The XML is unsafe, malformed or not a supported fiscal document (client-safe message)."""


@dataclass(frozen=True)
class FiscalXml:
    """Fields read from a fiscal XML, already normalized to the extraction schema."""

    doc_type: str
    fields: dict[str, Any]
    problems: list[str] = field(default_factory=list)

    @property
    def missing(self) -> list[str]:
        """Required fields the XML does not provide (the LLM has to fill them)."""
        return [name for name in REQUIRED_FIELDS[self.doc_type] if self.fields.get(name) is None]


def _text(root: Element, path: str) -> str | None:
    node = root.find(path)
    if node is None or node.text is None or not node.text.strip():
        return None
    return node.text.strip()


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _read_nfe(root: Element) -> tuple[dict[str, Any], list[str]]:
    inf = root.find(".//{*}infNFe")
    if inf is None:
        raise XmlRejected("NF-e XML without infNFe")
    key = (inf.get("Id") or "").removeprefix("NFe")
    problems = []
    protocol_key = _text(root, ".//{*}protNFe/{*}infProt/{*}chNFe")
    if protocol_key and protocol_key != key:
        problems.append("infNFe Id and protNFe chNFe differ")
    if not is_valid_access_key(key):
        problems.append("access key in infNFe Id fails the check digit")
    cfops = {node.text.strip() for node in inf.findall(".//{*}det/{*}prod/{*}CFOP") if node.text}
    if len(cfops) > 1:
        problems.append(f"items have {len(cfops)} different CFOPs")
    dh_emi = _text(inf, "{*}ide/{*}dhEmi")
    raw = {
        "number": _text(inf, "{*}ide/{*}nNF"),
        "series": _text(inf, "{*}ide/{*}serie"),
        "issue_date": dh_emi[:10] if dh_emi else None,
        "issuer_name": _text(inf, "{*}emit/{*}xNome"),
        "issuer_cnpj": _text(inf, "{*}emit/{*}CNPJ"),
        "recipient_name": _text(inf, "{*}dest/{*}xNome"),
        "recipient_cnpj": _text(inf, "{*}dest/{*}CNPJ"),
        "total_value": _text(inf, "{*}total/{*}ICMSTot/{*}vNF"),
        "access_key": key or None,
        "cfop": next(iter(cfops)) if len(cfops) == 1 else None,
    }
    return raw, problems


def _read_nfse(root: Element) -> tuple[dict[str, Any], list[str]]:
    inf = root.find(".//{*}InfNfse")
    if inf is None:
        raise XmlRejected("NFS-e XML without InfNfse")
    issued = _text(inf, "{*}DataEmissao")
    values = ".//{*}Servico/{*}Valores/"
    raw = {
        "number": _text(inf, "{*}Numero"),
        "issue_date": issued[:10] if issued else None,
        "issuer_name": _text(inf, ".//{*}PrestadorServico/{*}RazaoSocial"),
        "issuer_cnpj": _text(inf, ".//{*}PrestadorServico//{*}Cnpj"),
        "recipient_name": _text(inf, ".//{*}TomadorServico/{*}RazaoSocial"),
        "recipient_cnpj": _text(inf, ".//{*}TomadorServico//{*}Cnpj"),
        "total_value": _text(inf, values + "{*}ValorServicos"),
        "iss_rate": _text(inf, values + "{*}Aliquota"),
        "iss_value": _text(inf, values + "{*}ValorIss"),
        "verification_code": _text(inf, "{*}CodigoVerificacao"),
    }
    return raw, []


_READERS = {"nfeProc": ("nfe", _read_nfe), "NFe": ("nfe", _read_nfe)}
_READERS |= {"CompNfse": ("nfse", _read_nfse), "Nfse": ("nfse", _read_nfse)}


def read_fiscal_xml(data: bytes, max_bytes: int = DEFAULT_MAX_XML_BYTES) -> FiscalXml:
    """Parse a fiscal XML safely and map it to the extraction schema of its type."""
    if len(data) > max_bytes:
        raise XmlRejected(f"the XML is larger than {max_bytes:,} bytes")
    try:
        root = fromstring(data, forbid_dtd=True, forbid_entities=True, forbid_external=True)
    except DefusedXmlException as exc:
        raise XmlRejected("the XML uses a DTD or entities, which are not allowed") from exc
    except ParseError as exc:
        raise XmlRejected("the file is not well-formed XML") from exc
    reader = _READERS.get(_local_name(root.tag))
    if reader is None:
        raise XmlRejected("not an NF-e (nfeProc) or NFS-e (CompNfse) XML")
    doc_type, read = reader
    raw, problems = read(root)
    fields: dict[str, Any] = {}
    for name in field_names(doc_type):
        try:
            fields[name] = normalize_value(FIELD_KINDS[name], raw.get(name))
        except ValueError:
            fields[name] = None
            problems.append(f"{name} is not a valid {FIELD_KINDS[name]}")
    return FiscalXml(doc_type, fields, problems)
