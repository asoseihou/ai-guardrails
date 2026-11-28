"""Synthetic fiscal XMLs: a simplified NF-e 4.00 (``nfeProc``) and an ABRASF-style NFS-e.

The tag names follow the public layouts (``infNFe``, ``ide/nNF``, ``emit/CNPJ``,
``ICMSTot/vNF``, ``CompNfse/InfNfse``...), but only the fields this project uses
are emitted and nothing is signed: ``tpAmb=2`` marks them as test-environment
documents, like every PDF of the synthetic corpus.
"""

from __future__ import annotations

from typing import Any

# xml.etree (B405) only *builds* XML here; every parse goes through defusedxml.
from xml.etree.ElementTree import (  # nosec B405
    Element,
    SubElement,
    indent,
    tostring,
)

NFE_NS = "http://www.portalfiscal.inf.br/nfe"
NFSE_NS = "http://www.abrasf.org.br/nfse.xsd"


def _add(parent: Element, tag: str, text: object | None = None, **attrs: str) -> Element:
    child = SubElement(parent, tag, attrs)
    if text is not None:
        child.text = str(text)
    return child


def _money(value: float) -> str:
    return f"{value:.2f}"


def _serialize(root: Element) -> bytes:
    indent(root)
    body: bytes = tostring(root, encoding="utf-8", xml_declaration=False)
    return b'<?xml version="1.0" encoding="UTF-8"?>\n' + body


def _nfe_party(parent: Element, tag: str, party: dict[str, str], address_tag: str) -> None:
    node = _add(parent, tag)
    _add(node, "CNPJ", party["cnpj"])
    _add(node, "xNome", party["name"])
    address = _add(node, address_tag)
    _add(address, "xLgr", party["street"])
    _add(address, "xMun", party["city"])
    _add(address, "UF", party["uf"])


def render_nfe_xml(doc: dict[str, Any]) -> bytes:
    """NF-e as authorised by SEFAZ (``nfeProc``: the NF-e plus its protocol)."""
    f, lay = doc["fields"], doc["layout"]
    key = f["access_key"]
    root = Element("nfeProc", {"xmlns": NFE_NS, "versao": "4.00"})
    inf = _add(_add(root, "NFe"), "infNFe", Id=f"NFe{key}", versao="4.00")
    ide = _add(inf, "ide")
    for tag, value in (
        ("cUF", key[:2]),
        ("cNF", key[35:43]),
        ("natOp", lay["nature"]),
        ("mod", "55"),
        ("serie", f["series"]),
        ("nNF", f["number"]),
        ("dhEmi", f"{lay['issued_at'][:19]}-03:00"),
        ("tpNF", "1"),
        ("tpEmis", "1"),
        ("cDV", key[43]),
        ("tpAmb", "2"),
    ):
        _add(ide, tag, value)
    _nfe_party(inf, "emit", lay["issuer"], "enderEmit")
    _nfe_party(inf, "dest", lay["recipient"], "enderDest")
    for number, item in enumerate(lay["items"], start=1):
        prod = _add(_add(inf, "det", nItem=str(number)), "prod")
        for tag, value in (
            ("cProd", item["code"]),
            ("xProd", item["description"]),
            ("NCM", item["ncm"]),
            ("CFOP", item["cfop"]),
            ("uCom", item["unit"]),
            ("qCom", f"{item['qty']:.4f}"),
            ("vUnCom", _money(item["unit_price"])),
            ("vProd", _money(item["total"])),
        ):
            _add(prod, tag, value)
    totals = _add(_add(inf, "total"), "ICMSTot")
    for tag, value in (
        ("vBC", f["total_value"]),
        ("vICMS", lay["icms_value"]),
        ("vProd", f["total_value"]),
        ("vFrete", 0.0),
        ("vNF", f["total_value"]),
    ):
        _add(totals, tag, _money(value))
    if lay.get("notes"):
        _add(_add(inf, "infAdic"), "infCpl", lay["notes"])
    prot = _add(_add(root, "protNFe", versao="4.00"), "infProt")
    _add(prot, "tpAmb", "2")
    _add(prot, "chNFe", key)
    _add(prot, "nProt", lay["protocol"])
    _add(prot, "cStat", "100")
    _add(prot, "xMotivo", "Autorizado o uso da NF-e")
    return _serialize(root)


def render_nfse_xml(doc: dict[str, Any]) -> bytes:
    """Municipal NFS-e in the ABRASF 2.x shape (``CompNfse``)."""
    f, lay = doc["fields"], doc["layout"]
    root = Element("CompNfse", {"xmlns": NFSE_NS})
    inf = _add(_add(root, "Nfse", versao="2.04"), "InfNfse", Id=f"nfse{f['number']}")
    _add(inf, "Numero", f["number"])
    _add(inf, "CodigoVerificacao", f["verification_code"])
    _add(inf, "DataEmissao", lay["issued_at"][:19])
    provider = _add(inf, "PrestadorServico")
    _add(_add(_add(provider, "IdentificacaoPrestador"), "CpfCnpj"), "Cnpj", lay["issuer"]["cnpj"])
    _add(provider, "RazaoSocial", lay["issuer"]["name"])
    declaration = _add(_add(inf, "DeclaracaoPrestacaoServico"), "InfDeclaracaoPrestacaoServico")
    service = _add(declaration, "Servico")
    values = _add(service, "Valores")
    _add(values, "ValorServicos", _money(f["total_value"]))
    _add(values, "ValorIss", _money(f["iss_value"]))
    _add(values, "Aliquota", f"{f['iss_rate']:.2f}")
    _add(service, "ItemListaServico", lay["service_code"])
    _add(service, "Discriminacao", lay["description"])
    taker = _add(declaration, "TomadorServico")
    taker_id = _add(_add(taker, "IdentificacaoTomador"), "CpfCnpj")
    _add(taker_id, "Cnpj", lay["recipient"]["cnpj"])
    _add(taker, "RazaoSocial", lay["recipient"]["name"])
    return _serialize(root)


XML_RENDERERS = {"nfe": render_nfe_xml, "nfse": render_nfse_xml}
