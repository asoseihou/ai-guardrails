"""Render synthetic documents as PDFs with reportlab.

The layouts imitate the *structure* of DANFE (NF-e), DACTE (CT-e), municipal
NFS-e, boleto and miscellaneous documents: labelled boxes, an access key printed
in groups of four, totals and tables. Every page carries a footer stating that
the document is synthetic and has no fiscal value.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen.canvas import Canvas

from docai.domain.access_key import format_access_key
from docai.domain.cnpj import format_cnpj

PAGE_W, PAGE_H = A4
MARGIN = 28
CONTENT_W = PAGE_W - 2 * MARGIN
FOOTER = "DOCUMENTO SINTÉTICO GERADO PARA TESTES - SEM VALOR FISCAL"


# ---------------------------------------------------------------- formatting
def brl(value: float) -> str:
    """Brazilian money format: 1234.5 -> '1.234,50'."""
    text = f"{value:,.2f}"
    return text.replace(",", "X").replace(".", ",").replace("X", ".")


def br_date(iso: str) -> str:
    """ISO date -> DD/MM/YYYY."""
    return date.fromisoformat(iso[:10]).strftime("%d/%m/%Y")


def br_datetime(iso: str) -> str:
    """ISO datetime -> DD/MM/YYYY HH:MM."""
    return datetime.fromisoformat(iso).strftime("%d/%m/%Y %H:%M")


def weight_br(kg: float) -> str:
    """Weight in kg with three decimals, Brazilian style: 17742.292 -> '17742,292'."""
    return f"{kg:.3f}".replace(".", ",")


def dotted_number(number: str | int) -> str:
    """'12345' -> '000.012.345' (how DANFE prints nNF)."""
    padded = f"{int(number):09d}"
    return f"{padded[:3]}.{padded[3:6]}.{padded[6:]}"


# ---------------------------------------------------------------- page helper
class Page:
    """Tiny wrapper that uses top-left coordinates (y grows downwards)."""

    def __init__(self, path: Path) -> None:
        self.canvas = Canvas(str(path), pagesize=A4, invariant=1)
        self.canvas.setTitle("Synthetic document")
        self.canvas.setLineWidth(0.6)

    def _y(self, y: float) -> float:
        return PAGE_H - y

    def box(self, x: float, y: float, w: float, h: float, title: str | None = None) -> None:
        """Rectangle with an optional small title."""
        self.canvas.rect(x, self._y(y + h), w, h)
        if title:
            self.text(x + 3, y + 8, title, size=6)

    def text(
        self,
        x: float,
        y: float,
        value: str,
        size: float = 8,
        bold: bool = False,
        mono: bool = False,
        right: bool = False,
    ) -> None:
        """Draw a string (top-left coordinates)."""
        font = "Courier" if mono else ("Helvetica-Bold" if bold else "Helvetica")
        self.canvas.setFont(font, size)
        if right:
            self.canvas.drawRightString(x, self._y(y), value)
        else:
            self.canvas.drawString(x, self._y(y), value)

    def labelled(self, x: float, y: float, w: float, label: str, value: str, **kw: Any) -> None:
        """A field box: small label on top, value below."""
        self.box(x, y, w, 24, label)
        self.text(x + 3, y + 19, value, size=kw.get("size", 8.5), bold=kw.get("bold", False))

    def section(self, y: float, title: str) -> None:
        """Bold section heading."""
        self.text(MARGIN, y, title, size=7, bold=True)

    def barcode(self, x: float, y: float, w: float, h: float, seed: str) -> None:
        """Decorative barcode derived from ``seed``."""
        cursor = x
        for i, char in enumerate(seed * 3):
            bar = 0.6 + (ord(char) + i) % 3 * 0.6
            if i % 2 == 0:
                self.canvas.rect(cursor, self._y(y + h), bar, h, stroke=0, fill=1)
            cursor += bar + 0.7
            if cursor > x + w:
                break

    def new_page(self) -> None:
        """Finish the current page (with its footer) and start another one."""
        self.footer()
        self.canvas.showPage()
        self.canvas.setLineWidth(0.6)

    def footer(self) -> None:
        """The synthetic-document disclaimer."""
        self.text(PAGE_W / 2 - 150, PAGE_H - 18, FOOTER, size=7, bold=True)

    def save(self) -> None:
        """Finish the page and write the PDF."""
        self.footer()
        self.canvas.showPage()
        self.canvas.save()


def _party_block(
    page: Page, x: float, y: float, w: float, title: str, party: dict[str, str]
) -> None:
    page.box(x, y, w, 58, title)
    page.text(x + 4, y + 20, f"NOME / RAZÃO SOCIAL: {party['name']}", size=8, bold=True)
    page.text(x + 4, y + 32, f"ENDEREÇO: {party['street']}", size=7.5)
    page.text(x + 4, y + 43, f"MUNICÍPIO: {party['city']} UF: {party['uf']}", size=7.5)
    page.text(x + 4, y + 54, f"CNPJ: {format_cnpj(party['cnpj'])}", size=8)


# ---------------------------------------------------------------- NF-e (DANFE)
def render_nfe(doc: dict[str, Any], path: Path) -> None:
    """Draw a DANFE-like NF-e page."""
    f, lay = doc["fields"], doc["layout"]
    issuer, recipient = lay["issuer"], lay["recipient"]
    p = Page(path)
    y = 30
    p.box(MARGIN, y, 260, 92, "IDENTIFICAÇÃO DO EMITENTE")
    p.text(MARGIN + 6, y + 26, issuer["name"], size=10, bold=True)
    p.text(MARGIN + 6, y + 42, issuer["street"], size=7.5)
    p.text(MARGIN + 6, y + 54, f"{issuer['city']} - {issuer['uf']}", size=7.5)
    p.text(MARGIN + 6, y + 70, f"CNPJ: {format_cnpj(issuer['cnpj'])}", size=8.5)
    x2 = MARGIN + 260
    p.box(x2, y, 118, 92)
    p.text(x2 + 32, y + 16, "DANFE", size=14, bold=True)
    p.text(x2 + 6, y + 28, "Documento Auxiliar da Nota", size=6.5)
    p.text(x2 + 6, y + 36, "Fiscal Eletrônica", size=6.5)
    p.text(x2 + 6, y + 50, "0 - ENTRADA   1 - SAÍDA   [1]", size=6.5)
    p.text(x2 + 6, y + 64, f"Nº {dotted_number(f['number'])}", size=9, bold=True)
    p.text(x2 + 6, y + 76, f"SÉRIE {int(f['series']):03d}", size=8)
    p.text(x2 + 6, y + 87, "FOLHA 1/1", size=7)
    x3 = x2 + 118
    p.box(x3, y, MARGIN + CONTENT_W - x3, 92, "CONTROLE DO FISCO")
    p.barcode(x3 + 8, y + 16, 150, 30, f["access_key"])
    p.text(x3 + 6, y + 62, "Consulta de autenticidade no portal", size=6.5)
    p.text(x3 + 6, y + 70, "nacional da NF-e", size=6.5)
    y += 92
    p.box(MARGIN, y, CONTENT_W, 26, "CHAVE DE ACESSO")
    p.text(MARGIN + 6, y + 20, format_access_key(f["access_key"]), size=10, mono=True)
    y += 26
    p.labelled(MARGIN, y, 300, "NATUREZA DA OPERAÇÃO", lay["nature"])
    p.labelled(
        MARGIN + 300,
        y,
        CONTENT_W - 300,
        "PROTOCOLO DE AUTORIZAÇÃO DE USO",
        f"{lay['protocol']} - {br_datetime(lay['issued_at'])}",
    )
    y += 36
    p.section(y, "DESTINATÁRIO / REMETENTE")
    y += 4
    _party_block(p, MARGIN, y, 400, "", recipient)
    p.labelled(MARGIN + 400, y, CONTENT_W - 400, "DATA DA EMISSÃO", br_date(f["issue_date"]))
    p.labelled(MARGIN + 400, y + 30, CONTENT_W - 400, "DATA DA SAÍDA", br_date(f["issue_date"]))
    y += 70
    p.section(y, "CÁLCULO DO IMPOSTO")
    y += 4
    total = f["total_value"]
    cols = [
        ("BASE DE CÁLCULO DO ICMS", brl(total)),
        ("VALOR DO ICMS", brl(lay["icms_value"])),
        ("VALOR TOTAL DOS PRODUTOS", brl(total)),
        ("VALOR DO FRETE", "0,00"),
        ("VALOR TOTAL DA NOTA", brl(total)),
    ]
    width = CONTENT_W / len(cols)
    for i, (label, value) in enumerate(cols):
        p.labelled(MARGIN + i * width, y, width, label, value, bold=i == len(cols) - 1)
    y += 36
    p.section(y, "DADOS DOS PRODUTOS / SERVIÇOS")
    y += 4
    headers = [
        ("CÓDIGO", 0),
        ("DESCRIÇÃO DO PRODUTO", 48),
        ("NCM", 232),
        ("CFOP", 282),
        ("UN", 318),
        ("QTD", 360),
        ("V. UNITÁRIO", 440),
        ("V. TOTAL", 535),
    ]
    p.box(MARGIN, y, CONTENT_W, 16 + 13 * len(lay["items"]))
    for label, dx in headers:
        right = label in {"QTD", "V. UNITÁRIO", "V. TOTAL"}
        p.text(MARGIN + 4 + dx, y + 10, label, size=6.5, bold=True, right=right)
    row_y = y + 24
    for item in lay["items"]:
        p.text(MARGIN + 4, row_y, item["code"], size=7)
        p.text(MARGIN + 52, row_y, item["description"], size=7)
        p.text(MARGIN + 236, row_y, item["ncm"], size=7)
        p.text(MARGIN + 286, row_y, item["cfop"], size=7)
        p.text(MARGIN + 322, row_y, item["unit"], size=7)
        p.text(MARGIN + 364, row_y, str(item["qty"]), size=7, right=True)
        p.text(MARGIN + 444, row_y, brl(item["unit_price"]), size=7, right=True)
        p.text(MARGIN + 539, row_y, brl(item["total"]), size=7, right=True)
        row_y += 13
    y = row_y + 10
    p.section(y, "DADOS ADICIONAIS")
    p.box(MARGIN, y + 4, CONTENT_W, 40, "INFORMAÇÕES COMPLEMENTARES")
    p.text(
        MARGIN + 4, y + 26, "Documento emitido por ME ou EPP optante pelo Simples Nacional: NÃO."
    )
    if lay.get("notes"):  # free text written by the issuer (used to test prompt injection)
        p.text(MARGIN + 4, y + 37, str(lay["notes"])[:110], size=7.5)
    p.save()


# ---------------------------------------------------------------- CT-e (DACTE)
def render_cte(doc: dict[str, Any], path: Path) -> None:
    """Draw a DACTE-like CT-e page (plus the attached DANFE page, if any)."""
    f, lay = doc["fields"], doc["layout"]
    p = Page(path)
    y = _cte_header(p, f, lay)
    y = _cte_parties(p, lay, y)
    y = _cte_values(p, f, lay, y)
    _cte_documents(p, doc, y)
    p.save()


def _cte_header(p: Page, f: dict[str, Any], lay: dict[str, Any]) -> float:
    issuer = lay["issuer"]
    y = 30
    p.box(MARGIN, y, 250, 84, "EMITENTE")
    p.text(MARGIN + 6, y + 24, issuer["name"], size=10, bold=True)
    p.text(MARGIN + 6, y + 40, issuer["street"], size=7.5)
    p.text(MARGIN + 6, y + 52, f"{issuer['city']} - {issuer['uf']}", size=7.5)
    p.text(MARGIN + 6, y + 68, f"CNPJ: {format_cnpj(issuer['cnpj'])}", size=8.5)
    x2 = MARGIN + 250
    p.box(x2, y, 130, 84)
    p.text(x2 + 40, y + 18, "DACTE", size=14, bold=True)
    p.text(x2 + 6, y + 32, "Documento Auxiliar do Conhecimento", size=6)
    p.text(x2 + 6, y + 40, "de Transporte Eletrônico", size=6)
    p.text(x2 + 6, y + 58, "MODAL", size=6.5)
    p.text(x2 + 6, y + 70, "RODOVIÁRIO", size=8, bold=True)
    x3 = x2 + 130
    p.box(x3, y, MARGIN + CONTENT_W - x3, 84)
    p.text(x3 + 6, y + 14, "MODELO    SÉRIE    NÚMERO      FL", size=6.5)
    p.text(
        x3 + 6,
        y + 26,
        f"57        {f['series']}        {f['number']}      1/1",
        size=8.5,
        bold=True,
    )
    p.text(x3 + 6, y + 44, "DATA E HORA DE EMISSÃO", size=6.5)
    p.text(x3 + 6, y + 56, br_datetime(lay["issued_at"]), size=8.5)
    p.text(x3 + 6, y + 74, "TIPO DO CT-E: NORMAL", size=6.5)
    y += 84
    p.box(MARGIN, y, CONTENT_W, 26, "CHAVE DE ACESSO")
    p.text(MARGIN + 6, y + 20, format_access_key(f["access_key"]), size=10, mono=True)
    y += 26
    p.labelled(MARGIN, y, 200, "PROTOCOLO DE AUTORIZAÇÃO DE USO", lay["protocol"])
    nature = "PRESTAÇÃO DE SERVIÇO DE TRANSPORTE A ESTABELECIMENTO COMERCIAL"
    p.labelled(
        MARGIN + 200,
        y,
        CONTENT_W - 200,
        "CFOP - NATUREZA DA PRESTAÇÃO",
        f"{f['cfop']} - {nature}",
        size=7.5,
    )
    y += 30
    sender, recipient = lay["sender"], lay["recipient"]
    p.labelled(
        MARGIN, y, CONTENT_W / 2, "INÍCIO DA PRESTAÇÃO", f"{sender['city']} - {sender['uf']}"
    )
    p.labelled(
        MARGIN + CONTENT_W / 2,
        y,
        CONTENT_W / 2,
        "TÉRMINO DA PRESTAÇÃO",
        f"{recipient['city']} - {recipient['uf']}",
    )
    y += 30
    return y


def _cte_parties(p: Page, lay: dict[str, Any], y: float) -> float:
    sender, recipient = lay["sender"], lay["recipient"]
    p.section(y, "REMETENTE")
    y += 4
    _party_block(p, MARGIN, y, CONTENT_W, "", sender)
    y += 64
    p.section(y, "DESTINATÁRIO")
    y += 4
    _party_block(p, MARGIN, y, CONTENT_W, "", recipient)
    y += 64
    taker = sender if lay["taker"] == "sender" else recipient
    p.labelled(
        MARGIN,
        y,
        CONTENT_W,
        "TOMADOR DO SERVIÇO",
        f"{taker['name']} - CNPJ: {format_cnpj(taker['cnpj'])}",
    )
    y += 30
    return y


def _cte_values(p: Page, f: dict[str, Any], lay: dict[str, Any], y: float) -> float:
    p.labelled(MARGIN, y, 220, "PRODUTO PREDOMINANTE", lay["cargo"])
    p.labelled(MARGIN + 220, y, 160, "VALOR TOTAL DA CARGA", brl(lay["cargo_value"]))
    p.labelled(MARGIN + 380, y, CONTENT_W - 380, "PESO BRUTO (KG)", weight_br(f["gross_weight"]))
    y += 34
    p.section(y, "COMPONENTES DO VALOR DA PRESTAÇÃO DO SERVIÇO")
    y += 4
    comps = f["freight_components"]
    p.box(MARGIN, y, 300, 18 + 12 * len(comps))
    p.text(MARGIN + 6, y + 11, "NOME", size=6.5, bold=True)
    p.text(MARGIN + 290, y + 11, "VALOR", size=6.5, bold=True, right=True)
    row_y = y + 24
    for comp in comps:
        p.text(MARGIN + 6, row_y, comp["name"], size=8)
        p.text(MARGIN + 290, row_y, brl(comp["value"]), size=8, right=True)
        row_y += 12
    y = row_y + 6
    p.labelled(
        MARGIN, y, 300, "VALOR TOTAL DA PRESTAÇÃO DO SERVIÇO", brl(f["total_value"]), bold=True
    )
    p.labelled(MARGIN + 300, y, CONTENT_W - 300, "VALOR A RECEBER", brl(f["total_value"]))
    y += 30
    p.labelled(MARGIN, y, 200, "VENCIMENTO DO FRETE", br_date(f["due_date"]))
    observation = f"OS {f['service_order']}" if f["service_order"] else "-"
    p.labelled(MARGIN + 200, y, CONTENT_W - 200, "OBSERVAÇÕES", observation)
    y += 34
    return y


def _cte_documents(p: Page, doc: dict[str, Any], y: float) -> None:
    lay = doc["layout"]
    p.section(y, "DOCUMENTOS ORIGINÁRIOS")
    y += 4
    p.box(MARGIN, y, CONTENT_W, 16 + 13 * len(lay["referenced_keys"]))
    p.text(MARGIN + 6, y + 11, "TIPO DOC", size=6.5, bold=True)
    p.text(MARGIN + 70, y + 11, "CHAVE DA NF-E", size=6.5, bold=True)
    row_y = y + 24
    for ref in lay["referenced_keys"]:
        p.text(MARGIN + 6, row_y, "NF-e", size=8)
        p.text(MARGIN + 70, row_y, format_access_key(ref), size=8, mono=True)
        row_y += 13
    if lay.get("notes"):  # free text written by the issuer (used to test prompt injection)
        p.section(row_y + 8, "INFORMAÇÕES ADICIONAIS")
        p.text(MARGIN + 4, row_y + 20, str(lay["notes"])[:110], size=7.5)
    if lay.get("attachment"):
        p.new_page()
        _render_attached_danfe(p, doc)


def _render_attached_danfe(p: Page, doc: dict[str, Any]) -> None:
    """Second page: the DANFE of the cargo, attached to the CT-e (a classic source of traps)."""
    att, sender = doc["layout"]["attachment"], doc["layout"]["sender"]
    y = 40
    p.text(MARGIN, y, "ANEXO - DANFE DA MERCADORIA TRANSPORTADA", size=11, bold=True)
    y += 16
    p.text(MARGIN, y, f"NF-e Nº {dotted_number(att['number'])}  EMITENTE: {sender['name']}", size=8)
    y += 10
    p.box(MARGIN, y, CONTENT_W, 26, "CHAVE DE ACESSO DA NF-E")
    p.text(MARGIN + 6, y + 20, format_access_key(att["key"]), size=10, mono=True)
    y += 30
    p.labelled(MARGIN, y, 180, "PESO BRUTO", weight_br(att["gross_weight"]))
    p.labelled(MARGIN + 180, y, 180, "VALOR TOTAL DA NOTA", brl(att["value"]))
    y += 34
    p.section(y, "FATURA / DUPLICATAS")
    y += 4
    p.box(MARGIN, y, CONTENT_W, 22)
    p.text(
        MARGIN + 6,
        y + 15,
        f"DUP 001  VENC. {br_date(att['due_date'])}  VALOR {brl(att['value'])}",
        size=8.5,
    )


# ---------------------------------------------------------------- NFS-e
def render_nfse(doc: dict[str, Any], path: Path) -> None:
    """Draw a municipal NFS-e page."""
    f, lay = doc["fields"], doc["layout"]
    issuer, recipient = lay["issuer"], lay["recipient"]
    p = Page(path)
    y = 30
    p.box(MARGIN, y, 360, 70)
    p.text(
        MARGIN + 8, y + 20, f"PREFEITURA MUNICIPAL DE {issuer['city'].upper()}", size=11, bold=True
    )
    p.text(MARGIN + 8, y + 34, "SECRETARIA MUNICIPAL DA FAZENDA", size=8)
    p.text(MARGIN + 8, y + 54, "NOTA FISCAL DE SERVIÇOS ELETRÔNICA - NFS-e", size=10, bold=True)
    x2 = MARGIN + 360
    w2 = CONTENT_W - 360
    p.box(x2, y, w2, 70)
    p.text(x2 + 6, y + 12, "Número da Nota", size=6.5)
    p.text(x2 + 6, y + 23, f"{int(f['number']):08d}", size=9, bold=True)
    p.text(x2 + 6, y + 36, "Data e Hora de Emissão", size=6.5)
    p.text(x2 + 6, y + 47, br_datetime(lay["issued_at"]), size=8.5)
    p.text(x2 + 6, y + 58, "Código de Verificação", size=6.5)
    p.text(x2 + 6, y + 68, f["verification_code"], size=8.5, bold=True)
    y += 80
    p.section(y, "PRESTADOR DE SERVIÇOS")
    _party_block(p, MARGIN, y + 4, CONTENT_W, "", issuer)
    y += 70
    p.section(y, "TOMADOR DE SERVIÇOS")
    _party_block(p, MARGIN, y + 4, CONTENT_W, "", recipient)
    y += 70
    p.section(y, "DISCRIMINAÇÃO DOS SERVIÇOS")
    services = f["services"]
    height = 62 + 12 * len(services)
    p.box(MARGIN, y + 4, CONTENT_W, height)
    p.text(MARGIN + 6, y + 18, lay["description"], size=8)
    if f["service_order"]:
        p.text(MARGIN + 6, y + 30, f"ORDEM DE SERVIÇO Nº {f['service_order']}", size=8)
    p.text(
        MARGIN + 6,
        y + 42,
        f"Código do Serviço: {lay['service_code']} - {lay['service_name']}",
        size=7.5,
    )
    p.text(MARGIN + 6, y + 56, "SERVIÇOS", size=6.5, bold=True)
    p.text(MARGIN + 400, y + 56, "QTD", size=6.5, bold=True, right=True)
    p.text(MARGIN + 530, y + 56, "VALOR (R$)", size=6.5, bold=True, right=True)
    row_y = y + 68
    for item in services:
        p.text(MARGIN + 6, row_y, item["description"], size=8)
        p.text(MARGIN + 400, row_y, str(item["quantity"]), size=8, right=True)
        p.text(MARGIN + 530, row_y, brl(item["value"]), size=8, right=True)
        row_y += 12
    y += height + 14
    p.box(MARGIN, y, CONTENT_W, 24)
    p.text(
        MARGIN + 6,
        y + 16,
        f"VALOR TOTAL DO SERVIÇO = R$ {brl(f['total_value'])}",
        size=10,
        bold=True,
    )
    y += 30
    cols = [
        ("Valor dos Serviços (R$)", brl(f["total_value"])),
        ("Deduções (R$)", "0,00"),
        ("Base de Cálculo (R$)", brl(f["total_value"])),
        ("Alíquota (%)", brl(f["iss_rate"])),
        ("Valor do ISS (R$)", brl(f["iss_value"])),
    ]
    width = CONTENT_W / len(cols)
    for i, (label, value) in enumerate(cols):
        p.labelled(MARGIN + i * width, y, width, label, value)
    y += 30
    federal = ["PIS (R$)", "COFINS (R$)", "INSS (R$)", "IR (R$)", "CSLL (R$)"]
    for i, label in enumerate(federal):
        p.labelled(MARGIN + i * width, y, width, label, "0,00")
    y += 34
    p.text(MARGIN, y, "ISS Retido: Não    Regime Especial de Tributação: Nenhum", size=7.5)
    p.save()


# ---------------------------------------------------------------- boleto
def render_boleto(doc: dict[str, Any], path: Path) -> None:
    """Draw a boleto (bank slip)."""
    lay = doc["layout"]
    ben, payer = lay["beneficiary"], lay["payer"]
    p = Page(path)
    y = 40
    p.text(MARGIN, y, lay["bank_name"], size=12, bold=True)
    p.text(MARGIN + 190, y, f"| {lay['bank_code']}-9 |", size=12, bold=True)
    p.text(MARGIN + 260, y, lay["line"], size=9, mono=True)
    y += 10
    p.labelled(MARGIN, y, 400, "LOCAL DE PAGAMENTO", "PAGÁVEL EM QUALQUER BANCO ATÉ O VENCIMENTO")
    p.labelled(MARGIN + 400, y, CONTENT_W - 400, "VENCIMENTO", br_date(lay["due_date"]), bold=True)
    y += 24
    p.labelled(MARGIN, y, 400, "BENEFICIÁRIO", f"{ben['name']} - CNPJ {format_cnpj(ben['cnpj'])}")
    p.labelled(MARGIN + 400, y, CONTENT_W - 400, "NOSSO NÚMERO", lay["our_number"])
    y += 24
    p.labelled(MARGIN, y, 400, "DATA DO DOCUMENTO", br_date(lay["due_date"]))
    p.labelled(
        MARGIN + 400, y, CONTENT_W - 400, "(=) VALOR DO DOCUMENTO", brl(lay["amount"]), bold=True
    )
    y += 24
    p.box(MARGIN, y, CONTENT_W, 60, "INSTRUÇÕES (TEXTO DE RESPONSABILIDADE DO BENEFICIÁRIO)")
    p.text(MARGIN + 6, y + 22, "NÃO RECEBER APÓS 30 DIAS DO VENCIMENTO.", size=8)
    p.text(MARGIN + 6, y + 34, "APÓS O VENCIMENTO COBRAR MULTA DE 2% E JUROS DE 1% AO MÊS.", size=8)
    if lay["reference"]:
        p.text(MARGIN + 6, y + 46, lay["reference"], size=8)
    y += 60
    p.labelled(
        MARGIN,
        y,
        CONTENT_W,
        "PAGADOR",
        f"{payer['name']} - CNPJ {format_cnpj(payer['cnpj'])} - {payer['city']}/{payer['uf']}",
    )
    y += 34
    p.barcode(MARGIN, y, 320, 40, lay["line"].replace(" ", ""))
    p.text(MARGIN + 360, y + 20, "AUTENTICAÇÃO MECÂNICA", size=7)
    p.text(MARGIN + 360, y + 32, "FICHA DE COMPENSAÇÃO", size=8, bold=True)
    p.save()


# ---------------------------------------------------------------- other
def render_other(doc: dict[str, Any], path: Path) -> None:
    """Draw a non-fiscal document (CC-e, delivery receipt, ...)."""
    lay = doc["layout"]
    a, b = lay["company"], lay["counterpart"]
    kind = lay["kind"]
    p = Page(path)
    y = 40
    p.text(MARGIN, y, a["name"], size=11, bold=True)
    p.text(MARGIN, y + 12, f"CNPJ: {format_cnpj(a['cnpj'])} - {a['city']}/{a['uf']}", size=8)
    y += 40
    when = br_date(lay["date"])
    ref = lay["ref_number"]
    lines: list[str]
    if kind == "cce":
        title = "CARTA DE CORREÇÃO ELETRÔNICA - CC-e"
        lines = [
            f"Chave de acesso da NF-e: {format_access_key(lay['ref_key'])}",
            f"Data do evento: {when}   Sequência do evento: 1",
            "Correção: onde se lê PESO BRUTO 120 KG, leia-se PESO BRUTO 125 KG.",
            "A Carta de Correção não pode alterar valores, dados cadastrais do",
            "remetente ou do destinatário, nem a data de emissão ou de saída.",
        ]
    elif kind == "delivery":
        title = "COMPROVANTE DE ENTREGA"
        lines = [
            f"Recebemos de {a['name']} os volumes referentes ao CT-e nº {ref}.",
            f"Destinatário: {b['name']}",
            f"Data do recebimento: {when}",
            "Volumes conferidos sem avarias aparentes.",
            "Assinatura do recebedor: ______________________   Documento: __________",
        ]
    elif kind == "service_order":
        title = f"ORDEM DE SERVIÇO Nº {ref}"
        lines = [
            f"Cliente: {b['name']}",
            f"Data de abertura: {when}",
            "Descrição: inspeção e manutenção preventiva de equipamentos.",
            "Prazo de execução: 5 dias úteis após aprovação do orçamento.",
            "Responsável técnico: ______________________",
        ]
    elif kind == "inspection":
        title = "RELATÓRIO DE INSPEÇÃO DE CARGA"
        lines = [
            f"Inspeção realizada em {when} nas instalações de {b['name']}.",
            "Embalagens: íntegras.   Lacres: conferidos.   Umidade: ausente.",
            "Amarração da carga: adequada.   Sinalização: conforme.",
            f"Referência interna: INSP-{ref}",
            "Conclusão: carga liberada para transporte.",
        ]
    else:
        title = "PACKING LIST"
        lines = [f"Destinatário: {b['name']}", f"Data: {when}   Referência: PL-{ref}", ""]
        lines += [f"{it['qty']:>4} x {it['description']}" for it in lay["items"]]
    p.text(MARGIN, y, title, size=13, bold=True)
    y += 24
    p.box(MARGIN, y, CONTENT_W, 16 + 14 * len(lines))
    for i, line in enumerate(lines):
        p.text(MARGIN + 8, y + 18 + 14 * i, line, size=8.5)
    p.save()


RENDERERS = {
    "nfe": render_nfe,
    "cte": render_cte,
    "nfse": render_nfse,
    "boleto": render_boleto,
    "other": render_other,
}


def render_document(doc: dict[str, Any], path: Path) -> None:
    """Render ``doc`` with the renderer of its type."""
    RENDERERS[doc["doc_type"]](doc, path)
