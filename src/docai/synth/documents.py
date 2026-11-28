"""Ground-truth generation for synthetic fiscal documents.

Each document is a dict with:

* ``doc_id``    stable id, also the prefix of the PDF filename
* ``doc_type``  one of ``nfe``, ``cte``, ``nfse``, ``boleto``, ``other``
* ``filename``  PDF filename (contains an unrelated batch number on purpose:
  LLMs sometimes copy numbers from the filename into the document number)
* ``fields``    the gold extraction (empty for boleto/other)
* ``layout``    extra content needed to render a realistic page

Money is generated in integer cents and stored as floats with two decimals.
"""

from __future__ import annotations

import itertools
import random
from datetime import date, datetime, timedelta
from typing import Any

from docai.domain.access_key import MODEL_CTE, MODEL_NFE, UF_CODES, build_access_key
from docai.synth.names import (
    BANKS,
    CARGO_DESCRIPTIONS,
    FREIGHT_COMPONENTS,
    PRODUCTS,
    SERVICES,
    Company,
)

DEFAULT_MIX: dict[str, float] = {
    "nfe": 0.30,
    "cte": 0.25,
    "nfse": 0.20,
    "boleto": 0.13,
    "other": 0.12,
}

_START = date(2025, 1, 1)
_END = date(2026, 10, 31)


def _money(cents: int) -> float:
    return round(cents / 100, 2)


def _random_date(rng: random.Random) -> date:
    return _START + timedelta(days=rng.randint(0, (_END - _START).days))


def _random_datetime(rng: random.Random, day: date) -> datetime:
    return datetime(day.year, day.month, day.day, rng.randint(7, 19), rng.randint(0, 59))


def _two_distinct(rng: random.Random, pool: list[Company], k: int = 2) -> list[Company]:
    return rng.sample(pool, k)


def _filename(doc_id: str, rng: random.Random) -> str:
    tag = rng.choice(
        [
            f"lote-{rng.randint(10_000, 99_999)}",
            f"anexo-{rng.randint(100_000, 999_999)}",
            f"upload{rng.randint(1_000_000, 9_999_999)}",
            f"scan-{rng.randint(1_000, 9_999)}",
        ]
    )
    return f"{doc_id}__{tag}.pdf"


ATTACHMENT_SHARE = 0.6
SERVICE_ORDER_SHARE = 0.7
SERVICE_EXTRAS = ["MATERIAL DE CONSUMO", "DESLOCAMENTO DE EQUIPE", "HORAS ADICIONAIS"]


def _weight(rng: random.Random) -> float:
    """Gross weight in kg with three decimals (printed as 17742,292)."""
    return round(rng.randint(10_000, 30_000_000) / 1000, 3)


def _service_order(rng: random.Random, issued: date) -> str | None:
    """A client service-order reference (``OS 01234/2026``), present on most documents."""
    number = rng.randint(1, 99_999)
    return f"{number:05d}/{issued.year}" if rng.random() < SERVICE_ORDER_SHARE else None


def _service_lines(rng: random.Random, service_name: str, total_cents: int) -> list[dict[str, Any]]:
    """1-3 service lines (description, quantity, value) that add up to the total."""
    n = rng.randint(1, 3)
    names = [service_name.upper(), *rng.sample(SERVICE_EXTRAS, n - 1)]
    cuts = sorted(rng.sample(range(1, total_cents), n - 1)) if n > 1 else []
    bounds = [0, *cuts, total_cents]
    return [
        {"description": name, "quantity": rng.randint(1, 10), "value": _money(hi - lo)}
        for name, (lo, hi) in zip(names, itertools.pairwise(bounds), strict=True)
    ]


def _key_code(rng: random.Random, number: int) -> int:
    code = rng.randint(10_000_000, 99_999_999)
    return code if code != number else code + 1


def make_nfe(doc_id: str, rng: random.Random, pool: list[Company]) -> dict[str, Any]:
    """Ground truth and layout of a synthetic NF-e."""
    issuer, recipient = _two_distinct(rng, pool)
    issued = _random_date(rng)
    series = rng.choice([1, 1, 1, 2, 3, 10])
    number = rng.randint(1, 999_999)
    same_state = issuer.uf == recipient.uf
    cfop = rng.choice(["5101", "5102"] if same_state else ["6101", "6102"])
    items = []
    total_cents = 0
    for product, ncm in rng.sample(PRODUCTS, rng.randint(1, 5)):
        qty = rng.randint(1, 20)
        unit_cents = rng.randint(500, 250_000)
        total_cents += qty * unit_cents
        items.append(
            {
                "code": f"{rng.randint(1000, 99999)}",
                "description": product.upper(),
                "ncm": ncm,
                "cfop": cfop,
                "unit": "UN",
                "qty": qty,
                "unit_price": _money(unit_cents),
                "total": _money(qty * unit_cents),
            }
        )
    icms_rate = 18 if same_state else 12
    key = build_access_key(
        cuf=UF_CODES[issuer.uf],
        year=issued.year,
        month=issued.month,
        cnpj=issuer.cnpj,
        model=MODEL_NFE,
        series=series,
        number=number,
        code=_key_code(rng, number),
    )
    return {
        "doc_type": "nfe",
        "fields": {
            "number": str(number),
            "series": str(series),
            "issue_date": issued.isoformat(),
            "issuer_name": issuer.name,
            "issuer_cnpj": issuer.cnpj,
            "recipient_name": recipient.name,
            "recipient_cnpj": recipient.cnpj,
            "total_value": _money(total_cents),
            "access_key": key,
            "cfop": cfop,
        },
        "layout": {
            "issuer": _company_dict(issuer),
            "recipient": _company_dict(recipient),
            "issued_at": _random_datetime(rng, issued).isoformat(),
            "nature": "VENDA DE MERCADORIA",
            "protocol": f"{UF_CODES[issuer.uf]}{rng.randint(10**12, 10**13 - 1)}",
            "items": items,
            "icms_rate": icms_rate,
            "icms_value": _money(round(total_cents * icms_rate / 100)),
        },
    }


def _referenced_key(rng: random.Random, sender: Company, issued: date) -> str:
    """Access key of an NF-e (the cargo's invoice) issued by the sender shortly before."""
    ref_day = issued - timedelta(days=rng.randint(0, 20))
    ref_number = rng.randint(1, 999_999)
    return build_access_key(
        cuf=UF_CODES[sender.uf],
        year=ref_day.year,
        month=ref_day.month,
        cnpj=sender.cnpj,
        model=MODEL_NFE,
        series=1,
        number=ref_number,
        code=_key_code(rng, ref_number),
    )


def _attachment(rng: random.Random, key: str, issued: date) -> dict[str, Any] | None:
    """The DANFE of the cargo attached to the CT-e PDF (its own weight and due dates)."""
    if rng.random() >= ATTACHMENT_SHARE:
        return None
    return {
        "key": key,
        "number": int(key[25:34]),
        "gross_weight": _weight(rng),
        "due_date": (issued + timedelta(days=rng.randint(46, 90))).isoformat(),
        "value": _money(rng.randint(1_000_000, 200_000_000)),
    }


def make_cte(doc_id: str, rng: random.Random, pool: list[Company]) -> dict[str, Any]:
    """Ground truth and layout of a synthetic CT-e (sometimes with an attached DANFE)."""
    carrier, sender, recipient = _two_distinct(rng, pool, 3)
    issued = _random_date(rng)
    series = rng.choice([1, 1, 2])
    number = rng.randint(1, 999_999)
    cfop = "5353" if sender.uf == recipient.uf else "6353"
    components = [("FRETE PESO", rng.randint(30_000, 2_000_000))]
    for name in rng.sample(FREIGHT_COMPONENTS, rng.randint(1, 4)):
        components.append((name, rng.randint(500, 60_000)))
    total_cents = sum(c for _, c in components)
    referenced = [_referenced_key(rng, sender, issued) for _ in range(rng.randint(1, 2))]
    key = build_access_key(
        cuf=UF_CODES[carrier.uf],
        year=issued.year,
        month=issued.month,
        cnpj=carrier.cnpj,
        model=MODEL_CTE,
        series=series,
        number=number,
        code=_key_code(rng, number),
    )
    weight = _weight(rng)
    due = issued + timedelta(days=rng.randint(15, 45))
    service_order = _service_order(rng, issued)
    attachment = _attachment(rng, referenced[0], issued)
    return {
        "doc_type": "cte",
        "fields": {
            "number": str(number),
            "series": str(series),
            "issue_date": issued.isoformat(),
            "issuer_name": carrier.name,
            "issuer_cnpj": carrier.cnpj,
            "sender_name": sender.name,
            "sender_cnpj": sender.cnpj,
            "recipient_name": recipient.name,
            "recipient_cnpj": recipient.cnpj,
            "total_value": _money(total_cents),
            "access_key": key,
            "cfop": cfop,
            "freight_components": [{"name": n, "value": _money(c)} for n, c in components],
            "gross_weight": weight,
            "due_date": due.isoformat(),
            "referenced_keys": referenced,
            "service_order": service_order,
        },
        "layout": {
            "issuer": _company_dict(carrier),
            "sender": _company_dict(sender),
            "recipient": _company_dict(recipient),
            "taker": rng.choice(["sender", "recipient"]),
            "issued_at": _random_datetime(rng, issued).isoformat(),
            "cargo": rng.choice(CARGO_DESCRIPTIONS),
            "cargo_value": _money(rng.randint(1_000_000, 200_000_000)),
            "referenced_keys": referenced,
            "protocol": f"{UF_CODES[carrier.uf]}{rng.randint(10**12, 10**13 - 1)}",
            "attachment": attachment,
        },
    }


def make_nfse(doc_id: str, rng: random.Random, pool: list[Company]) -> dict[str, Any]:
    """Ground truth and layout of a synthetic NFS-e."""
    issuer, recipient = _two_distinct(rng, pool)
    issued = _random_date(rng)
    number = rng.randint(1, 99_999)
    service_code, service_name = rng.choice(SERVICES)
    total_cents = rng.randint(20_000, 5_000_000)
    rate = rng.choice([2.0, 3.0, 5.0])
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    code = "".join(rng.choice(alphabet) for _ in range(8))
    services = _service_lines(rng, service_name, total_cents)
    service_order = _service_order(rng, issued)
    return {
        "doc_type": "nfse",
        "fields": {
            "number": str(number),
            "issue_date": issued.isoformat(),
            "issuer_name": issuer.name,
            "issuer_cnpj": issuer.cnpj,
            "recipient_name": recipient.name,
            "recipient_cnpj": recipient.cnpj,
            "total_value": _money(total_cents),
            "iss_rate": rate,
            "iss_value": _money(round(total_cents * rate / 100)),
            "verification_code": f"{code[:4]}-{code[4:]}",
            "services": services,
            "service_order": service_order,
        },
        "layout": {
            "issuer": _company_dict(issuer),
            "recipient": _company_dict(recipient),
            "issued_at": _random_datetime(rng, issued).isoformat(),
            "service_code": service_code,
            "service_name": service_name,
            "description": f"{service_name.upper()} CONFORME CONTRATO {rng.randint(100, 999)}/2025",
        },
    }


def make_boleto(doc_id: str, rng: random.Random, pool: list[Company]) -> dict[str, Any]:
    """Layout of a synthetic boleto (not extracted)."""
    beneficiary, payer = _two_distinct(rng, pool)
    bank_name, bank_code = rng.choice(BANKS)
    due = _random_date(rng)
    amount_cents = rng.randint(10_000, 3_000_000)
    digits = "".join(str(rng.randint(0, 9)) for _ in range(47))
    line = (
        f"{bank_code}{digits[3:5]}.{digits[5:10]} {digits[10:15]}.{digits[15:21]} "
        f"{digits[21:26]}.{digits[26:32]} {digits[32]} {digits[33:47]}"
    )
    return {
        "doc_type": "boleto",
        "fields": {},
        "layout": {
            "bank_name": bank_name,
            "bank_code": bank_code,
            "beneficiary": _company_dict(beneficiary),
            "payer": _company_dict(payer),
            "due_date": due.isoformat(),
            "amount": _money(amount_cents),
            "our_number": f"{rng.randint(10**9, 10**10 - 1)}",
            "line": line,
            "reference": rng.choice(
                [f"REFERENTE À NF {rng.randint(1, 999_999)}", "REFERENTE A SERVIÇOS PRESTADOS", ""]
            ),
        },
    }


def make_other(doc_id: str, rng: random.Random, pool: list[Company]) -> dict[str, Any]:
    """Layout of a non-fiscal document (not extracted)."""
    a, b = _two_distinct(rng, pool)
    day = _random_date(rng)
    kind = rng.choice(["cce", "delivery", "service_order", "inspection", "packing"])
    layout: dict[str, Any] = {
        "kind": kind,
        "company": _company_dict(a),
        "counterpart": _company_dict(b),
        "date": day.isoformat(),
        "ref_number": rng.randint(1, 999_999),
    }
    if kind == "cce":
        layout["ref_key"] = build_access_key(
            cuf=UF_CODES[a.uf],
            year=day.year,
            month=day.month,
            cnpj=a.cnpj,
            model=MODEL_NFE,
            series=1,
            number=layout["ref_number"],
            code=_key_code(rng, layout["ref_number"]),
        )
    if kind == "packing":
        layout["items"] = [
            {"description": p.upper(), "qty": rng.randint(1, 50)}
            for p, _ in rng.sample(PRODUCTS, rng.randint(2, 5))
        ]
    return {"doc_type": "other", "fields": {}, "layout": layout}


_MAKERS = {
    "nfe": make_nfe,
    "cte": make_cte,
    "nfse": make_nfse,
    "boleto": make_boleto,
    "other": make_other,
}


def make_document(
    doc_type: str, doc_id: str, rng: random.Random, pool: list[Company]
) -> dict[str, Any]:
    """One synthetic document of ``doc_type`` with its file name."""
    doc = _MAKERS[doc_type](doc_id, rng, pool)
    return {"doc_id": doc_id, "filename": _filename(doc_id, rng), **doc}


def sample_types(n: int, rng: random.Random, mix: dict[str, float] | None = None) -> list[str]:
    """Deterministic type assignment that follows ``mix`` as closely as possible."""
    mix = mix or DEFAULT_MIX
    total = sum(mix.values())
    counts = {t: int(n * w / total) for t, w in mix.items()}
    remaining = n - sum(counts.values())
    for t in sorted(mix, key=lambda t: -mix[t])[:remaining]:
        counts[t] += 1
    types = [t for t, c in counts.items() for _ in range(c)]
    rng.shuffle(types)
    return types


def _company_dict(company: Company) -> dict[str, str]:
    return {
        "name": company.name,
        "cnpj": company.cnpj,
        "city": company.city,
        "uf": company.uf,
        "street": company.street,
    }
