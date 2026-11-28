"""Kind-specific normalization (representation only — never changes meaning).

``normalize_value`` raises ``ValueError`` when the value cannot be a valid
instance of its kind; the schema guardrail turns that into a *nulled* finding.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from datetime import date, datetime
from typing import Any

from docai.domain.cnpj import normalize_cnpj
from docai.domain.text import digits_only

_CNPJ_SHAPE = re.compile(r"[0-9A-Z]{12}[0-9]{2}")


def parse_money(value: Any) -> float:
    """Parse BRL/US-formatted amounts (``"1.234,56"``, ``"1,234.56"``, ``1234.56``)."""
    if isinstance(value, bool):
        raise ValueError("boolean is not a number")
    if isinstance(value, int | float):
        number = float(value)
    else:
        text = str(value).strip().replace("R$", "").replace(" ", "")
        if "," in text and "." in text:
            if text.rfind(",") > text.rfind("."):
                text = text.replace(".", "").replace(",", ".")
            else:
                text = text.replace(",", "")
        elif "," in text:
            text = text.replace(".", "").replace(",", ".")
        elif text.count(".") > 1:
            text = text.replace(".", "")
        number = float(text)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"not a non-negative amount: {value!r}")
    return round(number, 2)


def parse_date(value: Any) -> str:
    """Parse common date formats into ISO ``YYYY-MM-DD``."""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%d/%m/%Y %H:%M"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"unrecognised date {value!r}")


def _doc_number(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    digits = digits_only(value)
    if not digits or len(digits) > 15 or int(digits) == 0:
        raise ValueError(f"not a document number: {value!r}")
    return str(int(digits))


def _components(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("components must be a non-empty list")
    out = []
    for item in value:
        if not isinstance(item, dict) or "name" not in item or "value" not in item:
            raise ValueError(f"bad component {item!r}")
        name = " ".join(str(item["name"]).upper().split())
        if not name:
            raise ValueError("component without a name")
        out.append({"name": name, "value": parse_money(item["value"])})
    return out


def _cnpj(value: Any) -> str:
    cnpj = normalize_cnpj(value)
    if not _CNPJ_SHAPE.fullmatch(cnpj):
        raise ValueError(f"a CNPJ has 14 positions, got {value!r}")
    return cnpj


def _access_key(value: Any) -> str:
    key = digits_only(value)
    if len(key) != 44:
        raise ValueError(f"an access key has 44 digits, got {len(key)}")
    return key


def _rate(value: Any) -> float:
    rate = parse_money(value)
    if rate > 100:
        raise ValueError(f"rate above 100%: {value!r}")
    return rate


def _name(value: Any) -> str:
    name = " ".join(str(value).split())
    if not re.search(r"[A-Za-zÀ-ÿ]", name):
        raise ValueError(f"not a name: {value!r}")
    return name


def _cfop(value: Any) -> str:
    cfop = digits_only(value)
    if len(cfop) != 4 or cfop[0] not in "1234567":
        raise ValueError(f"a CFOP has 4 digits starting with 1-7, got {value!r}")
    return cfop


def _code(value: Any) -> str:
    code = str(value).strip().upper()
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9\-./]{2,40}", code):
        raise ValueError(f"not a verification code: {value!r}")
    return code


def _weight(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean is not a weight")
    if isinstance(value, int | float):
        number = float(value)
    else:
        text = str(value).strip().upper().removesuffix("KG").strip()
        if "," in text:
            text = text.replace(".", "").replace(",", ".")
        number = float(text)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"not a positive weight: {value!r}")
    return round(number, 3)


def _reference(value: Any) -> str:
    ref = re.sub(r"\s+", "", str(value)).upper()
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9./-]{1,19}", ref):
        raise ValueError(f"not a reference number: {value!r}")
    return ref


def _key_list(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError("referenced keys must be a non-empty list")
    return [_access_key(item) for item in value]


def _services(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("services must be a non-empty list")
    out = []
    for item in value:
        if not isinstance(item, dict) or not {"description", "value"} <= set(item):
            raise ValueError(f"bad service line {item!r}")
        description = " ".join(str(item["description"]).upper().split())
        quantity = int(item.get("quantity") or 1)
        if not description or quantity <= 0:
            raise ValueError(f"bad service line {item!r}")
        out.append(
            {"description": description, "quantity": quantity, "value": parse_money(item["value"])}
        )
    return out


_NORMALIZERS: dict[str, Callable[[Any], Any]] = {
    "doc_number": _doc_number,
    "cnpj": _cnpj,
    "access_key": _access_key,
    "date": parse_date,
    "money": parse_money,
    "rate": _rate,
    "name": _name,
    "cfop": _cfop,
    "code": _code,
    "components": _components,
    "weight": _weight,
    "reference": _reference,
    "key_list": _key_list,
    "services": _services,
}


def normalize_value(kind: str, value: Any) -> Any:
    """Canonical representation of ``value`` for a field ``kind``; empty → ``None``.

    Raises ``ValueError`` when the value cannot be a valid instance of the kind.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        normalizer = _NORMALIZERS[kind]
    except KeyError:
        raise ValueError(f"unknown field kind {kind!r}") from None
    return normalizer(value)


def lenient_normalize(kind: str, value: Any) -> Any:
    """Best-effort normalization for *comparisons* (falls back to the raw value)."""
    try:
        return normalize_value(kind, value)
    except (ValueError, TypeError):
        return value
