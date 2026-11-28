"""Semantic equality of field values (representation differences do not count)."""

from __future__ import annotations

from typing import Any

from docai.domain.normalize import lenient_normalize
from docai.domain.schemas import FIELD_KINDS
from docai.domain.text import normalize_words


def same_value(name: str, predicted: Any, truth: Any) -> bool:
    """True when two values of field ``name`` mean the same thing.

    >>> same_value("issuer_cnpj", "11.222.333/0001-81", "11222333000181")
    True
    >>> same_value("total_value", "1.234,56", 1234.56)
    True
    >>> same_value("issuer_name", "AURORA LOGÍSTICA LTDA", "Aurora Logistica Ltda")
    True
    >>> same_value("number", None, "1")
    False
    """
    kind = FIELD_KINDS[name]
    p, t = lenient_normalize(kind, predicted), lenient_normalize(kind, truth)
    if p is None or t is None:
        return p is None and t is None
    if kind in ("money", "rate", "weight"):
        try:
            return abs(float(p) - float(t)) < (0.0005 if kind == "weight" else 0.005)
        except (TypeError, ValueError):
            return False
    if kind == "name":
        return normalize_words(str(p)) == normalize_words(str(t))
    if kind in ("components", "services", "key_list"):
        try:
            return _list_key(kind, p) == _list_key(kind, t)
        except (TypeError, KeyError):
            return False
    return bool(p == t)


def _list_key(kind: str, items: Any) -> list[Any]:
    if kind == "key_list":
        return sorted(items)
    if kind == "services":
        return sorted((s["description"], s["quantity"], round(s["value"], 2)) for s in items)
    return sorted((c["name"], round(c["value"], 2)) for c in items)
