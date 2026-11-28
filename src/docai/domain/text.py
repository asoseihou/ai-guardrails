"""Small text helpers shared across modules (no heavy imports)."""

from __future__ import annotations

import re
import unicodedata


def strip_accents(text: str) -> str:
    """Remove combining accents (``ç`` -> ``c``)."""
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def normalize_words(text: str) -> str:
    """Lowercase, accent-free, punctuation collapsed to single spaces."""
    text = strip_accents(text).lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def digits_only(value: object) -> str:
    """Keep only the digits of ``value``."""
    return re.sub(r"\D", "", str(value))


def parse_brl(text: str) -> float:
    """'1.234,56' -> 1234.56"""
    return round(float(text.replace(".", "").replace(",", ".")), 2)


def money_digits(value: float) -> str:
    """Digits of a value printed with two decimals: 1234.5 -> '123450'."""
    return f"{value:.2f}".replace(".", "").lstrip("-")
