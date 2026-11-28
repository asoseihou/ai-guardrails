"""CNPJ (Brazilian company registry number) check digits.

A CNPJ has 14 positions: an 8-position root, a 4-position branch number
(``0001`` is usually the headquarters) and 2 check digits computed with a
weighted modulo-11 rule. Since July 2026 the first 12 positions may also be
uppercase letters (alphanumeric CNPJ); each character is valued as its ASCII
code minus 48, so digits keep their usual value and the rule is unchanged for
numeric CNPJs.

>>> cnpj_check_digits("112223330001")
'81'
>>> is_valid_cnpj("11.222.333/0001-81")
True
>>> is_valid_cnpj("11.222.333/0001-80")
False
>>> is_valid_cnpj("12.ABC.345/01DE-35")  # alphanumeric example
True
>>> format_cnpj("11222333000181")
'11.222.333/0001-81'
"""

from __future__ import annotations

import re

CNPJ_LENGTH = 14
_WEIGHTS_1 = (5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2)
_WEIGHTS_2 = (6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2)
_BASE_RE = re.compile(r"[0-9A-Z]{12}")
_FULL_RE = re.compile(r"[0-9A-Z]{12}[0-9]{2}")


def _char_value(char: str) -> int:
    return ord(char) - 48


def _mod11_digit(values: list[int], weights: tuple[int, ...]) -> int:
    remainder = sum(v * w for v, w in zip(values, weights, strict=True)) % 11
    return 0 if remainder < 2 else 11 - remainder


def normalize_cnpj(value: object) -> str:
    """Strip punctuation and whitespace; keep digits and uppercase letters."""
    return re.sub(r"[^0-9A-Z]", "", str(value).upper())


def cnpj_check_digits(base12: str) -> str:
    """Return the two check digits for a 12-position CNPJ base."""
    base = normalize_cnpj(base12)
    if not _BASE_RE.fullmatch(base):
        raise ValueError(f"CNPJ base must have 12 alphanumeric positions, got {base12!r}")
    values = [_char_value(c) for c in base]
    first = _mod11_digit(values, _WEIGHTS_1)
    second = _mod11_digit([*values, first], _WEIGHTS_2)
    return f"{first}{second}"


def is_valid_cnpj(value: object) -> bool:
    """True when ``value`` is a well-formed CNPJ with correct check digits."""
    cnpj = normalize_cnpj(value)
    if not _FULL_RE.fullmatch(cnpj):
        return False
    if len(set(cnpj)) == 1:  # 00000000000000, 11111111111111, ... are placeholders
        return False
    return cnpj[12:] == cnpj_check_digits(cnpj[:12])


def format_cnpj(value: object) -> str:
    """Print a CNPJ as ``12.345.678/0001-95``."""
    cnpj = normalize_cnpj(value)
    if len(cnpj) != CNPJ_LENGTH:
        raise ValueError(f"CNPJ must have 14 positions, got {value!r}")
    return f"{cnpj[:2]}.{cnpj[2:5]}.{cnpj[5:8]}/{cnpj[8:12]}-{cnpj[12:]}"
