"""An index over a document's text layer, for literal-presence checks.

Presence is tested on *tokens*, never on the concatenation of every digit in the
page (in which almost any short number would appear by chance).
"""

from __future__ import annotations

import re
from functools import cached_property

from docai.domain.access_key import find_keys_in_text, is_valid_access_key
from docai.domain.cnpj import normalize_cnpj
from docai.domain.text import digits_only, money_digits, normalize_words

_NUM_TOKEN = re.compile(r"\d(?:[\d.,/\-]*\d)?")
_ALNUM_TOKEN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9.\-/]*[A-Za-z0-9])?")
_CNPJ_PRINTED = re.compile(r"\b[0-9A-Z]{2}\.[0-9A-Z]{3}\.[0-9A-Z]{3}/[0-9A-Z]{4}-\d{2}\b")


class DocumentText:
    """Token indexes over a document's text layer for presence tests."""

    def __init__(self, text: str) -> None:
        self.text = text

    @cached_property
    def num_tokens(self) -> set[str]:
        """Digit strings of every numeric token."""
        return {digits_only(t) for t in _NUM_TOKEN.findall(self.text)}

    @cached_property
    def num_tokens_unpadded(self) -> set[str]:
        """Numeric tokens without leading zeros."""
        return {t.lstrip("0") or "0" for t in self.num_tokens}

    @cached_property
    def alnum_tokens(self) -> set[str]:
        """Alphanumeric tokens, uppercase, punctuation removed."""
        return {re.sub(r"[^A-Z0-9]", "", t.upper()) for t in _ALNUM_TOKEN.findall(self.text)}

    @cached_property
    def words(self) -> str:
        """Normalized words, space-delimited on both ends."""
        return f" {normalize_words(self.text)} "

    @cached_property
    def keys(self) -> list[str]:
        """Every 44-digit key printed in groups of four."""
        return find_keys_in_text(self.text)

    @cached_property
    def lines(self) -> list[str]:
        """The text split into lines."""
        return self.text.splitlines()

    # ----------------------------------------------------------- presence tests
    def contains_number(self, value: str) -> bool:
        """Document numbers may be printed zero-padded or with dots (000.012.345)."""
        return (digits_only(value).lstrip("0") or "0") in self.num_tokens_unpadded

    def contains_digits(self, value: str) -> bool:
        """Exact digit token (CFOP and the like)."""
        return digits_only(value) in self.num_tokens

    def contains_cnpj(self, value: str) -> bool:
        """The CNPJ is printed (numeric or alphanumeric)."""
        cnpj = normalize_cnpj(value)
        return cnpj in self.num_tokens or cnpj in self.alnum_tokens

    def contains_money(self, value: float) -> bool:
        """The amount is printed with two decimals."""
        return money_digits(value) in self.num_tokens

    def contains_weight(self, value: float) -> bool:
        """Weights are printed with three decimals (``17742,292``)."""
        return f"{value:.3f}".replace(".", "") in self.num_tokens

    def contains_date(self, iso: str) -> bool:
        """The date is printed as DD/MM/YYYY."""
        year, month, day = iso.split("-")
        return f"{day}{month}{year}" in self.num_tokens

    def contains_phrase(self, phrase: str) -> bool:
        """The words appear consecutively (accent- and case-insensitive)."""
        words = normalize_words(phrase)
        return bool(words) and f" {words} " in self.words

    def contains_code(self, code: str) -> bool:
        """The code is printed (punctuation-insensitive)."""
        return re.sub(r"[^A-Z0-9]", "", code.upper()) in self.alnum_tokens

    def contains_key(self, key: str) -> bool:
        """The 44-digit key is printed."""
        return digits_only(key) in self.keys

    # ----------------------------------------------------------- readers
    def valid_keys(self, model: str | None = None) -> list[str]:
        """Distinct printed keys whose check digit is valid (optionally of one model)."""
        out: list[str] = []
        for key in self.keys:
            wanted = model is None or key[20:22] == model
            if wanted and key not in out and is_valid_access_key(key):
                out.append(key)
        return out

    def cnpj_after_phrase(self, phrase: str, max_lines: int = 12) -> list[str]:
        """For each line containing ``phrase``, the first CNPJ printed at or after it."""
        target = f" {normalize_words(phrase)} "
        found: list[str] = []
        for i, line in enumerate(self.lines):
            if target not in f" {normalize_words(line)} ":
                continue
            start = line.upper().find(phrase.split()[0].upper())
            tail = line[start:] if start >= 0 else line
            window = [tail, *self.lines[i + 1 : i + 1 + max_lines]]
            for candidate in window:
                match = _CNPJ_PRINTED.search(candidate)
                if match:
                    found.append(normalize_cnpj(match.group(0)))
                    break
        return found
