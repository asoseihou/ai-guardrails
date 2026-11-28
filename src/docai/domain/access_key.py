"""NF-e / CT-e 44-digit access key ("chave de acesso").

Layout (positions are 1-based)::

    cUF   AAMM  CNPJ            mod  serie  nNF        tpEmis  cNF       DV
    1-2   3-6   7-20            21-22 23-25 26-34      35      36-43     44

* ``cUF``   IBGE code of the issuer's state (35 = SP, 33 = RJ, ...)
* ``AAMM``  year and month of issue
* ``CNPJ``  issuer CNPJ
* ``mod``   document model: 55 = NF-e, 57 = CT-e
* ``serie`` series, ``nNF`` document number, ``tpEmis`` emission type (1 = normal)
* ``cNF``   random numeric code, ``DV`` modulo-11 check digit

The check digit uses weights 2..9 cycling from right to left over the first
43 digits; remainders 0 and 1 map to digit 0.

>>> key = build_access_key(cuf="35", year=2025, month=6, cnpj="11222333000181",
...                        model="55", series=1, number=12345, code=12345678)
>>> key
'35250611222333000181550010000123451123456781'
>>> is_valid_access_key(key)
True
>>> parse_access_key(key).number
12345
>>> format_access_key(key)[:14]
'3525 0611 2223'
"""

from __future__ import annotations

import re
from dataclasses import dataclass

KEY_LENGTH = 44
MODEL_NFE = "55"
MODEL_CTE = "57"
MODEL_BY_DOC_TYPE = {"nfe": MODEL_NFE, "cte": MODEL_CTE}
DOC_TYPE_BY_MODEL = {v: k for k, v in MODEL_BY_DOC_TYPE.items()}

# IBGE state codes (public).
UF_CODES: dict[str, str] = {
    "RO": "11",
    "AC": "12",
    "AM": "13",
    "RR": "14",
    "PA": "15",
    "AP": "16",
    "TO": "17",
    "MA": "21",
    "PI": "22",
    "CE": "23",
    "RN": "24",
    "PB": "25",
    "PE": "26",
    "AL": "27",
    "SE": "28",
    "BA": "29",
    "MG": "31",
    "ES": "32",
    "RJ": "33",
    "SP": "35",
    "PR": "41",
    "SC": "42",
    "RS": "43",
    "MS": "50",
    "MT": "51",
    "GO": "52",
    "DF": "53",
}

# Eleven groups of four digits, optionally separated by single spaces, not
# glued to other digits. This is how keys are printed on DANFE/DACTE pages.
KEY_PATTERN = re.compile(r"(?<!\d)(\d{4}(?: ?\d{4}){10})(?!\d)")


def key_check_digit(key43: str) -> str:
    """Modulo-11 check digit of the first 43 digits of an access key."""
    if not re.fullmatch(r"\d{43}", key43):
        raise ValueError("expected 43 digits")
    total, weight = 0, 2
    for char in reversed(key43):
        total += int(char) * weight
        weight = 2 if weight == 9 else weight + 1
    remainder = total % 11
    return "0" if remainder < 2 else str(11 - remainder)


def normalize_key(value: object) -> str:
    """Digits only."""
    return re.sub(r"\D", "", str(value))


def is_valid_access_key(value: object) -> bool:
    """44 digits with a correct modulo-11 check digit."""
    key = normalize_key(value)
    return len(key) == KEY_LENGTH and key[-1] == key_check_digit(key[:43])


@dataclass(frozen=True)
class AccessKey:
    """The fields encoded in an access key."""

    cuf: str
    aamm: str
    cnpj: str
    model: str
    series: int
    number: int
    emission_type: str
    code: str
    dv: str

    @property
    def value(self) -> str:
        """The 44-digit key."""
        return (
            f"{self.cuf}{self.aamm}{self.cnpj}{self.model}{self.series:03d}"
            f"{self.number:09d}{self.emission_type}{self.code}{self.dv}"
        )

    @property
    def year(self) -> int:
        """Four-digit year of issue (from AAMM)."""
        return 2000 + int(self.aamm[:2])

    @property
    def month(self) -> int:
        """Month of issue (from AAMM)."""
        return int(self.aamm[2:])

    @property
    def doc_type(self) -> str | None:
        """``nfe`` or ``cte`` from the model code, if known."""
        return DOC_TYPE_BY_MODEL.get(self.model)


def build_access_key(
    *,
    cuf: str,
    year: int,
    month: int,
    cnpj: str,
    model: str,
    series: int,
    number: int,
    code: int,
    emission_type: int = 1,
) -> str:
    """Assemble the 43 digits and append the check digit."""
    if not (1 <= month <= 12):
        raise ValueError("month must be 1..12")
    body = (
        f"{int(cuf):02d}{year % 100:02d}{month:02d}{cnpj}{model}{series:03d}"
        f"{number:09d}{emission_type}{code:08d}"
    )
    if len(body) != 43 or not body.isdigit():
        raise ValueError(f"invalid key components (got {len(body)} positions)")
    return body + key_check_digit(body)


def parse_access_key(value: object) -> AccessKey:
    """Split a 44-digit key into its fields (the check digit is not verified)."""
    key = normalize_key(value)
    if len(key) != KEY_LENGTH:
        raise ValueError(f"access key must have 44 digits, got {len(key)}")
    return AccessKey(
        cuf=key[0:2],
        aamm=key[2:6],
        cnpj=key[6:20],
        model=key[20:22],
        series=int(key[22:25]),
        number=int(key[25:34]),
        emission_type=key[34],
        code=key[35:43],
        dv=key[43],
    )


def format_access_key(value: object) -> str:
    """Groups of four digits, as printed on DANFE/DACTE pages."""
    key = normalize_key(value)
    return " ".join(key[i : i + 4] for i in range(0, len(key), 4))


def find_keys_in_text(text: str) -> list[str]:
    """All 44-digit sequences printed in groups of four (valid or not)."""
    return [normalize_key(m.group(1)) for m in KEY_PATTERN.finditer(text)]
