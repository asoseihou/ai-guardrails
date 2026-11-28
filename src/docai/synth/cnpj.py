"""Synthetic CNPJ generation and realistic misreads (test data only).

The validation rules live in :mod:`docai.domain.cnpj`; this module only makes
up numbers for the synthetic corpus and corrupts them for the simulated LLM.
"""

from __future__ import annotations

import random

from docai.domain.cnpj import cnpj_check_digits, normalize_cnpj

# Synthetic CNPJs use branch numbers in this range. Real establishments with a
# branch number >= 9000 are extremely rare, which makes an accidental collision
# with a registered company practically impossible.
SYNTHETIC_BRANCH_RANGE = (9001, 9899)


def generate_cnpj(rng: random.Random, branch: int | None = None) -> str:
    """Generate a numeric CNPJ with valid check digits (synthetic branch range)."""
    root = f"{rng.randint(1, 9)}{rng.randint(0, 9_999_999):07d}"
    if branch is None:
        branch = rng.randint(*SYNTHETIC_BRANCH_RANGE)
    base = f"{root}{branch:04d}"
    return base + cnpj_check_digits(base)


def corrupt_cnpj(cnpj: str, rng: random.Random, mode: str | None = None) -> str:
    """Return a different 14-digit string that looks like a misread of ``cnpj``.

    ``mode`` is ``"swap"`` (two adjacent different digits transposed) or
    ``"substitute"`` (one digit replaced). The result usually fails the check
    digits, but not always — exactly like a real misread, which is why the
    guardrails also check that the value is literally present in the document.
    """
    digits = list(normalize_cnpj(cnpj))
    mode = mode or rng.choice(["swap", "substitute"])
    if mode == "swap":
        positions = [i for i in range(len(digits) - 1) if digits[i] != digits[i + 1]]
        if positions:
            i = rng.choice(positions)
            digits[i], digits[i + 1] = digits[i + 1], digits[i]
            return "".join(digits)
    i = rng.randrange(len(digits))
    digits[i] = rng.choice([d for d in "0123456789" if d != digits[i]])
    return "".join(digits)
