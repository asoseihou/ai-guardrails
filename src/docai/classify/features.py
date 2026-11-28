"""Hashed character n-gram features (the "hashing trick").

Text is lowercased, accents removed and every digit mapped to ``0`` (numbers
are noise for document-type classification). Each character n-gram is hashed
with CRC32 into a fixed number of buckets, so there is no vocabulary to store
and unseen words still map somewhere sensible.
"""

from __future__ import annotations

import random
import re
import zlib

from docai.domain.text import strip_accents

N_BUCKETS = 2**15
NGRAM_RANGE = (3, 4)
MAX_CHARS = 3000
_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789 .,/-"


def normalize_for_features(text: str) -> str:
    """Lowercase, accent-free, digits as ``0``, whitespace collapsed."""
    text = strip_accents(text).lower()
    text = re.sub(r"\d", "0", text)
    return re.sub(r"\s+", " ", text).strip()


def ngram_ids(
    text: str,
    n_buckets: int = N_BUCKETS,
    ngram_range: tuple[int, int] | list[int] = NGRAM_RANGE,
    max_chars: int = MAX_CHARS,
) -> list[int]:
    """Hashed character n-gram ids of ``text`` (at least one id)."""
    padded = f" {normalize_for_features(text)[:max_chars]} "
    low, high = ngram_range
    ids = [
        zlib.crc32(padded[i : i + n].encode()) % n_buckets
        for n in range(low, high + 1)
        for i in range(len(padded) - n + 1)
    ]
    return ids or [0]


def augment(
    text: str, rng: random.Random, crop: tuple[float, float] = (0.3, 0.8), noise: float = 0.04
) -> str:
    """Simulate partial pages and OCR noise: a random window of lines + char edits."""
    lines = [line for line in text.splitlines() if line.strip()] or [text]
    keep = max(1, round(len(lines) * rng.uniform(*crop)))
    start = rng.randint(0, len(lines) - keep)
    window = "\n".join(lines[start : start + keep])
    out: list[str] = []
    for char in window:
        r = rng.random()
        if r < noise / 3:
            continue  # deletion
        if r < 2 * noise / 3:
            out.append(rng.choice(_ALPHABET))  # substitution
        elif r < noise:
            out.extend((char, rng.choice(_ALPHABET)))  # insertion
        else:
            out.append(char)
    return "".join(out)
