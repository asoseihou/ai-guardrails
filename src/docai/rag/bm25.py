"""Okapi BM25 in pure Python (no dependencies).

BM25 is a strong, transparent baseline for a small, terminology-heavy corpus:
fiscal vocabulary ("CFOP", "tomador", "modulo 11") is matched exactly, scores are
explainable, and there is no embedding model to download or keep in sync.
"""

from __future__ import annotations

import math
from collections import Counter

from docai.domain.text import normalize_words

_STOPWORDS_TEXT = (
    "a an and are as at be by can do does for from has have how in is it its of on or "
    "that the this to was what when where which who why with will "
    "de da do das dos e em o os um uma para por com no na nos nas que se ao"
)
STOPWORDS = frozenset(_STOPWORDS_TEXT.split())


def tokenize(text: str) -> list[str]:
    """Normalized, stop-word-free tokens with light plural stemming."""
    tokens = []
    for word in normalize_words(text).split():
        if word in STOPWORDS:
            continue
        if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]  # light plural stemming (works for English and Portuguese)
        tokens.append(word)
    return tokens


class BM25:
    """Okapi BM25 over pre-tokenized documents."""

    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = [Counter(doc) for doc in documents]
        self.lengths = [len(doc) for doc in documents]
        self.avg_len = sum(self.lengths) / max(1, len(self.lengths))
        df: Counter[str] = Counter()
        for doc in self.docs:
            df.update(doc.keys())
        n = len(self.docs)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def score(self, query: list[str], index: int) -> float:
        """BM25 score of document ``index`` for ``query``."""
        doc, length = self.docs[index], self.lengths[index]
        total = 0.0
        for term in query:
            tf = doc.get(term, 0)
            if not tf:
                continue
            norm = tf + self.k1 * (1 - self.b + self.b * length / self.avg_len)
            total += self.idf[term] * tf * (self.k1 + 1) / norm
        return total

    def search(self, query: list[str], k: int = 3) -> list[tuple[int, float]]:
        """Top ``k`` ``(index, score)`` pairs with a positive score."""
        scored = [(i, self.score(query, i)) for i in range(len(self.docs))]
        scored = [s for s in scored if s[1] > 0]
        scored.sort(key=lambda s: (-s[1], s[0]))
        return scored[:k]
