"""Keyword fallback classifier (used when torch or trained weights are unavailable)."""

from __future__ import annotations

from docai.classify.base import Prediction
from docai.domain.text import strip_accents

_KEYWORDS: dict[str, tuple[str, ...]] = {
    "nfe": ("DANFE", "DADOS DOS PRODUTOS", "CALCULO DO IMPOSTO"),
    "cte": ("DACTE", "CONHECIMENTO DE TRANSPORTE", "COMPONENTES DO VALOR DA PRESTACAO"),
    "nfse": ("NFS-E", "PRESTADOR DE SERVICOS", "TOMADOR DE SERVICOS"),
    "boleto": ("FICHA DE COMPENSACAO", "NOSSO NUMERO", "LOCAL DE PAGAMENTO"),
}


class KeywordClassifier:
    """Keyword fallback used when torch or trained weights are missing."""

    name = "keyword-rules"
    labels = ("nfe", "cte", "nfse", "boleto", "other")

    def predict(self, text: str) -> Prediction:
        """Classify by counting each type's characteristic headings."""
        upper = strip_accents(text).upper()
        scores = {label: sum(k in upper for k in words) for label, words in _KEYWORDS.items()}
        best = max(scores, key=lambda k: scores[k])
        if scores[best] == 0:
            return Prediction("other", 1.0, {"other": 1.0})
        total = sum(scores.values())
        probs = {label: round(score / total, 4) for label, score in scores.items() if score}
        return Prediction(best, probs[best], probs)
