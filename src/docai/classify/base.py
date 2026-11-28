"""Common classifier types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Prediction:
    """Predicted document type, its confidence and every class probability."""

    label: str
    confidence: float
    probabilities: dict[str, float]


class DocumentClassifier(Protocol):
    """Anything that maps a text layer to a :class:`Prediction`."""

    name: str

    def predict(self, text: str) -> Prediction:
        """Classify one text layer."""
