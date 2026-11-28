"""Document-type classifier (PyTorch) with a keyword fallback."""

from docai.classify.base import DocumentClassifier, Prediction
from docai.classify.infer import TorchDocTypeClassifier, load_classifier
from docai.classify.rules import KeywordClassifier

__all__ = [
    "DocumentClassifier",
    "KeywordClassifier",
    "Prediction",
    "TorchDocTypeClassifier",
    "load_classifier",
]
