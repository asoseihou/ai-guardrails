"""Load a trained classifier, with a keyword fallback."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from docai.classify.base import DocumentClassifier, Prediction
from docai.classify.rules import KeywordClassifier

log = logging.getLogger(__name__)


class TorchDocTypeClassifier:
    """The trained n-gram MLP behind the :class:`DocumentClassifier` protocol."""

    name = "torch-ngram-mlp"

    def __init__(self, model: Any, report: dict[str, Any]) -> None:
        self.model = model.eval()
        self.report = report
        self.labels: list[str] = list(report["labels"])
        cfg = report["config"]
        self._n_buckets = cfg["n_buckets"]
        self._ngram_range = (cfg["ngram_low"], cfg["ngram_high"])

    @classmethod
    def load(cls, path: Path) -> TorchDocTypeClassifier:
        """Load weights and report saved by :func:`docai.classify.train.save_classifier`."""
        import torch

        from docai.classify.model import DocTypeMLP

        bundle = torch.load(path, map_location="cpu", weights_only=True)
        report = bundle["report"]
        cfg = report["config"]
        model = DocTypeMLP(
            cfg["n_buckets"], len(report["labels"]), cfg["emb_dim"], cfg["hidden"], cfg["dropout"]
        )
        model.load_state_dict(bundle["state_dict"])
        return cls(model, report)

    def predict(self, text: str) -> Prediction:
        """Classify one text layer."""
        import torch

        from docai.classify.features import ngram_ids
        from docai.classify.model import pack

        ids, offsets = pack([ngram_ids(text, self._n_buckets, self._ngram_range)])
        with torch.no_grad():
            probs = torch.softmax(self.model(ids, offsets), dim=1)[0].tolist()
        by_label = {label: round(p, 4) for label, p in zip(self.labels, probs, strict=True)}
        best = max(by_label, key=lambda k: by_label[k])
        return Prediction(best, by_label[best], by_label)


def load_classifier(path: Path | None) -> DocumentClassifier:
    """Trained PyTorch model when available, keyword rules otherwise."""
    if path is not None and Path(path).exists():
        try:
            return TorchDocTypeClassifier.load(Path(path))
        except ImportError:
            log.warning("torch is not installed; using the keyword classifier")
        except Exception as exc:  # corrupted/incompatible weights must not take the API down
            log.warning("could not load classifier weights (%s); using keyword rules", exc)
    return KeywordClassifier()
