"""Train / evaluate the document-type classifier.

* Split by document (stratified 60/20/20 train/val/test), so no document ever
  contributes text to two splits.
* Training samples are *augmented* views (random line windows + character
  noise) so the model cannot memorize page headers.
* Early stopping on validation loss; the best epoch's weights are kept.
* The test split is scored twice: on short noisy *fragments* (8-20% of the
  lines, 6% character noise — the hard setting) and on the full clean text
  (what the pipeline sees).
"""

from __future__ import annotations

# Seeded `random` (B311) shuffles and augments training data: reproducibility, not security.
import copy
import functools
import random
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from docai.classify.features import N_BUCKETS, NGRAM_RANGE, augment, ngram_ids
from docai.classify.model import DocTypeMLP, pack
from docai.domain.doc_types import DOC_TYPES


@dataclass
class TrainConfig:
    """Hyper-parameters and augmentation settings of a training run."""

    seed: int = 13
    epochs: int = 40
    patience: int = 4
    batch_size: int = 32
    lr: float = 3e-3
    weight_decay: float = 1e-4
    emb_dim: int = 48
    hidden: int = 64
    dropout: float = 0.2
    augment_per_doc: int = 8
    eval_views: int = 4
    min_delta: float = 1e-3
    # training views: 15-80% of the lines, 4% character noise
    train_crop: tuple[float, float] = (0.15, 0.8)
    train_noise: float = 0.04
    # hard test views: short fragments (8-20% of the lines), 6% character noise
    fragment_crop: tuple[float, float] = (0.08, 0.2)
    fragment_noise: float = 0.06
    n_buckets: int = N_BUCKETS
    ngram_low: int = NGRAM_RANGE[0]
    ngram_high: int = NGRAM_RANGE[1]


def stratified_split(
    labels: dict[str, str], seed: int, fractions: tuple[float, float] = (0.6, 0.2)
) -> dict[str, list[str]]:
    """Split document ids per class into train / validation / test (by document)."""
    by_label: dict[str, list[str]] = defaultdict(list)
    for doc_id in sorted(labels):
        by_label[labels[doc_id]].append(doc_id)
    rng = random.Random(seed)  # nosec B311
    split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for ids in by_label.values():
        rng.shuffle(ids)
        n_train = round(len(ids) * fractions[0])
        n_val = round(len(ids) * fractions[1])
        split["train"] += ids[:n_train]
        split["val"] += ids[n_train : n_train + n_val]
        split["test"] += ids[n_train + n_val :]
    return split


def _views(
    text: str,
    rng: random.Random,
    n: int,
    include_full: bool,
    crop: tuple[float, float],
    noise: float,
) -> list[str]:
    views = [augment(text, rng, crop=crop, noise=noise) for _ in range(n)]
    return [text, *views] if include_full else views


def _encode(texts: list[str], cfg: TrainConfig) -> list[list[int]]:
    return [ngram_ids(t, cfg.n_buckets, (cfg.ngram_low, cfg.ngram_high)) for t in texts]


def _batches(n: int, size: int, rng: random.Random | None) -> list[list[int]]:
    order = list(range(n))
    if rng is not None:
        rng.shuffle(order)
    return [order[i : i + size] for i in range(0, n, size)]


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module, x: list[list[int]], y: list[int], batch: int
) -> tuple[float, list[int]]:
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    total, preds = 0.0, []
    for idx in _batches(len(x), batch, None):
        ids, offsets = pack([x[i] for i in idx])
        logits = model(ids, offsets)
        target = torch.tensor([y[i] for i in idx])
        total += loss_fn(logits, target).item()
        preds += logits.argmax(dim=1).tolist()
    return total / max(1, len(x)), preds


def confusion(labels: list[str], truth: list[int], preds: list[int]) -> list[list[int]]:
    """Confusion matrix (rows = truth, columns = prediction)."""
    matrix = [[0] * len(labels) for _ in labels]
    for t, p in zip(truth, preds, strict=True):
        matrix[t][p] += 1
    return matrix


Samples = tuple[list[list[int]], list[int]]


def _samples(
    ids: list[str],
    texts: dict[str, str],
    labels: dict[str, str],
    rng: random.Random,
    cfg: TrainConfig,
    *,
    n_views: int,
    include_full: bool,
    fragments: bool = False,
) -> Samples:
    """Encoded views of each document (augmented with ``rng``) and their class indexes."""
    crop = cfg.fragment_crop if fragments else cfg.train_crop
    noise = cfg.fragment_noise if fragments else cfg.train_noise
    xs: list[list[int]] = []
    ys: list[int] = []
    for doc_id in ids:
        views = _views(texts[doc_id], rng, n_views, include_full, crop, noise)
        xs += _encode(views, cfg)
        ys += [DOC_TYPES.index(labels[doc_id])] * len(views)
    return xs, ys


def _train_epoch(
    model: DocTypeMLP,
    optimizer: torch.optim.Optimizer,
    train: Samples,
    cfg: TrainConfig,
    batch_rng: random.Random,
) -> float:
    """One pass over the shuffled training samples; returns the mean loss."""
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    x, y = train
    running = 0.0
    for idx in _batches(len(x), cfg.batch_size, batch_rng):
        ids, offsets = pack([x[i] for i in idx])
        target = torch.tensor([y[i] for i in idx])
        optimizer.zero_grad()
        loss = loss_fn(model(ids, offsets), target)
        loss.backward()
        optimizer.step()
        running += loss.item() * len(idx)
    return running / len(x)


def _fit(
    model: DocTypeMLP, train: Samples, val: Samples, cfg: TrainConfig, log: Callable[[str], None]
) -> tuple[list[dict[str, float]], int]:
    """Train with early stopping on validation loss; keep the best epoch's weights."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    history: list[dict[str, float]] = []
    batch_rng = random.Random(cfg.seed + 1)  # nosec B311
    for epoch in range(1, cfg.epochs + 1):
        train_loss = _train_epoch(model, optimizer, train, cfg, batch_rng)
        val_loss, val_preds = _evaluate(model, val[0], val[1], cfg.batch_size)
        val_acc = _accuracy(val_preds, val[1])
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_acc": val_acc,
            }
        )
        log(
            f"  epoch {epoch:2d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.3f}"
        )
        if val_loss < best_loss - cfg.min_delta:
            best_loss, best_state, best_epoch, stale = (
                val_loss,
                copy.deepcopy(model.state_dict()),
                epoch,
                0,
            )
            continue
        stale += 1
        if stale >= cfg.patience:
            log(f"  early stopping at epoch {epoch} (best epoch {best_epoch})")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_epoch


def _accuracy(preds: list[int], truth: list[int]) -> float:
    return round(sum(p == t for p, t in zip(preds, truth, strict=True)) / len(truth), 4)


def train_classifier(
    texts: dict[str, str],
    labels: dict[str, str],
    cfg: TrainConfig | None = None,
    log: Callable[[str], None] = print,
) -> tuple[DocTypeMLP, dict[str, Any]]:
    """Train with early stopping and return the best model and its evaluation report."""
    cfg = cfg or TrainConfig()
    started = time.perf_counter()
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)  # nosec B311
    split = stratified_split(labels, cfg.seed)
    samples = functools.partial(_samples, texts=texts, labels=labels, rng=rng, cfg=cfg)
    train = samples(split["train"], n_views=cfg.augment_per_doc, include_full=True)
    val = samples(split["val"], n_views=cfg.eval_views, include_full=True)
    noisy = samples(split["test"], n_views=cfg.eval_views, include_full=False, fragments=True)
    clean = samples(split["test"], n_views=0, include_full=True)

    model = DocTypeMLP(cfg.n_buckets, len(DOC_TYPES), cfg.emb_dim, cfg.hidden, cfg.dropout)
    history, best_epoch = _fit(model, train, val, cfg, log)
    _, noisy_preds = _evaluate(model, noisy[0], noisy[1], cfg.batch_size)
    _, clean_preds = _evaluate(model, clean[0], clean[1], cfg.batch_size)
    report = {
        "labels": list(DOC_TYPES),
        "config": asdict(cfg),
        "split_sizes": {k: len(v) for k, v in split.items()},
        "n_train_samples": len(train[0]),
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "history": history,
        "test_fragment_accuracy": _accuracy(noisy_preds, noisy[1]),
        "test_fragment_samples": len(noisy[1]),
        "test_clean_accuracy": _accuracy(clean_preds, clean[1]),
        "test_clean_samples": len(clean[1]),
        "confusion_matrix_fragments": confusion(list(DOC_TYPES), noisy[1], noisy_preds),
        "train_seconds": round(time.perf_counter() - started, 1),
        "n_parameters": sum(p.numel() for p in model.parameters()),
    }
    return model, report


def save_classifier(path: Path, model: DocTypeMLP, report: dict[str, Any]) -> None:
    """Save weights and the evaluation report in one ``torch.save`` bundle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "report": report}, path)
