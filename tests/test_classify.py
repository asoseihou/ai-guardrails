import random

import pytest

from docai.classify import KeywordClassifier, load_classifier
from docai.classify.features import augment, ngram_ids, normalize_for_features
from docai.domain.doc_types import DOC_TYPES

torch = pytest.importorskip("torch")

from docai.classify.infer import TorchDocTypeClassifier  # noqa: E402
from docai.classify.train import (  # noqa: E402
    TrainConfig,
    save_classifier,
    stratified_split,
    train_classifier,
)


def test_features_are_stable_and_bounded():
    ids = ngram_ids("DANFE 000.123.456", n_buckets=1024)
    assert ids == ngram_ids("danfe 999.999.999", n_buckets=1024)  # case and digits normalized
    assert all(0 <= i < 1024 for i in ids)
    assert normalize_for_features("Série 001  NÚMERO") == "serie 000 numero"
    assert ngram_ids("") == [0]


def test_augment_is_seeded():
    text = "\n".join(f"line {i} with some words" for i in range(30))
    assert augment(text, random.Random(1)) == augment(text, random.Random(1))
    assert len(augment(text, random.Random(2), crop=(0.1, 0.1))) < len(text)


def test_stratified_split_is_disjoint():
    labels = {f"d{i}": DOC_TYPES[i % 5] for i in range(50)}
    split = stratified_split(labels, seed=0)
    ids = split["train"] + split["val"] + split["test"]
    assert sorted(ids) == sorted(labels)
    assert {labels[i] for i in split["test"]} == set(DOC_TYPES)


def test_keyword_classifier(corpus, texts):
    clf = KeywordClassifier()
    for doc in corpus:
        assert clf.predict(texts[doc["doc_id"]]).label == doc["doc_type"]


def test_train_save_load_predict(corpus, texts, tmp_path):
    labels = {d["doc_id"]: d["doc_type"] for d in corpus}
    cfg = TrainConfig(epochs=4, augment_per_doc=2, eval_views=1, n_buckets=4096, seed=3)
    model, report = train_classifier(texts, labels, cfg, log=lambda *_: None)
    assert report["epochs_run"] <= 4
    assert 0 <= report["test_clean_accuracy"] <= 1
    assert len(report["confusion_matrix_fragments"]) == len(DOC_TYPES)
    path = tmp_path / "clf.pt"
    save_classifier(path, model, report)
    loaded = load_classifier(path)
    assert isinstance(loaded, TorchDocTypeClassifier)
    text = texts[corpus[0]["doc_id"]]
    prediction = loaded.predict(text)
    assert prediction.label in DOC_TYPES
    assert abs(sum(prediction.probabilities.values()) - 1) < 1e-3
    model.eval()
    assert TorchDocTypeClassifier(model, report).predict(text) == prediction


def test_training_is_deterministic(corpus, texts):
    labels = {d["doc_id"]: d["doc_type"] for d in corpus}
    cfg = TrainConfig(epochs=2, augment_per_doc=1, eval_views=1, n_buckets=2048, seed=5)
    _, a = train_classifier(texts, labels, cfg, log=lambda *_: None)
    _, b = train_classifier(texts, labels, cfg, log=lambda *_: None)
    assert a["history"] == b["history"]


def test_fallback_when_weights_missing(tmp_path):
    assert isinstance(load_classifier(tmp_path / "missing.pt"), KeywordClassifier)
    broken = tmp_path / "broken.pt"
    broken.write_bytes(b"not a model")
    assert isinstance(load_classifier(broken), KeywordClassifier)
