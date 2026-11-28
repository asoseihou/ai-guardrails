from __future__ import annotations

import json
from pathlib import Path

import pytest

from docai.extract.pdf import pdf_to_text
from docai.synth import generate_corpus

TINY_TYPES = ["nfe"] * 6 + ["cte"] * 6 + ["nfse"] * 6 + ["boleto"] * 3 + ["other"] * 3


@pytest.fixture(scope="session")
def corpus_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("corpus")
    generate_corpus(0, out, seed=1234, types=TINY_TYPES)
    return out


@pytest.fixture(scope="session")
def corpus(corpus_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((corpus_dir / "truth").glob("*.json"))]


@pytest.fixture(scope="session")
def texts(corpus_dir: Path, corpus: list[dict]) -> dict[str, str]:
    return {d["doc_id"]: pdf_to_text(corpus_dir / "pdf" / d["filename"]) for d in corpus}


@pytest.fixture(scope="session")
def by_type(corpus: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for doc in corpus:
        out.setdefault(doc["doc_type"], []).append(doc)
    return out


@pytest.fixture
def api_env(corpus_dir, tmp_path, monkeypatch):
    """Settings for API tests: tiny corpus as samples, keyword classifier, no rate limit."""
    from docai import container

    for name in ("DOCAI_API_TOKEN", "DOCAI_CORS_ORIGINS", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DOCAI_DATA_DIR", str(corpus_dir))
    monkeypatch.setenv("DOCAI_SAMPLES_DIR", str(corpus_dir))  # serve the tiny corpus as samples
    monkeypatch.setenv("DOCAI_MODEL_PATH", str(tmp_path / "no-model.pt"))  # keyword fallback
    monkeypatch.setenv("DOCAI_DB_PATH", str(tmp_path / "docai.db"))
    monkeypatch.setenv("DOCAI_PROVIDER", "simulated")
    monkeypatch.setenv("DOCAI_RATE_LIMIT_PER_MINUTE", "0")
    container.reset()
    yield monkeypatch
    container.reset()
