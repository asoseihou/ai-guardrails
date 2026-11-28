import json

import pytest

from docai.classify import KeywordClassifier
from docai.eval import (
    blast_radius,
    evaluate,
    extract_corpus,
    load_corpus,
    routing_summary,
    same_value,
    summarize,
    usage_summary,
)
from docai.eval.report import render_markdown, results_table, update_readme, write_reports
from docai.extract.providers import ErrorRates, SimulatedLLMProvider, TruthStore


@pytest.fixture(scope="module")
def clean_records(corpus_dir):
    provider = SimulatedLLMProvider(TruthStore([corpus_dir]), ErrorRates.none())
    return extract_corpus(load_corpus(corpus_dir), provider, KeywordClassifier())


@pytest.fixture(scope="module")
def noisy(corpus_dir):
    provider = SimulatedLLMProvider(TruthStore([corpus_dir]), ErrorRates().scaled(3), seed=11)
    records = extract_corpus(load_corpus(corpus_dir), provider, KeywordClassifier())
    return provider, records


def test_same_value():
    assert same_value("issuer_cnpj", "11.222.333/0001-81", "11222333000181")
    assert same_value("total_value", "1.234,56", 1234.56)
    assert same_value("number", "000012345", "12345")
    assert same_value("issuer_name", "AURORA LOGÍSTICA LTDA", "Aurora Logistica Ltda")
    assert not same_value("total_value", 1234.5, 1234.56)
    assert not same_value("number", None, "1")
    assert same_value("number", None, None)


def test_perfect_model_is_perfect(clean_records):
    for config in ("off", "dv-only", "full"):
        s = summarize(evaluate(clean_records, config), config)
        assert s["field_accuracy"] == 1.0
        assert s["straight_through_rate"] == 1.0
        assert s["review_rate"] == 0.0
        assert s["false_corrections"] == 0


def test_guardrails_reduce_silent_errors(noisy):
    _, records = noisy
    off = summarize(evaluate(records, "off"), "off")
    full = summarize(evaluate(records, "full"), "full")
    failed = sum(r.error is not None for r in records if r.doc.doc_type in ("nfe", "cte", "nfse"))
    assert off["review_rate"] == round(failed / off["n_docs"], 4)  # only failed extractions
    assert off["silent_error_docs"] > 0
    assert full["silent_error_docs"] < off["silent_error_docs"]
    assert full["field_accuracy"] >= off["field_accuracy"]
    assert full["wrong_values_published"] < off["wrong_values_published"]
    assert full["false_corrections"] == 0
    assert full["corrected_fields"] > 0
    assert sum(sum(v.values()) for v in full["error_modes"].values()) > 0


def test_blast_radius(noisy):
    _, records = noisy
    a, b = evaluate(records, "dv-only"), evaluate(records, "full")
    radius = blast_radius(a, b)
    assert radius["totals"].get("regressed", 0) == 0
    assert radius["totals"].get("fixed", 0) > 0
    assert radius["fields_compared"] == sum(len(o.fields) for o in a)
    assert blast_radius(a, a)["totals"].get("changed", 0) == 0


def test_usage_and_routing(noisy):
    provider, records = noisy
    usage = usage_summary(records, provider)
    assert usage["simulated"] is True
    assert usage["calls"] == 18 and usage["cost_usd_total"] > 0
    routing = routing_summary(records)
    assert routing["routing_accuracy"] == 1.0


def test_reports(noisy, tmp_path):
    provider, records = noisy
    outcomes = {c: evaluate(records, c) for c in ("off", "dv-only", "full")}
    results = {
        "corpus": {"n": len(records), "seed": 1234, "counts": {"nfe": 6}},
        "provider_seed": 11,
        "error_rates": ErrorRates().to_dict(),
        "usage": usage_summary(records, provider),
        "routing": routing_summary(records),
        "configs": {c: summarize(o, c) for c, o in outcomes.items()},
        "replays": [
            {"a": "dv-only", "b": "full", **blast_radius(outcomes["dv-only"], outcomes["full"])}
        ],
        "rag": {
            "n_questions": 2,
            "k": 3,
            "hit_at_1": 0.5,
            "hit_at_3": 1.0,
            "mrr": 0.75,
            "rows": [],
        },
    }
    markdown = render_markdown(results)
    assert "simulated" in markdown.lower() and "not** measurements of a real model" in markdown
    table = results_table(results)
    assert "Silent-error documents" in table and "simulated LLM" in table
    _, js = write_reports(results, tmp_path)
    assert json.loads(js.read_text())["configs"]["full"]["false_corrections"] == 0
    readme = tmp_path / "README.md"
    readme.write_text("intro\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\noutro\n")
    assert update_readme(readme, table)
    assert "old" not in readme.read_text() and table in readme.read_text()
