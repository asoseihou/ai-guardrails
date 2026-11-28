"""Run tracking (SQLite), the /metrics/runs endpoint and the dashboard page."""

import json

from fastapi.testclient import TestClient

from docai import container
from docai.cli import main
from docai.eval.tracking import run_record
from docai.storage import RunRecord, RunRepository


def _run(**overrides):
    values = {
        "command": "eval",
        "config": {"noise": 1.0, "seed": 7},
        "n_docs": 10,
        "field_accuracy": 0.95,
        "review_rate": 0.2,
        "silent_error_rate": 0.01,
        "straight_through": 0.79,
        "cost_per_doc": 0.004,
        "latency_p50": 1.1,
        "latency_p95": 2.3,
        "retries": 1,
        "per_field": {"number": {"raw_accuracy": 0.9, "final_accuracy": 1.0}},
    }
    return RunRecord(**{**values, **overrides})


def test_run_repository_keeps_order_and_limit(tmp_path):
    repo = RunRepository(tmp_path / "runs.db")
    ids = [repo.add(_run(created_at=f"2026-01-0{i}T00:00:00+00:00")) for i in range(1, 4)]
    runs = repo.list()
    assert [r.id for r in runs] == ids  # oldest first
    assert [r.id for r in repo.list(limit=2)] == ids[1:]
    assert runs[0].per_field["number"]["final_accuracy"] == 1.0


def test_run_record_from_results():
    summary = {
        "n_docs": 5,
        "field_accuracy": 0.9,
        "review_rate": 0.4,
        "silent_error_rate": 0.0,
        "straight_through_rate": 0.6,
        "per_field": {"cfop": {"raw_accuracy": 0.8, "final_accuracy": 0.8, "n": 5}},
    }
    results = {
        "configs": {"full": summary},
        "usage": {
            "provider": "simulated",
            "cost_usd_per_doc": 0.01,
            "latency_s_p50": 1.0,
            "latency_s_p95": 2.0,
        },
        "recovery": {"retries": 3},
        "corpus": {"n": 8, "seed": 1},
        "provider_seed": 7,
        "noise": 2.0,
    }
    run = run_record(results, "eval")
    assert (run.field_accuracy, run.retries, run.latency_p95) == (0.9, 3, 2.0)
    assert run.config["noise"] == 2.0 and "n" not in run.per_field["cfop"]


def test_metrics_endpoint_and_dashboard_page(api_env, tmp_path):
    from docai.api.app import create_app

    client = TestClient(create_app())
    assert client.get("/metrics/runs").json() == {"runs": []}
    container.get_run_repository().add(_run())
    runs = client.get("/metrics/runs").json()["runs"]
    assert len(runs) == 1 and runs[0]["field_accuracy"] == 0.95
    assert client.get("/metrics/runs?limit=0").status_code == 422
    page = client.get("/dashboard")
    assert page.status_code == 200 and 'src="/static/dashboard.js"' in page.text
    assert "<script>" not in page.text and "style=" not in page.text
    assert "'unsafe-inline'" not in page.headers["content-security-policy"]
    assert client.get("/static/dashboard.js").status_code == 200


def test_cli_eval_records_a_run(api_env, corpus_dir, tmp_path, capsys):
    args = [
        "eval",
        "--data",
        str(corpus_dir),
        "--model",
        str(tmp_path / "none.pt"),
        "--report-dir",
        str(tmp_path / "reports"),
        "--noise",
        "2",
    ]
    assert main(args) == 0
    assert "recorded run" in capsys.readouterr().out
    runs = container.get_run_repository().list()
    assert len(runs) == 1 and runs[0].config["noise"] == 2.0
    report = json.loads((tmp_path / "reports" / "eval_report.json").read_text())
    assert report["recovery"]["calls"] > 0 and "xml" in report
    assert main([*args, "--no-track"]) == 0
    assert len(container.get_run_repository().list()) == 1
