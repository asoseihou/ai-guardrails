"""Turn an evaluation result into a run record for the dashboard."""

from __future__ import annotations

from typing import Any

from docai.storage.runs import RunRecord


def run_record(results: dict[str, Any], command: str, config: str = "full") -> RunRecord:
    """Headline numbers of ``results`` under the ``config`` guardrails."""
    summary = results["configs"][config]
    usage = results.get("usage", {})
    recovery = results.get("recovery", {})
    return RunRecord(
        command=command,
        config={
            "guardrails": config,
            "provider": usage.get("provider"),
            "model": usage.get("model"),
            "simulated": usage.get("simulated", True),
            "seed": results.get("provider_seed"),
            "noise": results.get("noise", 1.0),
            "corpus_n": results["corpus"]["n"],
            "corpus_seed": results["corpus"].get("seed"),
        },
        n_docs=summary["n_docs"],
        field_accuracy=summary["field_accuracy"],
        review_rate=summary["review_rate"],
        silent_error_rate=summary["silent_error_rate"],
        straight_through=summary["straight_through_rate"],
        cost_per_doc=float(usage.get("cost_usd_per_doc", 0.0)),
        latency_p50=float(usage.get("latency_s_p50", 0.0)),
        latency_p95=float(usage.get("latency_s_p95", 0.0)),
        retries=int(recovery.get("retries", 0)),
        per_field={
            name: {"raw_accuracy": v["raw_accuracy"], "final_accuracy": v["final_accuracy"]}
            for name, v in summary["per_field"].items()
        },
    )
