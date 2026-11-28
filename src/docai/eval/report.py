"""Markdown + JSON reports."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

PER_FIELD_RATES = {"cnpj_digit_swap", "dropped_field"}

CONFIG_ORDER = ("off", "dv-only", "text", "full")
CONFIG_LABELS = {
    "off": "No guardrails",
    "dv-only": "Check digits (v1)",
    "text": "+ presence & consistency (v2)",
    "full": "+ layout anchors & registry (v3)",
}

SIMULATED_NOTE = (
    "> **Note on honesty.** Extraction metrics in this report come from a **seeded simulated "
    "LLM**: the ground truth of each synthetic document with realistic errors injected at fixed "
    "rates (see *Error model*). They measure what the guardrails do with those errors; they are "
    "**not** measurements of a real model. Run `python -m docai eval --provider anthropic` "
    "(or `gemini` / `bedrock`) with credentials to benchmark a real provider on the same corpus."
)


def pct(value: float) -> str:
    """``0.1234`` -> ``'12.3%'``."""
    return f"{100 * value:.1f}%"


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def headline_rows(configs: dict[str, dict[str, Any]]) -> list[list[Any]]:
    """Rows of the headline table: one metric per row, one guardrail config per column."""
    names = [c for c in CONFIG_ORDER if c in configs]
    metrics: list[tuple[str, Callable[[dict[str, Any]], Any]]] = [
        ("Field accuracy", lambda s: pct(s["field_accuracy"])),
        (
            "Straight-through documents (all fields right, no review)",
            lambda s: pct(s["straight_through_rate"]),
        ),
        ("Documents sent to human review", lambda s: pct(s["review_rate"])),
        (
            "**Silent-error documents** (wrong data, no review)",
            lambda s: f"**{pct(s['silent_error_rate'])}** ({s['silent_error_docs']})",
        ),
        ("Wrong values published", lambda s: s["wrong_values_published"]),
        (
            "Fields corrected / nulled / flagged",
            lambda s: f"{s['corrected_fields']} / {s['nulled_fields']} / {s['flagged_fields']}",
        ),
        ("False corrections (a correct value changed)", lambda s: s["false_corrections"]),
    ]
    return [[label, *(fn(configs[n]) for n in names)] for label, fn in metrics]


def results_table(results: dict[str, Any]) -> str:
    """Compact table for the README."""
    configs = results["configs"]
    names = [c for c in CONFIG_ORDER if c in configs]
    table = _table(["Metric", *(CONFIG_LABELS[n] for n in names)], headline_rows(configs))
    extra = []
    clf = results.get("classifier")
    if clf:
        extra.append(
            f"| Document-type classifier, test accuracy (full text / short noisy fragments) | "
            f"{pct(clf['test_clean_accuracy'])} / {pct(clf['test_fragment_accuracy'])} |"
        )
    rag = results.get("rag")
    if rag:
        extra.append(
            f"| RAG retrieval on {rag['n_questions']} questions (hit@1 / hit@3 / MRR) | "
            f"{pct(rag['hit_at_1'])} / {pct(rag['hit_at_3'])} / {rag['mrr']:.3f} |"
        )
    xml = results.get("xml")
    if xml and xml["n_docs"]:
        llm, first = xml["strategies"]["llm_only"], xml["strategies"]["xml_first"]
        extra.append(
            f"| XML-first vs LLM-only on the {xml['n_docs']} documents with an XML "
            f"(field accuracy · LLM calls) | {pct(first['field_accuracy'])} · "
            f"{first['llm_calls']} vs {pct(llm['field_accuracy'])} · {llm['llm_calls']} |"
        )
    recovery = results.get("recovery")
    if recovery and recovery["truncated"]:
        extra.append(
            f"| Truncated answers recovered by one summary-mode retry (rest sent to review) "
            f"| {recovery['recovered']} of {recovery['truncated']} · "
            f"${recovery['cost_avoided_usd']:.4f} saved vs a "
            f"{recovery['naive_max_retries']}-retry loop |"
        )
    usage = results.get("usage", {})
    if usage.get("calls"):
        extra.append(
            f"| Estimated cost per 1,000 documents (illustrative pricing, simulated tokens) | "
            f"${usage['cost_usd_per_1000_docs']:.2f} |"
        )
    corpus = results["corpus"]
    caption = (
        f"Extraction rows: {results['configs'][names[0]]['n_docs']} extractable documents "
        f"(NF-e, CT-e, NFS-e) out of {corpus['n']} synthetic PDFs, seeded **simulated LLM** "
        f"(seed {results['provider_seed']})."
    )
    other = "\n\n| Other components | Result |\n|---|---|\n" + "\n".join(extra) if extra else ""
    return f"{table}\n\n{caption}{other}"


XML_LABELS = {
    "llm_only": "LLM only (XML ignored)",
    "llm_xml_cross_check": "LLM + XML cross-check (XML wins)",
    "xml_first": "XML-first (LLM only as fallback)",
}


def _xml_section(xml: dict[str, Any] | None) -> list[str]:
    if not xml or not xml["n_docs"]:
        return []
    rows = [
        [
            XML_LABELS[name],
            row["llm_calls"],
            pct(row["field_accuracy"]),
            pct(row["straight_through_rate"]),
            pct(row["review_rate"]),
            pct(row["silent_error_rate"]),
            f"${row['cost_usd_per_doc']:.5f}",
        ]
        for name, row in xml["strategies"].items()
    ]
    return [
        "## XML-first vs LLM-only",
        "",
        f"The {xml['n_docs']} NF-e / NFS-e documents that ship with their fiscal XML, "
        f"`{xml['config']}` guardrails in every row. The XML is read deterministically "
        "(defusedxml); the LLM is only called when the XML is missing or incomplete.",
        "",
        _table(
            [
                "Strategy",
                "LLM calls",
                "Field accuracy",
                "Straight-through",
                "Review",
                "Silent errors",
                "Cost / doc",
            ],
            rows,
        ),
        "",
    ]


def _header(results: dict[str, Any], names: list[str]) -> list[str]:
    configs = results["configs"]
    usage = results["usage"]
    note = (
        SIMULATED_NOTE
        if usage.get("simulated", True)
        else "> Extraction metrics come from a **real provider** "
        f"({usage.get('provider')}, {usage.get('model')})."
    )
    corpus = results["corpus"]
    return [
        "# Evaluation report",
        "",
        f"Generated by `python -m docai {results.get('command', 'demo')}` on "
        f"{results.get('generated_on', 'n/a')}.",
        "",
        note,
        "",
        "## Corpus",
        "",
        f"{corpus['n']} synthetic PDFs (seed {corpus['seed']}): "
        + ", ".join(f"{k} {v}" for k, v in corpus["counts"].items())
        + f"; {corpus.get('with_xml', 0)} ship with their fiscal XML. Extraction is evaluated "
        f"on the {configs[names[0]]['n_docs']} NF-e / CT-e / NFS-e documents "
        f"({configs[names[0]]['n_fields']} fields).",
        "",
        "## Headline",
        "",
        _table(["Metric", *(CONFIG_LABELS[n] for n in names)], headline_rows(configs)),
        "",
        "*Silent error*: the document passes without review but contains at least one wrong "
        "field. *False correction*: a field the model got right that a guardrail changed.",
        "",
    ]


def _error_mode_section(results: dict[str, Any], names: list[str]) -> list[str]:
    configs = results["configs"]
    full = configs["full"]
    rows = []
    for mode, verdicts in full["error_modes"].items():
        missed = [configs[n]["error_modes"].get(mode, {}).get("missed", 0) for n in names]
        rows.append(
            [
                f"`{mode}`",
                sum(verdicts.values()),
                verdicts.get("fixed", 0),
                verdicts.get("caught", 0),
                *missed,
            ]
        )
    rates = ", ".join(
        f"`{k}`={v}" + (" per field" if k in PER_FIELD_RATES else "")
        for k, v in results["error_rates"].items()
    )
    return [
        "## Error model and the measured effect of each guardrail level",
        "",
        f"Injected-error rates: {rates} (others per document).",
        "",
        "*Fixed* / *Caught* are for the full guardrails (v3): fixed = the final value is right "
        "(corrected from the access key, the XML or a deterministic re-read of the box that "
        "owns the value); caught = still wrong or empty but the document goes to review. The "
        "*Missed* columns count errors published silently at each level.",
        "",
        _table(
            [
                "Error mode",
                "Injected",
                "Fixed (v3)",
                "Caught (v3)",
                *(f"Missed: {CONFIG_LABELS[n]}" for n in names),
            ],
            rows,
        ),
        "",
        "`party_role_swap` and `payer_into_recipient` produce values that are printed, valid "
        "and correctly paired; only the layout anchors (the CNPJ printed inside each role "
        "block) can see them. Anchors depend on the printed layout: on unknown layouts they "
        "do nothing and the v2 checks still apply.",
        "",
    ]


def _field_sections(full: dict[str, Any]) -> list[str]:
    return [
        "## Per-field accuracy",
        "",
        _table(
            ["Field", "Docs", "Raw model output", "After full guardrails"],
            [
                [f"`{k}`", v["n"], pct(v["raw_accuracy"]), pct(v["final_accuracy"])]
                for k, v in full["per_field"].items()
            ],
        ),
        "",
        "## Per document type (full guardrails)",
        "",
        _table(
            ["Type", "Docs", "Straight-through", "Silent errors"],
            [
                [k, v["n"], pct(v["straight_through_rate"]), pct(v["silent_error_rate"])]
                for k, v in full["per_type"].items()
            ],
        ),
        "",
    ]


def _replay_section(replay: dict[str, Any]) -> list[str]:
    t, d = replay["totals"], replay["documents"]
    rows = [
        [
            f"`{k}`",
            v.get("changed", 0),
            v.get("fixed", 0),
            v.get("withheld", 0),
            v.get("regressed", 0),
        ]
        for k, v in replay["per_field"].items()
        if v.get("changed", 0)
    ]
    return [
        f"## Replay: `{replay['a']}` → `{replay['b']}` (blast radius)",
        "",
        f"Same model outputs, two guardrail configurations. {replay['fields_compared']} "
        f"fields compared: **{t.get('changed', 0)} changed**, {t.get('fixed', 0)} fixed, "
        f"{t.get('withheld', 0)} wrong values withheld for review, "
        f"**{t.get('regressed', 0)} regressed**. Documents: "
        f"+{d.get('straight_through_gained', 0)} straight-through, "
        f"-{d.get('straight_through_lost', 0)} straight-through lost, "
        f"{d.get('silent_errors_removed', 0)} silent errors removed, "
        f"{d.get('silent_errors_added', 0)} added.",
        "",
        _table(["Field", "Changed", "Fixed", "Withheld", "Regressed"], rows),
        "",
    ]


def _classifier_section(clf: dict[str, Any], routing: dict[str, Any] | None) -> list[str]:
    labels = clf["labels"]
    matrix = clf["confusion_matrix_fragments"]
    out = [
        "## Document-type classifier (PyTorch)",
        "",
        f"Hashed character 3-4-gram EmbeddingBag + MLP, {clf['n_parameters']:,} parameters, "
        f"trained in {clf['train_seconds']}s on CPU ({clf['epochs_run']} epochs, best "
        f"{clf['best_epoch']}, early stopping). Split by document: "
        + " / ".join(f"{v} {k}" for k, v in clf["split_sizes"].items())
        + " documents.",
        "",
        f"- Test accuracy on full clean text: **{pct(clf['test_clean_accuracy'])}** "
        f"({clf['test_clean_samples']} documents)",
        f"- Test accuracy on short noisy fragments (8-20% of the lines, 6% character noise): "
        f"**{pct(clf['test_fragment_accuracy'])}** ({clf['test_fragment_samples']} fragments)",
        "",
        "Confusion matrix on fragments (rows = truth, columns = prediction):",
        "",
        _table(["", *labels], [[f"**{labels[i]}**", *row] for i, row in enumerate(matrix)]),
        "",
    ]
    if routing:
        out += [
            f"Routing accuracy of the pipeline over all {routing['n_docs']} documents: "
            f"{pct(routing['routing_accuracy'])} (misrouted: {len(routing['misrouted'])}).",
            "",
        ]
    return out


def _rag_section(rag: dict[str, Any]) -> list[str]:
    return [
        "## RAG retrieval (BM25)",
        "",
        f"{rag['n_questions']} hand-written questions (English and Portuguese) over the "
        f"`knowledge/` notes: **hit@1 {pct(rag['hit_at_1'])}, hit@3 {pct(rag['hit_at_3'])}, "
        f"MRR {rag['mrr']:.3f}**.",
        "",
        _table(
            ["Question", "Rank of first relevant section"],
            [[r["question"], r["rank"] or "—"] for r in rag["rows"]],
        ),
        "",
    ]


def _usage_section(usage: dict[str, Any]) -> list[str]:
    tag = " (simulated)" if usage.get("simulated") else ""
    return [
        "## Cost and latency" + tag,
        "",
        f"{usage['calls']} documents sent to the model ({usage.get('attempts', usage['calls'])} "
        f"calls including retries), {usage['input_tokens']:,} input / "
        f"{usage['output_tokens']:,} output tokens (≈4 characters per token). Pricing: "
        f"{usage['pricing']}.",
        "",
        f"- Estimated cost: ${usage['cost_usd_total']:.4f} total, "
        f"${usage['cost_usd_per_1000_docs']:.2f} per 1,000 documents",
        f"- Latency{tag}: p50 {usage.get('latency_s_p50', usage['latency_s_mean'])}s, "
        f"p95 {usage['latency_s_p95']}s",
        "",
    ]


def _recovery_section(recovery: dict[str, Any]) -> list[str]:
    return [
        "## Truncation recovery",
        "",
        f"{recovery['truncated']} of {recovery['calls']} first answers were cut at the output "
        f"limit. One retry in summary mode recovered **{recovery['recovered']}**; "
        f"{recovery['sent_to_review']} went to human review. Retries: {recovery['retries']} "
        f"calls, ${recovery['retry_cost_usd']:.4f}.",
        "",
        f"A naive loop that repeats the identical request up to {recovery['naive_max_retries']} "
        f"times would have made {recovery['naive_extra_calls']} extra full-budget calls and, at "
        "temperature 0, recovered none of them (the answer is cut at the same place every "
        f"time). **Cost avoided: ${recovery['cost_avoided_usd']:.4f}** on this corpus.",
        "",
    ]


REPRODUCE = ["## Reproduce", "", "```bash", "make install && make demo", "```", ""]


def render_markdown(results: dict[str, Any]) -> str:
    """The full evaluation report."""
    configs = results["configs"]
    names = [c for c in CONFIG_ORDER if c in configs]
    out = _header(results, names)
    if "full" in configs:
        out += _error_mode_section(results, names) + _field_sections(configs["full"])
    for replay in results.get("replays", []):
        out += _replay_section(replay)
    out += _xml_section(results.get("xml"))
    if results.get("classifier"):
        out += _classifier_section(results["classifier"], results.get("routing"))
    if results.get("rag"):
        out += _rag_section(results["rag"])
    if (recovery := results.get("recovery")) and recovery["calls"]:
        out += _recovery_section(recovery)
    if results["usage"].get("calls"):
        out += _usage_section(results["usage"])
    if timings := results.get("timings"):
        out += ["## Runtime of this run", "", ", ".join(f"{k} {v}s" for k, v in timings.items())]
        out += [""]
    return "\n".join(out + REPRODUCE)


def write_reports(results: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    """Write ``eval_report.md`` and ``eval_report.json`` into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path, json_path = out_dir / "eval_report.md", out_dir / "eval_report.json"
    md_path.write_text(render_markdown(results), encoding="utf-8")
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return md_path, json_path


def update_readme(readme: Path, table: str) -> bool:
    """Replace the block between the RESULTS markers of the README; False if not found."""
    start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    text = readme.read_text(encoding="utf-8")
    if start not in text or end not in text:
        return False
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    readme.write_text(f"{head}{start}\n{table}\n{end}{tail}", encoding="utf-8")
    return True
