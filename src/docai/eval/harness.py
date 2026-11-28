"""Evaluation harness: run a provider once over a corpus, then score any number of
guardrail configurations against the ground truth.

Definitions
-----------
* **field accuracy** — fields whose final value equals the ground truth (a
  ``null`` counts as wrong).
* **needs review** — the guardrails nulled or flagged at least one field. With
  guardrails ``off`` nothing is ever sent to review.
* **straight-through** — every field right *and* no human review needed.
* **silent error** — the document is *not* sent to review but has at least one
  wrong field: bad data published without anyone noticing. This is the number
  guardrails exist to drive to zero.
* **false correction** — a field the model got right that a guardrail changed
  (including nulling it). Must be 0.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from docai.classify import DocumentClassifier, Prediction
from docai.domain.compare import same_value
from docai.domain.doc_types import EXTRACTABLE_TYPES
from docai.domain.schemas import field_names
from docai.extract.pdf import cached_pdf_text
from docai.extract.providers import ExtractionContext, LLMProvider, ProviderError, Usage
from docai.extract.xml_reader import FiscalXml, read_fiscal_xml
from docai.guardrails import Action, GuardrailConfig, GuardrailResult, get_config
from docai.pipeline.core import guard, needs_llm, xml_decision
from docai.pipeline.recovery import (
    TRUNCATION_REASON,
    extract_with_recovery,
    truncation_summary,
)


# ----------------------------------------------------------------- corpus
@dataclass
class CorpusDoc:
    """One synthetic document and its ground truth."""

    doc_id: str
    doc_type: str
    filename: str
    pdf_path: Path
    fields: dict[str, Any]
    xml_path: Path | None = None


def load_corpus(data_dir: Path, limit: int | None = None) -> list[CorpusDoc]:
    """Read ``<data_dir>/truth/*.json`` (sorted by id), optionally the first ``limit``."""
    docs = []
    for path in sorted((data_dir / "truth").glob("*.json")):
        truth = json.loads(path.read_text(encoding="utf-8"))
        xml_name = truth.get("xml_filename")
        docs.append(
            CorpusDoc(
                doc_id=truth["doc_id"],
                doc_type=truth["doc_type"],
                filename=truth["filename"],
                pdf_path=data_dir / "pdf" / truth["filename"],
                fields=truth.get("fields", {}),
                xml_path=data_dir / "xml" / xml_name if xml_name else None,
            )
        )
    return docs[:limit] if limit else docs


# ----------------------------------------------------------------- extraction
@dataclass
class RawRecord:
    """One document after classification and a single provider call (reused by every config)."""

    doc: CorpusDoc
    text: str
    prediction: Prediction
    data: dict[str, Any] | None = None
    usage: Usage | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    xml: FiscalXml | None = None


def extract_corpus(
    docs: list[CorpusDoc],
    provider: LLMProvider,
    classifier: DocumentClassifier,
    *,
    text_cache: Path | None = None,
) -> list[RawRecord]:
    """Classify every document and run the provider on the extractable ones (once)."""
    records = []
    for doc in docs:
        text = cached_pdf_text(doc.pdf_path, text_cache)
        prediction = classifier.predict(text)
        record = RawRecord(doc=doc, text=text, prediction=prediction)
        if doc.xml_path is not None and doc.xml_path.exists():
            record.xml = read_fiscal_xml(doc.xml_path.read_bytes())
        if prediction.label in EXTRACTABLE_TYPES:
            context = ExtractionContext(doc_id=doc.doc_id, filename=doc.filename)
            try:
                recovery = extract_with_recovery(provider, text, prediction.label, context)
            except ProviderError as exc:
                record.error = str(exc)
            else:
                record.usage, record.meta = recovery.usage, recovery.meta()
                if recovery.result is not None:
                    record.data = recovery.result.data
                else:
                    record.error = recovery.attempts[-1].error
        records.append(record)
    return records


# ----------------------------------------------------------------- scoring
@dataclass
class FieldOutcome:
    """One field of one document: truth, raw, final and whether each is right."""

    name: str
    truth: Any
    raw: Any
    final: Any
    raw_ok: bool
    final_ok: bool
    status: str


@dataclass
class DocOutcome:
    """Scored result of one document under one configuration."""

    doc_id: str
    true_type: str
    pred_type: str
    fields: dict[str, FieldOutcome]
    needs_review: bool
    findings: list[dict[str, Any]]
    injected: list[dict[str, Any]]
    llm_called: bool = True
    has_xml: bool = False

    @property
    def all_ok(self) -> bool:
        """Every published field is right."""
        return all(f.final_ok for f in self.fields.values())

    @property
    def raw_all_ok(self) -> bool:
        """Every raw model field is right."""
        return all(f.raw_ok for f in self.fields.values())

    @property
    def straight_through(self) -> bool:
        """Right and not sent to review."""
        return self.all_ok and not self.needs_review

    @property
    def silent_error(self) -> bool:
        """Not sent to review but at least one field is wrong."""
        return not self.needs_review and not self.all_ok


XML_MODES = ("ignore", "cross_check", "first")


def _guard_record(
    rec: RawRecord, cfg: GuardrailConfig, xml_mode: str, registry: Mapping[str, str] | None
) -> tuple[GuardrailResult, bool]:
    """Guardrail result for one record under ``xml_mode`` (same policy as the pipeline)."""
    xml, _ = xml_decision(rec.prediction.label, rec.xml if xml_mode != "ignore" else None)
    xml_first = xml_mode == "first"
    llm_called = needs_llm(xml, xml_first)
    result = guard(
        rec.prediction.label,
        rec.text,
        rec.data if llm_called else None,
        xml,
        cfg,
        xml_first=xml_first,
        registry=registry,
    )
    if llm_called and rec.data is None:  # a failed extraction is visible, never silent
        result.review_reasons.append(rec.error or TRUNCATION_REASON)
    return result, llm_called


def evaluate(
    records: list[RawRecord],
    config: str | GuardrailConfig,
    *,
    xml_mode: str = "ignore",
    registry: Mapping[str, str] | None = None,
) -> list[DocOutcome]:
    """Score every extractable document of ``records`` under one guardrail configuration.

    ``xml_mode``: ``ignore`` (LLM only), ``cross_check`` (LLM output checked
    against the XML when there is one) or ``first`` (XML-first, LLM as fallback).
    ``registry`` (CNPJ -> name master data) enables the name-follows-CNPJ check.
    """
    if xml_mode not in XML_MODES:
        raise ValueError(f"xml_mode must be one of {XML_MODES}")
    cfg = get_config(config)
    outcomes = []
    for rec in records:
        if rec.doc.doc_type not in EXTRACTABLE_TYPES:
            continue
        result, llm_called = _guard_record(rec, cfg, xml_mode, registry)
        final = result.record
        statuses = {k: str(v) for k, v in result.field_status.items()}
        raw_source = rec.data if llm_called else (rec.xml.fields if rec.xml else {})
        fields = {}
        for name in field_names(rec.doc.doc_type):
            truth = rec.doc.fields.get(name)
            raw = (raw_source or {}).get(name)
            value = final.get(name)
            fields[name] = FieldOutcome(
                name=name,
                truth=truth,
                raw=raw,
                final=value,
                raw_ok=same_value(name, raw, truth),
                final_ok=same_value(name, value, truth),
                status=statuses.get(name, str(Action.OK)),
            )
        outcomes.append(
            DocOutcome(
                doc_id=rec.doc.doc_id,
                true_type=rec.doc.doc_type,
                pred_type=rec.prediction.label,
                fields=fields,
                needs_review=result.needs_review,
                findings=[f.to_dict() for f in result.findings],
                injected=rec.meta.get("injected_errors", []) if llm_called else [],
                llm_called=llm_called,
                has_xml=rec.xml is not None,
            )
        )
    return outcomes


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _per_field(fields: list[FieldOutcome]) -> dict[str, dict[str, Any]]:
    by_name: dict[str, list[FieldOutcome]] = defaultdict(list)
    for f in fields:
        by_name[f.name].append(f)
    return {
        name: {
            "n": len(items),
            "raw_accuracy": _rate(sum(f.raw_ok for f in items), len(items)),
            "final_accuracy": _rate(sum(f.final_ok for f in items), len(items)),
        }
        for name, items in sorted(by_name.items())
    }


def _per_type(outcomes: list[DocOutcome]) -> dict[str, dict[str, Any]]:
    out = {}
    for doc_type in EXTRACTABLE_TYPES:
        docs = [o for o in outcomes if o.true_type == doc_type]
        if docs:
            out[doc_type] = {
                "n": len(docs),
                "straight_through_rate": _rate(sum(o.straight_through for o in docs), len(docs)),
                "silent_error_rate": _rate(sum(o.silent_error for o in docs), len(docs)),
            }
    return out


def _error_modes(outcomes: list[DocOutcome]) -> dict[str, dict[str, int]]:
    """For each injected error: fixed (right in the end), caught (review) or missed (silent)."""
    modes: dict[str, Counter[str]] = defaultdict(Counter)
    for o in outcomes:
        for err in o.injected:
            outcome = o.fields.get(err["field"])
            if outcome is None:
                continue
            if outcome.final_ok:
                verdict = "fixed"
            elif outcome.status in (Action.NULLED, Action.FLAGGED) or o.needs_review:
                verdict = "caught"
            else:
                verdict = "missed"
            modes[err["type"]][verdict] += 1
    return {k: dict(v) for k, v in sorted(modes.items())}


def summarize(outcomes: list[DocOutcome], config_name: str) -> dict[str, Any]:
    """Aggregate metrics of one configuration (see the module docstring)."""
    n_docs = len(outcomes)
    fields = [f for o in outcomes for f in o.fields.values()]
    statuses = Counter(f.status for f in fields)
    published_wrong = [
        f for o in outcomes if not o.needs_review for f in o.fields.values() if not f.final_ok
    ]
    return {
        "config": config_name,
        "n_docs": n_docs,
        "n_fields": len(fields),
        "raw_field_accuracy": _rate(sum(f.raw_ok for f in fields), len(fields)),
        "field_accuracy": _rate(sum(f.final_ok for f in fields), len(fields)),
        "docs_with_raw_errors": sum(not o.raw_all_ok for o in outcomes),
        "straight_through": sum(o.straight_through for o in outcomes),
        "straight_through_rate": _rate(sum(o.straight_through for o in outcomes), n_docs),
        "review_rate": _rate(sum(o.needs_review for o in outcomes), n_docs),
        "silent_error_docs": sum(o.silent_error for o in outcomes),
        "silent_error_rate": _rate(sum(o.silent_error for o in outcomes), n_docs),
        "wrong_values_published": sum(f.final is not None for f in published_wrong),
        "corrected_fields": statuses.get(str(Action.CORRECTED), 0),
        "nulled_fields": statuses.get(str(Action.NULLED), 0),
        "flagged_fields": statuses.get(str(Action.FLAGGED), 0),
        "false_corrections": sum(1 for f in fields if f.raw_ok and not f.final_ok),
        "per_field": _per_field(fields),
        "per_type": _per_type(outcomes),
        "error_modes": _error_modes(outcomes),
    }


def routing_summary(records: list[RawRecord]) -> dict[str, Any]:
    """Classifier routing accuracy over every document."""
    n = len(records)
    correct = sum(r.prediction.label == r.doc.doc_type for r in records)
    return {
        "n_docs": n,
        "routing_accuracy": _rate(correct, n),
        "misrouted": [
            {"doc_id": r.doc.doc_id, "true": r.doc.doc_type, "predicted": r.prediction.label}
            for r in records
            if r.prediction.label != r.doc.doc_type
        ],
        "provider_errors": sum(r.error is not None for r in records),
    }


def xml_comparison(
    records: list[RawRecord], config: str = "full", registry: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """XML-first vs LLM-only (and LLM + XML cross-check) on the documents that ship an XML."""
    with_xml = [r for r in records if r.xml is not None and r.doc.doc_type in EXTRACTABLE_TYPES]
    rows = {}
    for label, mode in (
        ("llm_only", "ignore"),
        ("llm_xml_cross_check", "cross_check"),
        ("xml_first", "first"),
    ):
        outcomes = evaluate(with_xml, config, xml_mode=mode, registry=registry)
        fields = [f for o in outcomes for f in o.fields.values()]
        llm_records = [r for r, o in zip(with_xml, outcomes, strict=True) if o.llm_called]
        cost = sum(r.usage.cost_usd for r in llm_records if r.usage)
        rows[label] = {
            "n_docs": len(outcomes),
            "llm_calls": len(llm_records),
            "field_accuracy": _rate(sum(f.final_ok for f in fields), len(fields)),
            "straight_through_rate": _rate(
                sum(o.straight_through for o in outcomes), len(outcomes)
            ),
            "review_rate": _rate(sum(o.needs_review for o in outcomes), len(outcomes)),
            "silent_error_rate": _rate(sum(o.silent_error for o in outcomes), len(outcomes)),
            "cost_usd_per_doc": round(cost / len(outcomes), 5) if outcomes else 0.0,
        }
    return {"config": config, "n_docs": len(with_xml), "strategies": rows}


def percentile(sorted_values: list[float], q: float) -> float:
    """Nearest-rank percentile of an already sorted list.

    >>> percentile([1.0, 2.0, 3.0, 4.0], 50)
    2.0
    """
    index = min(len(sorted_values) - 1, max(0, math.ceil(q / 100 * len(sorted_values)) - 1))
    return sorted_values[index]


def recovery_summary(records: list[RawRecord]) -> dict[str, Any]:
    """Truncation recovery over the corpus (see ``docai.pipeline.recovery``)."""
    return truncation_summary([r.meta for r in records if r.usage is not None])


def usage_summary(records: list[RawRecord], provider: LLMProvider) -> dict[str, Any]:
    """Tokens, estimated cost and latency percentiles of the provider calls."""
    usages = [r.usage for r in records if r.usage is not None]
    if not usages:
        return {"calls": 0}
    latencies = sorted(u.latency_s for u in usages)
    total_cost = sum(u.cost_usd for u in usages)
    return {
        "provider": provider.name,
        "model": provider.model,
        "simulated": provider.simulated,
        "pricing": provider.pricing.label,
        "calls": len(usages),
        "attempts": sum(r.meta.get("attempts", 1) for r in records if r.usage is not None),
        "input_tokens": sum(u.input_tokens for u in usages),
        "output_tokens": sum(u.output_tokens for u in usages),
        "cost_usd_total": round(total_cost, 4),
        "cost_usd_per_doc": round(total_cost / len(usages), 5),
        "cost_usd_per_1000_docs": round(1000 * total_cost / len(usages), 2),
        "latency_s_mean": round(statistics.mean(latencies), 3),
        "latency_s_p50": round(percentile(latencies, 50), 3),
        "latency_s_p95": round(percentile(latencies, 95), 3),
    }


# ----------------------------------------------------------------- replay
def blast_radius(a: list[DocOutcome], b: list[DocOutcome]) -> dict[str, Any]:
    """Field-by-field diff between two runs over the same corpus (A → B)."""
    b_by_id = {o.doc_id: o for o in b}
    per_field: dict[str, Counter[str]] = defaultdict(Counter)
    totals: Counter[str] = Counter()
    docs: Counter[str] = Counter()
    for doc_a in a:
        doc_b = b_by_id[doc_a.doc_id]
        for name, fa in doc_a.fields.items():
            fb = doc_b.fields[name]
            changed = not same_value(name, fa.final, fb.final)
            if not fa.final_ok and fb.final_ok:
                verdict = "fixed"
            elif fa.final_ok and not fb.final_ok:
                verdict = "regressed"
            elif not fa.final_ok and fa.final is not None and fb.final is None:
                verdict = "withheld"  # a wrong value is no longer published
            elif changed:
                verdict = "changed_still_wrong"
            else:
                verdict = "unchanged"
            per_field[name][verdict] += 1
            totals[verdict] += 1
            if changed:
                totals["changed"] += 1
                per_field[name]["changed"] += 1
        docs["straight_through_gained"] += doc_b.straight_through and not doc_a.straight_through
        docs["straight_through_lost"] += doc_a.straight_through and not doc_b.straight_through
        docs["silent_errors_removed"] += doc_a.silent_error and not doc_b.silent_error
        docs["silent_errors_added"] += doc_b.silent_error and not doc_a.silent_error
        docs["sent_to_review"] += doc_b.needs_review and not doc_a.needs_review
    return {
        "fields_compared": sum(len(o.fields) for o in a),
        "totals": dict(totals),
        "documents": dict(docs),
        "per_field": {k: dict(v) for k, v in sorted(per_field.items())},
    }
