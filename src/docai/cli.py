"""Command-line interface: ``python -m docai <command>`` (or the ``docai`` script)."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

from docai.domain.doc_types import DOC_TYPES
from docai.settings import Settings, get_settings

ALL_CONFIGS = ["off", "dv-only", "text", "full"]


# ------------------------------------------------------------------ helpers
def _say(message: str) -> None:
    print(message, flush=True)


def _load_texts(data_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    from docai.eval import load_corpus
    from docai.extract.pdf import cached_pdf_text

    texts, labels = {}, {}
    for doc in load_corpus(data_dir):
        texts[doc.doc_id] = cached_pdf_text(doc.pdf_path, data_dir / "text")
        labels[doc.doc_id] = doc.doc_type
    return texts, labels


def _classifier_report(model_path: Path) -> dict[str, Any] | None:
    from docai.classify import TorchDocTypeClassifier, load_classifier

    classifier = load_classifier(model_path)
    if isinstance(classifier, TorchDocTypeClassifier):
        return {k: v for k, v in classifier.report.items() if k != "history"}
    return None


def _rag_eval(knowledge: Path) -> dict[str, Any]:
    from docai.rag import KnowledgeBase, evaluate_retrieval, load_qa_set

    return evaluate_retrieval(KnowledgeBase.from_dir(knowledge), load_qa_set(), k=3)


def _corpus_summary(data_dir: Path, n: int) -> dict[str, Any]:
    """Size, seed, per-type counts and XML share of the corpus (from its manifest)."""
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    return {
        "n": n,
        "seed": manifest.get("seed"),
        "counts": manifest.get("counts", {}),
        "with_xml": manifest.get("with_xml", 0),
    }


def _run_eval(
    data_dir: Path,
    provider_name: str,
    configs: list[str],
    replays: list[tuple[str, str]],
    model_path: Path,
    *,
    limit: int | None = None,
    seed: int = 7,
    noise: float = 1.0,
) -> dict[str, Any]:
    """Extract the corpus once, score every config and collect the report sections."""
    from docai.classify import load_classifier
    from docai.eval import (
        blast_radius,
        evaluate,
        extract_corpus,
        load_corpus,
        recovery_summary,
        routing_summary,
        summarize,
        usage_summary,
        xml_comparison,
    )
    from docai.extract.providers import ErrorRates, TruthStore, get_provider

    docs = load_corpus(data_dir, limit)
    if not docs:
        raise SystemExit(f"no corpus in {data_dir}; run `python -m docai synth` first")
    classifier = load_classifier(model_path)
    rates = ErrorRates().scaled(noise)
    provider = get_provider(
        provider_name, get_settings(), truth_roots=[data_dir], seed=seed, rates=rates
    )
    _say(
        f"• extracting {len(docs)} documents with provider={provider.name} "
        f"model={provider.model} classifier={classifier.name}"
    )
    records = extract_corpus(docs, provider, classifier, text_cache=data_dir / "text")
    registry = {p["cnpj"]: p["name"] for p in TruthStore([data_dir]).parties()}
    outcomes = {name: evaluate(records, name, registry=registry) for name in configs}
    return {
        "generated_on": date.today().isoformat(),
        "corpus": _corpus_summary(data_dir, len(docs)),
        "provider_seed": seed,
        "noise": noise,
        "error_rates": rates.to_dict() if provider.simulated else {},
        "usage": usage_summary(records, provider),
        "routing": routing_summary(records),
        "xml": xml_comparison(records, registry=registry),
        "recovery": recovery_summary(records),
        "configs": {name: summarize(o, name) for name, o in outcomes.items()},
        "replays": [
            {"a": a, "b": b, **blast_radius(outcomes[a], outcomes[b])}
            for a, b in replays
            if a in outcomes and b in outcomes
        ],
    }


def _print_headline(results: dict[str, Any]) -> None:
    from docai.eval.report import results_table

    _say("")
    _say(results_table(results))
    _say("")


# ------------------------------------------------------------------ commands
def cmd_synth(args: argparse.Namespace) -> int:
    """Generate synthetic PDFs, XMLs and ground truth."""
    from docai.synth import generate_corpus

    out = Path(args.out)
    if args.fresh and out.exists():
        for sub in ("pdf", "truth", "text", "xml"):
            shutil.rmtree(out / sub, ignore_errors=True)
    types = args.types.split(",") if args.types else None
    if types and (bad := set(types) - set(DOC_TYPES)):
        raise SystemExit(f"unknown types: {sorted(bad)}")
    docs = generate_corpus(
        args.n, out, seed=args.seed, prefix=args.prefix, types=types, xml_share=args.xml_share
    )
    counts: dict[str, int] = {}
    for doc in docs:
        counts[doc["doc_type"]] = counts.get(doc["doc_type"], 0) + 1
    _say(f"wrote {len(docs)} documents to {out} {counts}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train the document-type classifier."""
    from docai.classify.train import TrainConfig, save_classifier, train_classifier

    texts, labels = _load_texts(Path(args.data))
    cfg = TrainConfig(epochs=args.epochs, seed=args.seed)
    model, report = train_classifier(texts, labels, cfg, log=_say)
    save_classifier(Path(args.out), model, report)
    _say(
        f"saved {args.out}: clean test acc {report['test_clean_accuracy']:.3f}, "
        f"fragment test acc {report['test_fragment_accuracy']:.3f} "
        f"({report['train_seconds']}s)"
    )
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Evaluate extraction + guardrails, write the reports and record the run."""
    from docai.eval.report import write_reports

    configs = args.configs.split(",")
    replays = [("text", "full"), ("dv-only", "full"), ("off", "full")]
    results = _run_eval(
        Path(args.data),
        args.provider,
        configs,
        replays,
        Path(args.model),
        limit=args.limit,
        seed=args.seed,
        noise=args.noise,
    )
    results["command"] = "eval"
    results["classifier"] = _classifier_report(Path(args.model))
    results["rag"] = _rag_eval(get_settings().knowledge_dir)
    _print_headline(results)
    md, js = write_reports(results, Path(args.report_dir))
    _say(f"wrote {md} and {js}")
    if not args.no_track:
        _track(results, "eval")
    return 0


def _track(results: dict[str, Any], command: str) -> None:
    """Append the run to the dashboard history (SQLite).

    The dashboard follows the ``full`` preset, so a run that did not evaluate it
    is reported but not recorded.
    """
    from docai.eval.tracking import run_record
    from docai.storage import RunRepository

    if "full" not in results.get("configs", {}):
        _say("run not recorded: the dashboard tracks the 'full' preset, which this run skipped")
        return
    db_path = get_settings().db_path
    run_id = RunRepository(db_path).add(run_record(results, command))
    _say(f"recorded run {run_id} in {db_path} (see /dashboard)")


def cmd_replay(args: argparse.Namespace) -> int:
    """Print the blast radius between two guardrail configurations."""
    results = _run_eval(
        Path(args.data),
        args.provider,
        [args.a, args.b],
        [(args.a, args.b)],
        Path(args.model),
        limit=args.limit,
    )
    replay = results["replays"][0]
    _say(f"\nblast radius {args.a} → {args.b} over {replay['fields_compared']} fields")
    _say(json.dumps({"totals": replay["totals"], "documents": replay["documents"]}, indent=2))
    _say(f"{'field':22s} {'changed':>8s} {'fixed':>6s} {'withheld':>9s} {'regressed':>10s}")
    for name, v in replay["per_field"].items():
        _say(
            f"{name:22s} {v.get('changed', 0):8d} {v.get('fixed', 0):6d} "
            f"{v.get('withheld', 0):9d} {v.get('regressed', 0):10d}"
        )
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    """Answer a question from the fiscal-rules notes."""
    from docai.extract.providers import detect_real_provider, get_provider
    from docai.rag import KnowledgeBase

    kb = KnowledgeBase.from_dir(get_settings().knowledge_dir)
    provider = None
    if args.llm:
        settings = get_settings()
        name = detect_real_provider(settings)
        if name is None:
            _say("(no real provider credentials found; answering extractively)")
        else:
            provider = get_provider(name, settings)
    answer = kb.ask(args.question, k=args.k, provider=provider)
    _say(f"\n{answer.answer}\n")
    for p in answer.passages:
        _say(f"[{p.rank}] {p.title}  ({p.citation}, score {p.score})")
    return 0


def cmd_rag_eval(args: argparse.Namespace) -> int:
    """Print retrieval hit@k and MRR on the QA set."""
    result = _rag_eval(get_settings().knowledge_dir)
    _say(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    return 0


def _demo_corpus(args: argparse.Namespace, data_dir: Path) -> None:
    """Generate the corpus unless an identical one (n, seed, generator version) exists."""
    from docai.synth import generate_corpus
    from docai.synth.corpus import GENERATOR_VERSION

    manifest = data_dir / "manifest.json"
    current = json.loads(manifest.read_text()) if manifest.exists() else {}
    wanted = (args.n, args.seed, GENERATOR_VERSION)
    if (
        not args.fresh
        and (current.get("n"), current.get("seed"), current.get("generator")) == wanted
    ):
        _say(f"[1/5] reusing synthetic corpus in {data_dir}")
        return
    _say(f"[1/5] generating {args.n} synthetic documents (seed {args.seed}) → {data_dir}")
    for sub in ("pdf", "truth", "text", "xml"):
        shutil.rmtree(data_dir / sub, ignore_errors=True)
    generate_corpus(args.n, data_dir, seed=args.seed)


def cmd_demo(args: argparse.Namespace) -> int:
    """Run everything offline: synth, train, evaluate, report, record."""
    from docai.classify.train import TrainConfig, save_classifier, train_classifier
    from docai.eval.report import results_table, update_readme, write_reports

    data_dir, model_path = Path(args.data), Path(args.model)
    timings: dict[str, float] = {}
    started = clock = time.perf_counter()

    def lap(name: str) -> None:
        nonlocal clock
        timings[name] = round(time.perf_counter() - clock, 2)
        clock = time.perf_counter()

    _demo_corpus(args, data_dir)
    lap("synth")
    _say("[2/5] extracting PDF text layers")
    texts, labels = _load_texts(data_dir)
    lap("pdf_text")
    _say("[3/5] training the PyTorch document-type classifier")
    model, report = train_classifier(texts, labels, TrainConfig(), log=_say)
    save_classifier(model_path, model, report)
    lap("train_classifier")
    _say("[4/5] extraction with the simulated LLM + guardrail evaluation and replay")
    replays = [("text", "full"), ("dv-only", "full"), ("off", "full")]
    results = _run_eval(data_dir, "simulated", ALL_CONFIGS, replays, model_path)
    lap("extract_and_eval")
    _say("[5/5] RAG retrieval evaluation")
    results["rag"] = _rag_eval(get_settings().knowledge_dir)
    lap("rag")
    timings["total"] = round(time.perf_counter() - started, 1)

    results |= {
        "command": "demo",
        "classifier": {k: v for k, v in report.items() if k != "history"},
        "timings": timings,
    }
    md, js = write_reports(results, Path(args.report_dir))
    _print_headline(results)
    _say(f"wrote {md} and {js} in {timings['total']}s")
    if not args.no_track:
        _track(results, "demo")
    if args.update_readme and update_readme(
        get_settings().home / "README.md", results_table(results)
    ):
        _say("updated the results table in README.md")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the FastAPI app with uvicorn."""
    import uvicorn

    uvicorn.run(
        "docai.api.app:app",
        host=args.host,
        port=args.port,
        reload=False,
        server_header=False,
        access_log=False,  # the app logs one line per request, with its request id
    )
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    """Run the MCP server over stdio."""
    from docai.mcp_server import main as mcp_main

    mcp_main()
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    """Enqueue local PDFs for human review, or list the queue (local, trusted use)."""
    import hashlib

    from docai import container
    from docai.review import reviewer_actor

    service = container.get_review_service()
    if args.action == "list":
        for item in service.repository.list(None if args.status == "all" else args.status):
            pending = ", ".join(item.pending_fields) or "-"
            _say(f"{item.id}  {item.status:8s} {item.doc_type:5s} {item.filename}  [{pending}]")
        return 0
    pipeline = container.get_pipeline()
    for path in map(Path, args.files):
        data = path.read_bytes()
        xml_path = path.parent.parent / "xml" / f"{path.stem}.xml"
        xml = xml_path.read_bytes() if args.with_xml and xml_path.exists() else None
        result = pipeline.run_pdf(data, filename=path.name, xml_data=xml)
        item_id, created = service.enqueue(
            result,
            text=result.text,
            document_sha256=hashlib.sha256(data).hexdigest(),
            actor=reviewer_actor("cli"),
        )
        if result.guardrails is None:
            verdict = f"not extracted ({result.prediction.label})"
        elif item_id is None:
            verdict = "straight-through, nothing to review"
        else:
            verdict = "queued" if created else "already queued"
        _say(f"{path.name}: {verdict}{f' as {item_id}' if item_id else ''}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate a CNPJ or a 44-digit access key."""
    from docai.tools import validate_access_key, validate_cnpj

    value = args.value
    result = (
        validate_access_key(value)
        if len("".join(c for c in value if c.isdigit())) > 14
        else validate_cnpj(value)
    )
    _say(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


# ------------------------------------------------------------------ parser
if TYPE_CHECKING:
    Subparsers = argparse._SubParsersAction[argparse.ArgumentParser]


def _add_data_commands(sub: Subparsers, settings: Settings) -> None:
    data_default, model_default = str(settings.data_dir), str(settings.model_path)
    p = sub.add_parser("synth", help="generate synthetic PDFs + ground truth")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--out", default=data_default)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prefix", default="doc")
    p.add_argument("--types", help="comma-separated explicit list of types (overrides --n)")
    p.add_argument("--xml-share", type=float, default=0.6, help="NF-e/NFS-e that ship an XML")
    p.add_argument("--fresh", action="store_true", help="delete previous pdf/truth/text first")
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("train", help="train the document-type classifier")
    p.add_argument("--data", default=data_default)
    p.add_argument("--out", default=model_default)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--seed", type=int, default=13)
    p.set_defaults(func=cmd_train)


def _add_eval_commands(sub: Subparsers, settings: Settings) -> None:
    data_default, model_default = str(settings.data_dir), str(settings.model_path)
    report_default = str(settings.reports_dir)
    for name, func, help_text in (
        ("eval", cmd_eval, "evaluate extraction + guardrails and write reports"),
        ("replay", cmd_replay, "blast-radius diff between two guardrail configs"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--data", default=data_default)
        p.add_argument("--model", default=model_default)
        p.add_argument(
            "--provider",
            default="simulated",
            choices=["simulated", "anthropic", "gemini", "bedrock", "auto"],
        )
        p.add_argument("--limit", type=int, default=None)
        p.set_defaults(func=func)
    eval_parser = sub.choices["eval"]
    eval_parser.add_argument("--configs", default=",".join(ALL_CONFIGS))
    eval_parser.add_argument("--report-dir", default=report_default)
    eval_parser.add_argument("--seed", type=int, default=7, help="simulated-LLM seed")
    eval_parser.add_argument(
        "--noise", type=float, default=1.0, help="simulated error-rate multiplier"
    )
    eval_parser.add_argument("--no-track", action="store_true", help="do not record the run")
    replay_parser = sub.choices["replay"]
    replay_parser.add_argument("--a", default="text")
    replay_parser.add_argument("--b", default="full")

    p = sub.add_parser("ask", help="ask the fiscal-rules knowledge base")
    p.add_argument("question")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--llm", action="store_true", help="synthesize with a real provider if set")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("rag-eval", help="retrieval hit@k on the QA set")
    p.set_defaults(func=cmd_rag_eval)

    p = sub.add_parser("demo", help="run everything offline and write the reports")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data", default=data_default)
    p.add_argument("--model", default=model_default)
    p.add_argument("--report-dir", default=report_default)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--update-readme", action="store_true")
    p.add_argument("--no-track", action="store_true", help="do not record the run")
    p.set_defaults(func=cmd_demo)


def _add_service_commands(sub: Subparsers) -> None:
    p = sub.add_parser("serve", help="run the FastAPI app")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("mcp", help="run the MCP server over stdio")
    p.set_defaults(func=cmd_mcp)

    p = sub.add_parser("review", help="human review queue (local): enqueue PDFs or list items")
    p.add_argument("action", choices=["enqueue", "list"])
    p.add_argument("files", nargs="*", help="PDF files to enqueue")
    p.add_argument("--with-xml", action="store_true", help="use ../xml/<name>.xml when present")
    p.add_argument("--status", default="pending", choices=["pending", "approved", "all"])
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("validate", help="validate a CNPJ or an access key")
    p.add_argument("value")
    p.set_defaults(func=cmd_validate)


def build_parser() -> argparse.ArgumentParser:
    """The ``docai`` argument parser."""
    parser = argparse.ArgumentParser(prog="docai", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    settings = get_settings()
    _add_data_commands(sub, settings)
    _add_eval_commands(sub, settings)
    _add_service_commands(sub)
    return parser


def _check_settings() -> None:
    """Exit with one readable line on bad ``DOCAI_*`` values, never echoing them (secrets)."""
    from pydantic import ValidationError

    try:
        get_settings()
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'settings'}: {error['msg']}"
            for error in exc.errors()
        )
        raise SystemExit(f"invalid configuration: {problems}") from None


def main(argv: list[str] | None = None) -> int:
    """Entry point of the ``docai`` command."""
    _check_settings()
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
