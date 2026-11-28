"""XML-first extraction: safe parsing, the XML-first policy and the cross-check guardrail."""

import pytest

from docai.classify import KeywordClassifier
from docai.domain.findings import Action, Provenance
from docai.eval import evaluate, extract_corpus, load_corpus, summarize, xml_comparison
from docai.extract.providers import ErrorRates, ProviderError, SimulatedLLMProvider, TruthStore
from docai.extract.xml_reader import FiscalXml, XmlRejected, read_fiscal_xml
from docai.guardrails import run_guardrails
from docai.pipeline import Pipeline


@pytest.fixture(scope="module")
def xml_docs(corpus, corpus_dir):
    return [
        (d, (corpus_dir / "xml" / d["xml_filename"]).read_bytes())
        for d in corpus
        if d["xml_filename"]
    ]


class NoCallProvider(SimulatedLLMProvider):
    """Fails the test if the LLM is called."""

    def complete(self, *args, **kwargs):
        raise AssertionError("the LLM must not be called when the XML is complete")


def test_corpus_ships_xml_for_part_of_nfe_and_nfse(corpus, xml_docs):
    assert xml_docs
    assert {d["doc_type"] for d, _ in xml_docs} <= {"nfe", "nfse"}
    assert any(d["xml_filename"] is None for d in corpus if d["doc_type"] == "nfe")


def test_xml_reader_matches_the_ground_truth(xml_docs):
    for doc, data in xml_docs:
        xml = read_fiscal_xml(data)
        assert xml.doc_type == doc["doc_type"] and xml.problems == []
        covered = {k: v for k, v in xml.fields.items() if v is not None}
        assert covered == {k: doc["fields"][k] for k in covered}, doc["doc_id"]
        # NF-e XMLs are complete; the NFS-e services list lives in free text, not in tags
        assert xml.missing == ([] if doc["doc_type"] == "nfe" else ["services"])


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            b'<?xml version="1.0"?><!DOCTYPE r [<!ENTITY x SYSTEM "file:///etc/passwd">]>'
            b"<nfeProc>&x;</nfeProc>",
            "DTD or entities",
        ),
        (
            b'<!DOCTYPE l [<!ENTITY a "aaaa"><!ENTITY b "&a;&a;&a;&a;">]><CompNfse>&b;</CompNfse>',
            "DTD or entities",
        ),
        (b"<nfeProc><unclosed></nfeProc>", "not well-formed"),
        (b"<invoice/>", "not an NF-e"),
        (b"<nfeProc/>", "without infNFe"),
        (b"<CompNfse/>", "without InfNfse"),
    ],
)
def test_unsafe_or_unsupported_xml_is_rejected(payload, message):
    with pytest.raises(XmlRejected, match=message):
        read_fiscal_xml(payload)


def test_oversized_xml_is_rejected(xml_docs):
    _, data = xml_docs[0]
    with pytest.raises(XmlRejected, match="larger than"):
        read_fiscal_xml(data, max_bytes=100)


def test_inconsistent_xml_is_reported(xml_docs):
    doc, data = next((d, x) for d, x in xml_docs if d["doc_type"] == "nfe")
    key = doc["fields"]["access_key"]
    broken = data.replace(f"NFe{key}".encode(), f"NFe{key[:-1]}{(int(key[-1]) + 1) % 10}".encode())
    xml = read_fiscal_xml(broken)
    assert any("check digit" in p for p in xml.problems)
    assert any("chNFe differ" in p for p in xml.problems)


def _pipeline(corpus_dir, provider_cls=SimulatedLLMProvider, rates=None, **kwargs):
    provider = provider_cls(TruthStore([corpus_dir]), rates or ErrorRates.none())
    return Pipeline(provider, KeywordClassifier(), "full", **kwargs)


def test_xml_first_skips_the_llm(corpus_dir, xml_docs, texts):
    pipeline = _pipeline(corpus_dir, NoCallProvider)
    for doc, data in xml_docs:
        if doc["doc_type"] != "nfe":
            continue
        result = pipeline.run_text(texts[doc["doc_id"]], xml=read_fiscal_xml(data))
        g = result.guardrails
        assert result.xml_status == "used" and g.source == "xml" and result.usage is None
        assert g.record == doc["fields"] and not g.needs_review
        assert set(g.provenance.values()) == {Provenance.XML}


def test_partial_xml_lets_the_llm_fill_only_what_it_lacks(corpus_dir, xml_docs, texts):
    pipeline = _pipeline(corpus_dir, rates=ErrorRates().scaled(4))
    for doc, data in xml_docs:
        if doc["doc_type"] != "nfse":
            continue
        result = pipeline.run_text(
            texts[doc["doc_id"]], doc_id=doc["doc_id"], xml=read_fiscal_xml(data)
        )
        g = result.guardrails
        assert result.xml_status == "cross_checked" and result.usage is not None
        assert g.provenance["total_value"] == Provenance.XML
        assert g.provenance["services"] != Provenance.XML
        assert g.record == doc["fields"], doc["doc_id"]


def test_cross_check_corrects_the_llm_and_records_each_disagreement(corpus_dir, xml_docs, texts):
    pipeline = _pipeline(corpus_dir, rates=ErrorRates().scaled(4), xml_first=False)
    corrections = 0
    for doc, data in xml_docs:
        result = pipeline.run_text(
            texts[doc["doc_id"]], doc_id=doc["doc_id"], xml=read_fiscal_xml(data)
        )
        g = result.guardrails
        assert result.xml_status == "cross_checked" and g.source == "llm+xml"
        assert g.record == doc["fields"], doc["doc_id"]
        corrections += sum(f.check == "xml_cross" for f in g.findings)
        for finding in g.findings:
            if finding.check == "xml_cross":
                assert finding.action == Action.CORRECTED and finding.after is not None
    assert corrections > 0


def test_incomplete_xml_lets_the_llm_fill_the_gaps(corpus_dir, xml_docs, texts):
    doc, data = next((d, x) for d, x in xml_docs if d["doc_type"] == "nfe")
    full = read_fiscal_xml(data)
    partial = FiscalXml(full.doc_type, {**full.fields, "issue_date": None})
    result = _pipeline(corpus_dir).run_text(texts[doc["doc_id"]], doc_id=doc["doc_id"], xml=partial)
    assert partial.missing == ["issue_date"]
    assert result.usage is not None and result.guardrails.source == "llm+xml"
    assert result.guardrails.record == doc["fields"]
    assert result.guardrails.provenance["issue_date"] != Provenance.XML


def test_xml_of_another_type_or_broken_is_ignored_and_reviewed(
    corpus_dir, xml_docs, texts, by_type
):
    doc, data = next((d, x) for d, x in xml_docs if d["doc_type"] == "nfse")
    cte = by_type["cte"][0]
    result = _pipeline(corpus_dir).run_text(
        texts[cte["doc_id"]], doc_id=cte["doc_id"], xml=read_fiscal_xml(data)
    )
    assert result.xml_status == "type_mismatch" and result.guardrails.needs_review
    assert "ignored" in result.guardrails.review_reasons[0]
    broken = FiscalXml("nfse", read_fiscal_xml(data).fields, ["bad"])
    result = _pipeline(corpus_dir).run_text(texts[doc["doc_id"]], doc_id=doc["doc_id"], xml=broken)
    assert result.xml_status == "inconsistent" and result.guardrails.needs_review


def test_xml_values_must_be_printed_on_the_pdf(xml_docs, texts):
    doc, data = xml_docs[0]
    fields = dict(read_fiscal_xml(data).fields)
    fields["total_value"] = round(fields["total_value"] + 1, 2)  # printout and XML disagree
    result = run_guardrails(doc["doc_type"], fields, texts[doc["doc_id"]], xml_fields=fields)
    assert result.status("total_value") == Action.FLAGGED
    assert any(f.check == "xml_printed" for f in result.findings)


def test_pipeline_reads_untrusted_xml_bytes(corpus_dir, xml_docs):
    doc, data = next((d, x) for d, x in xml_docs if d["doc_type"] == "nfe")
    pdf = (corpus_dir / "pdf" / doc["filename"]).read_bytes()
    pipeline = _pipeline(corpus_dir, NoCallProvider)
    assert pipeline.run_pdf(pdf, xml_data=data).guardrails.record == doc["fields"]
    with pytest.raises(XmlRejected):
        pipeline.run_pdf(pdf, xml_data=b"<!DOCTYPE x []><x/>")


def test_xml_comparison_in_the_eval(corpus_dir):
    provider = SimulatedLLMProvider(TruthStore([corpus_dir]), ErrorRates().scaled(3), seed=5)
    records = extract_corpus(load_corpus(corpus_dir), provider, KeywordClassifier())
    comparison = xml_comparison(records)
    rows = comparison["strategies"]
    n_nfse = sum(1 for r in records if r.xml is not None and r.doc.doc_type == "nfse")
    assert comparison["n_docs"] > 0 and rows["xml_first"]["llm_calls"] == n_nfse
    assert rows["xml_first"]["field_accuracy"] == 1.0
    assert rows["llm_xml_cross_check"]["field_accuracy"] == 1.0
    assert rows["llm_only"]["llm_calls"] == comparison["n_docs"]
    assert rows["llm_only"]["field_accuracy"] <= 1.0
    first = summarize(evaluate(records, "full", xml_mode="first"), "full")
    only = summarize(evaluate(records, "full"), "full")
    assert first["silent_error_docs"] <= only["silent_error_docs"]
    with pytest.raises(ValueError):
        evaluate(records, "full", xml_mode="nope")


def test_api_accepts_an_xml(api_env, corpus_dir, xml_docs):
    from fastapi.testclient import TestClient

    from docai.api.app import create_app

    client = TestClient(create_app())
    doc, data = next((d, x) for d, x in xml_docs if d["doc_type"] == "nfe")
    pdf = (corpus_dir / "pdf" / doc["filename"]).read_bytes()
    files = {
        "file": ("renamed.pdf", pdf, "application/pdf"),
        "xml": ("doc.xml", data, "application/xml"),
    }
    body = client.post("/extract", files=files, data={"noise": "4"}).json()
    assert body["source"] == "xml" and body["xml_status"] == "used"
    assert body["record"] == doc["fields"] and body["needs_review"] is False
    files["xml"] = ("x.xml", b"<!DOCTYPE a []><a/>", "application/xml")
    rejected = client.post("/extract", files=files)
    assert rejected.status_code == 422 and rejected.json()["detail"].startswith("XML rejected")


def test_provider_errors_still_surface_without_xml(corpus_dir, by_type, texts):
    class Failing(SimulatedLLMProvider):
        def complete(self, *args, **kwargs):
            raise ProviderError("down")

    doc = by_type["cte"][0]
    with pytest.raises(ProviderError):
        _pipeline(corpus_dir, Failing).run_text(texts[doc["doc_id"]])
