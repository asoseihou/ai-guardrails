"""Layout anchors, the company registry and the routing rule.

Each test injects one realistic failure (the simulator's error modes) and checks
that v2 (presence and consistency) misses or only nulls it while v3 (anchors)
re-reads the right value from the box that owns it.
"""

import copy
import random

import pytest

from docai.classify import Prediction
from docai.domain.compare import same_value
from docai.domain.findings import Action, Provenance
from docai.domain.layout import (
    looks_like_boleto,
    read_cte_due_date,
    read_cte_weight,
    read_labelled_references,
    read_nfse_services,
    read_referenced_keys,
    read_role_party,
    split_pages,
)
from docai.extract.heuristic import heuristic_extract
from docai.extract.prompts import DOC_TYPE_RULES, SYSTEM_PROMPT, build_user_prompt
from docai.extract.providers.simulated import ErrorRates, inject_errors
from docai.guardrails import run_guardrails
from docai.pipeline.core import route


def _first(by_type, doc_type, predicate=lambda d: True):
    doc = next(d for d in by_type[doc_type] if predicate(d))
    return doc, copy.deepcopy(doc["fields"])


def _with_attachment(doc):
    return doc["layout"].get("attachment") is not None


def _checks(result, field):
    return [(f.action, f.check) for f in result.findings if f.field == field]


# ---------------------------------------------------------------- readers
def test_layout_readers_match_the_ground_truth(corpus, texts):
    for doc in corpus:
        text, f = texts[doc["doc_id"]], doc["fields"]
        if doc["doc_type"] == "cte":
            assert read_cte_weight(text) == f["gross_weight"]
            assert read_cte_due_date(text) == f["due_date"]
            assert read_referenced_keys(text) == f["referenced_keys"]
            assert read_role_party(text, "REMETENTE")["cnpj"] == f["sender_cnpj"]
            assert read_role_party(text, "DESTINATÁRIO")["name"] == f["recipient_name"]
        if doc["doc_type"] == "nfse":
            assert same_value("services", read_nfse_services(text), f["services"])
            assert read_role_party(text, "PRESTADOR DE SERVIÇOS")["cnpj"] == f["issuer_cnpj"]
        if doc["doc_type"] == "nfe":
            assert read_role_party(text, "DESTINATÁRIO / REMETENTE")["cnpj"] == f["recipient_cnpj"]
        if doc["doc_type"] in ("cte", "nfse"):
            expected = [f["service_order"]] if f["service_order"] else []
            assert read_labelled_references(text) == expected
            extracted = heuristic_extract(text, doc["doc_type"])
            assert same_value("service_order", extracted["service_order"], f["service_order"])
    assert read_role_party("no blocks here", "REMETENTE") is None
    assert read_cte_weight("") is None and read_nfse_services("") is None


def test_attachments_live_on_later_pages(by_type, texts):
    doc, _ = _first(by_type, "cte", _with_attachment)
    pages = split_pages(texts[doc["doc_id"]])
    attachment = doc["layout"]["attachment"]
    assert len(pages) == 2 and "ANEXO - DANFE" in pages[1]
    assert f"{attachment['gross_weight']:.3f}".replace(".", ",") in pages[1]


# ---------------------------------------------------------------- traps
def test_weight_and_due_date_from_the_attachment_are_reread(by_type, texts):
    doc, fields = _first(by_type, "cte", _with_attachment)
    attachment = doc["layout"]["attachment"]
    fields["gross_weight"], fields["due_date"] = attachment["gross_weight"], attachment["due_date"]
    text = texts[doc["doc_id"]]
    v2 = run_guardrails("cte", fields, text, "text")
    assert v2.record["gross_weight"] == attachment["gross_weight"]  # printed, so v2 accepts it
    v3 = run_guardrails("cte", fields, text, "full")
    assert _checks(v3, "gross_weight") == [(Action.CORRECTED, "cte_main_page")]
    assert v3.record["gross_weight"] == doc["fields"]["gross_weight"]
    assert v3.record["due_date"] == doc["fields"]["due_date"]
    assert not v3.needs_review


def test_payer_copied_into_recipient_is_reread_from_its_block(by_type, texts):
    doc, fields = _first(by_type, "cte", lambda d: d["layout"]["taker"] == "sender")
    fields["recipient_name"], fields["recipient_cnpj"] = (
        fields["sender_name"],
        fields["sender_cnpj"],
    )
    v2 = run_guardrails("cte", fields, texts[doc["doc_id"]], "text")
    assert v2.needs_review  # same CNPJ in two roles: flagged, not fixed
    v3 = run_guardrails("cte", fields, texts[doc["doc_id"]], "full")
    assert v3.record["recipient_cnpj"] == doc["fields"]["recipient_cnpj"]
    assert v3.record["recipient_name"] == doc["fields"]["recipient_name"]
    assert (Action.CORRECTED, "role_anchor") in _checks(v3, "recipient_cnpj")
    assert not v3.needs_review


def test_party_role_swap_is_no_longer_a_blind_spot(by_type, texts):
    doc, fields = _first(by_type, "nfse")
    for suffix in ("name", "cnpj"):
        fields[f"issuer_{suffix}"], fields[f"recipient_{suffix}"] = (
            fields[f"recipient_{suffix}"],
            fields[f"issuer_{suffix}"],
        )
    text = texts[doc["doc_id"]]
    v2 = run_guardrails("nfse", fields, text, "text")
    assert not v2.needs_review and v2.record != doc["fields"]  # silent error at v2
    v3 = run_guardrails("nfse", fields, text, "full")
    assert v3.record == doc["fields"] and not v3.needs_review


def test_reference_from_an_address_number(by_type, texts):
    street_number = lambda d: d["layout"]["issuer"]["street"].rsplit(" ", 1)[-1]  # noqa: E731
    doc, fields = _first(by_type, "nfse", lambda d: d["fields"]["service_order"])
    fields["service_order"] = street_number(doc)
    v3 = run_guardrails("nfse", fields, texts[doc["doc_id"]], "full")
    assert _checks(v3, "service_order") == [(Action.CORRECTED, "reference_label")]
    assert v3.record["service_order"] == doc["fields"]["service_order"]
    # no labelled reference printed: an invented one is nulled, never kept
    doc, fields = _first(by_type, "cte", lambda d: d["fields"]["service_order"])
    text = texts[doc["doc_id"]].replace(f"OS {fields['service_order']}", "-")
    fields["service_order"] = street_number(doc)
    result = run_guardrails("cte", fields, text, "full")
    assert _checks(result, "service_order") == [(Action.NULLED, "reference_label")]
    assert result.record["service_order"] is None


def test_referenced_key_typo_is_reread(by_type, texts):
    doc, fields = _first(by_type, "cte")
    key = fields["referenced_keys"][0]
    fields["referenced_keys"] = [
        key[:-1] + str((int(key[-1]) + 1) % 10),
        *fields["referenced_keys"][1:],
    ]
    v3 = run_guardrails("cte", fields, texts[doc["doc_id"]], "full")
    assert _checks(v3, "referenced_keys") == [(Action.CORRECTED, "referenced_keys")]
    assert v3.record["referenced_keys"] == doc["fields"]["referenced_keys"]
    unreadable = run_guardrails("cte", fields, "no keys printed", "full")
    assert unreadable.record["referenced_keys"] is None


def test_services_are_checked_against_the_table_and_the_total(by_type, texts):
    doc, fields = _first(by_type, "nfse")
    fields["services"] = [{**fields["services"][0], "value": fields["services"][0]["value"] + 50}]
    v3 = run_guardrails("nfse", fields, texts[doc["doc_id"]], "full")
    assert v3.record["services"] == doc["fields"]["services"]
    unreadable = texts[doc["doc_id"]].replace("SERVIÇOS QTD VALOR (R$)", "")
    flagged = run_guardrails("nfse", fields, unreadable, "full")  # only the sum check can act
    assert (Action.FLAGGED, "nfse_services") in _checks(flagged, "services")


def test_name_follows_the_verified_cnpj_with_a_registry(by_type, texts):
    doc, fields = _first(by_type, "nfe")
    registry = {fields["issuer_cnpj"]: fields["issuer_name"]}
    fields["issuer_name"] = "Vizinha Transportes Ltda"  # plausible neighbour, not printed
    text = texts[doc["doc_id"]]
    without = run_guardrails("nfe", fields, text, "full")
    assert without.status("issuer_name") == Action.NULLED  # presence: never published
    with_registry = run_guardrails("nfe", fields, text, "full", registry=registry)
    assert with_registry.status("issuer_name") == Action.CORRECTED
    assert with_registry.record["issuer_name"] == doc["fields"]["issuer_name"]
    assert with_registry.provenance["issuer_name"] == Provenance.DOCUMENT_TEXT
    unprinted = {fields["issuer_cnpj"]: "Nome Que Não Está Impresso Ltda"}
    assert (
        run_guardrails("nfe", fields, text, "full", registry=unprinted).record["issuer_name"]
        is None
    )


@pytest.mark.parametrize("seed", range(4))
def test_new_error_modes_never_survive_v3(corpus, texts, seed):
    rates = ErrorRates(**{k: (0.9 if k in NEW_MODES else 0.0) for k in ErrorRates().to_dict()})
    registry = {
        d["fields"]["issuer_cnpj"]: d["fields"]["issuer_name"] for d in corpus if d["fields"]
    }
    pool = [{"name": n, "cnpj": c} for c, n in registry.items()]
    for doc in corpus:
        if not doc["fields"]:
            continue
        noisy, injected = inject_errors(
            doc["fields"],
            doc["doc_type"],
            random.Random(f"{seed}{doc['doc_id']}"),
            rates,
            party_pool=pool,
            layout=doc["layout"],
        )
        result = run_guardrails(
            doc["doc_type"], noisy, texts[doc["doc_id"]], "full", registry=registry
        )
        for error in injected:
            name = error["field"]
            ok = same_value(name, result.record[name], doc["fields"][name])
            assert ok or result.needs_review, (doc["doc_id"], error)


NEW_MODES = {
    "name_from_neighbour",
    "reference_from_address",
    "weight_from_attachment",
    "due_date_from_duplicate",
    "payer_into_recipient",
    "referenced_key_typo",
}


# ---------------------------------------------------------------- routing and prompts
def test_a_boleto_is_never_routed_as_an_invoice(by_type, texts):
    boleto = texts[by_type["boleto"][0]["doc_id"]]
    assert looks_like_boleto(boleto)
    assert not any(looks_like_boleto(texts[d["doc_id"]]) for d in by_type["nfe"] + by_type["cte"])
    wrong = Prediction("nfe", 0.6, {"nfe": 0.6, "boleto": 0.4})
    assert route(wrong, boleto).label == "boleto"
    right = Prediction("nfe", 0.9, {"nfe": 0.9})
    assert route(right, texts[by_type["nfe"][0]["doc_id"]]) is right


def test_prompts_anchor_every_document_type():
    for doc_type, rules in DOC_TYPE_RULES.items():
        prompt = build_user_prompt("TEXT", doc_type, "a.pdf")
        assert all(rule in prompt for rule in rules)
    cte = " ".join(DOC_TYPE_RULES["cte"])
    assert ("TOMADOR" in cte and "attached DANFE" in cte) or "attachments" in cte
    assert "never complete" in SYSTEM_PROMPT and "DD/MM/YYYY" in SYSTEM_PROMPT
