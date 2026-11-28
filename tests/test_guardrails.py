import copy
import random

import pytest

from docai.domain.access_key import build_access_key, format_access_key, parse_access_key
from docai.domain.cnpj import format_cnpj, is_valid_cnpj
from docai.domain.normalize import normalize_value
from docai.domain.schemas import FIELD_KINDS
from docai.extract.providers.simulated import ErrorRates, inject_errors
from docai.guardrails import Action, run_guardrails
from docai.guardrails.checks import FROM_KEY, is_printed
from docai.guardrails.text_index import DocumentText
from docai.synth.cnpj import corrupt_cnpj, generate_cnpj


def _doc(by_type, doc_type, i=0):
    doc = by_type[doc_type][i]
    return doc, copy.deepcopy(doc["fields"])


def _actions(result, field):
    return [(f.action, f.check) for f in result.findings if f.field == field]


# ----------------------------------------------------------------- no false alarms
def test_clean_extractions_pass_untouched(corpus, texts):
    for doc in corpus:
        if not doc["fields"]:
            continue
        result = run_guardrails(doc["doc_type"], doc["fields"], texts[doc["doc_id"]], "full")
        assert result.findings == [], (doc["doc_id"], [f.to_dict() for f in result.findings])
        assert not result.needs_review
        assert result.record == doc["fields"]


def test_representation_noise_is_normalized_not_flagged(by_type, texts):
    doc, fields = _doc(by_type, "nfe")
    fields["issuer_cnpj"] = format_cnpj(fields["issuer_cnpj"])
    fields["number"] = fields["number"].zfill(9)
    fields["issue_date"] = "/".join(reversed(fields["issue_date"].split("-")))
    fields["total_value"] = (
        f"R$ {fields['total_value']:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    )
    result = run_guardrails("nfe", fields, texts[doc["doc_id"]], "full")
    assert result.findings == []
    assert result.record == doc["fields"]


# ----------------------------------------------------------------- schema
@pytest.mark.parametrize(
    ("kind", "value", "expected"),
    [
        ("money", "1.234,56", 1234.56),
        ("money", "1,234.56", 1234.56),
        ("money", 99, 99.0),
        ("date", "26/03/2026", "2026-03-26"),
        ("doc_number", "000.012.345", "12345"),
        ("doc_number", 12345.0, "12345"),
        ("cnpj", "11.222.333/0001-81", "11222333000181"),
        ("cfop", "6.102", "6102"),
        ("access_key", "3525 0611 2223 3300 0181 5500 1000 0123 4512 3456 7811", None),
    ],
)
def test_normalize(kind, value, expected):
    if expected is None:
        assert len(normalize_value(kind, value)) == 44
    else:
        assert normalize_value(kind, value) == expected


@pytest.mark.parametrize(
    ("kind", "value"),
    [
        ("cnpj", "123"),
        ("money", "-5"),
        ("date", "31/02/2026"),
        ("cfop", "9102"),
        ("access_key", "1234"),
        ("doc_number", "0"),
        ("rate", 150),
    ],
)
def test_normalize_rejects(kind, value):
    with pytest.raises(ValueError):
        normalize_value(kind, value)


def test_schema_nulls_impossible_values(by_type, texts):
    doc, fields = _doc(by_type, "nfse")
    fields["recipient_cnpj"] = "123"
    result = run_guardrails("nfse", fields, texts[doc["doc_id"]], "text")
    assert _actions(result, "recipient_cnpj") == [(Action.NULLED, "schema")]
    assert result.needs_review
    # v3: the role anchor re-reads it from the TOMADOR block
    full = run_guardrails("nfse", fields, texts[doc["doc_id"]], "full")
    assert _actions(full, "recipient_cnpj") == [
        (Action.NULLED, "schema"),
        (Action.CORRECTED, "role_anchor"),
    ]
    assert full.record["recipient_cnpj"] == doc["fields"]["recipient_cnpj"]


# ----------------------------------------------------------------- CNPJ
def test_cnpj_with_bad_check_digits_is_nulled(by_type, texts):
    doc, fields = _doc(by_type, "nfe")
    rng = random.Random(3)
    wrong = fields["recipient_cnpj"]
    while is_valid_cnpj(wrong):
        wrong = corrupt_cnpj(fields["recipient_cnpj"], rng)
    fields["recipient_cnpj"] = wrong
    result = run_guardrails("nfe", fields, texts[doc["doc_id"]], "text")
    assert _actions(result, "recipient_cnpj") == [(Action.NULLED, "cnpj_dv")]
    assert result.record["recipient_cnpj"] is None


def test_valid_but_absent_cnpj_is_nulled(by_type, texts):
    doc, fields = _doc(by_type, "cte")
    fields["sender_cnpj"] = generate_cnpj(random.Random(5))  # valid DV, not in this document
    result = run_guardrails("cte", fields, texts[doc["doc_id"]], "text")
    assert _actions(result, "sender_cnpj") == [(Action.NULLED, "presence")]


# ----------------------------------------------------------------- access key
def test_key_typo_is_reread_from_the_document(by_type, texts):
    doc, fields = _doc(by_type, "nfe")
    key = fields["access_key"]
    fields["access_key"] = key[:10] + str((int(key[10]) + 1) % 10) + key[11:]
    result = run_guardrails("nfe", fields, texts[doc["doc_id"]], "full")
    assert _actions(result, "access_key") == [(Action.CORRECTED, "access_key")]
    assert result.record["access_key"] == key
    assert not result.needs_review


def test_cte_key_confused_with_referenced_nfe_key(by_type, texts):
    doc, fields = _doc(by_type, "cte")
    fields["access_key"] = doc["layout"]["referenced_keys"][0]  # valid, printed, but model 55
    result = run_guardrails("cte", fields, texts[doc["doc_id"]], "full")
    finding = result.findings[0]
    assert (finding.action, finding.check) == (Action.CORRECTED, "access_key")
    assert "model 55" in finding.reason
    assert result.record["access_key"] == doc["fields"]["access_key"]


def test_key_is_nulled_when_it_cannot_be_reread(by_type, texts):
    doc, fields = _doc(by_type, "nfe")
    key = fields["access_key"]
    fields["access_key"] = key[:-1] + str((int(key[-1]) + 1) % 10)
    text = texts[doc["doc_id"]].replace(format_access_key(key), "<illegible>")  # key not readable
    result = run_guardrails("nfe", fields, text, "full")
    assert result.record["access_key"] is None
    assert result.status("access_key") == Action.NULLED


# ----------------------------------------------------------------- key cross-checks
def test_hallucinated_issuer_cnpj_is_corrected_from_the_key(by_type, texts):
    doc, fields = _doc(by_type, "nfe")
    other = by_type["nfe"][1]["fields"]
    fields["issuer_cnpj"], fields["issuer_name"] = other["issuer_cnpj"], other["issuer_name"]
    result = run_guardrails("nfe", fields, texts[doc["doc_id"]], "full")
    assert _actions(result, "issuer_cnpj") == [(Action.CORRECTED, "key_cross")]
    assert result.record["issuer_cnpj"] == doc["fields"]["issuer_cnpj"]
    # the name cannot be derived from the key: it is nulled, never guessed
    assert result.record["issuer_name"] is None
    assert result.needs_review


def test_number_from_filename_is_corrected_and_dropped_series_recovered(by_type, texts):
    doc, fields = _doc(by_type, "cte")
    fields["number"], fields["series"] = "48213", None
    result = run_guardrails("cte", fields, texts[doc["doc_id"]], "full")
    assert result.record["number"] == doc["fields"]["number"]
    assert result.record["series"] == doc["fields"]["series"]
    assert {f.field for f in result.findings} == {"number", "series"}
    assert all(f.action == Action.CORRECTED for f in result.findings)
    assert result.provenance["number"] == FROM_KEY


def test_issue_month_must_match_the_key(by_type, texts):
    doc, fields = _doc(by_type, "nfe")
    year, month, day = fields["issue_date"].split("-")
    fields["issue_date"] = f"{year}-{int(month) % 12 + 1:02d}-{day if int(day) <= 28 else '28'}"
    result = run_guardrails("nfe", fields, texts[doc["doc_id"]], "full")
    assert _actions(result, "issue_date") == [(Action.NULLED, "key_cross")]


# ----------------------------------------------------------------- presence
def test_wrong_decimal_on_nfe_total_is_nulled(by_type, texts):
    doc, fields = _doc(by_type, "nfe")
    fields["total_value"] = round(fields["total_value"] * 100, 2)
    result = run_guardrails("nfe", fields, texts[doc["doc_id"]], "full")
    assert _actions(result, "total_value") == [(Action.NULLED, "presence")]


def test_nfse_number_from_filename_is_nulled(by_type, texts):
    doc, fields = _doc(by_type, "nfse")
    fields["number"] = "73519"
    result = run_guardrails("nfse", fields, texts[doc["doc_id"]], "full")
    assert result.record["number"] is None
    assert result.status("number") == Action.NULLED


# ----------------------------------------------------------------- consistency
def test_cte_component_label_swap_is_corrected(by_type, texts):
    doc, fields = _doc(by_type, "cte")
    comps = fields["freight_components"]
    comps[0]["value"], comps[1]["value"] = comps[1]["value"], comps[0]["value"]
    result = run_guardrails("cte", fields, texts[doc["doc_id"]], "full")
    assert _actions(result, "freight_components") == [(Action.CORRECTED, "cte_components")]
    assert result.record["freight_components"] == doc["fields"]["freight_components"]


def test_cte_wrong_total_recovered_from_printed_components(by_type, texts):
    doc, fields = _doc(by_type, "cte")
    fields["total_value"] = round(fields["total_value"] / 10, 2)
    result = run_guardrails("cte", fields, texts[doc["doc_id"]], "full")
    assert [f.action for f in result.findings if f.field == "total_value"] == [
        Action.NULLED,  # presence: not printed
        Action.CORRECTED,  # components re-read and their sum is printed
    ]
    assert result.record["total_value"] == doc["fields"]["total_value"]
    assert not result.needs_review


def test_nfse_iss_math_is_flagged(by_type, texts):
    doc, fields = _doc(by_type, "nfse")
    fields["iss_rate"] = 5.0 if fields["iss_rate"] != 5.0 else 2.0
    text = texts[doc["doc_id"]] + "\n" + f"{fields['iss_rate']:.2f}".replace(".", ",")
    result = run_guardrails("nfse", fields, text, "full")
    assert (Action.FLAGGED, "nfse_iss") in _actions(result, "iss_value")


def test_same_cnpj_in_two_roles_is_flagged(by_type, texts):
    doc, fields = _doc(by_type, "nfse")
    fields["recipient_cnpj"] = fields["issuer_cnpj"]
    fields["recipient_name"] = fields["issuer_name"]
    result = run_guardrails("nfse", fields, texts[doc["doc_id"]], "text")
    assert (Action.FLAGGED, "parties") in _actions(result, "recipient_cnpj")


def test_name_bound_to_another_cnpj_is_flagged(by_type, texts):
    doc, fields = _doc(by_type, "cte")
    fields["issuer_name"] = fields["sender_name"]  # printed, but in the sender block
    result = run_guardrails("cte", fields, texts[doc["doc_id"]], "full")
    assert (Action.FLAGGED, "parties") in _actions(result, "issuer_name")
    assert result.needs_review


def test_dropped_field_goes_to_review(by_type, texts):
    doc, fields = _doc(by_type, "nfse")
    fields["verification_code"] = None
    result = run_guardrails("nfse", fields, texts[doc["doc_id"]], "full")
    assert _actions(result, "verification_code") == [(Action.FLAGGED, "required")]


# ----------------------------------------------------------------- presets
def test_presets_differ_as_documented(by_type, texts):
    doc, fields = _doc(by_type, "nfe")
    fields["number"] = "48213"  # valid shape, wrong value
    off = run_guardrails("nfe", fields, texts[doc["doc_id"]], "off")
    v1 = run_guardrails("nfe", fields, texts[doc["doc_id"]], "dv-only")
    v2 = run_guardrails("nfe", fields, texts[doc["doc_id"]], "full")
    assert off.findings == [] and v1.findings == []  # neither can see this error
    assert v2.record["number"] == doc["fields"]["number"]
    with pytest.raises(ValueError):
        run_guardrails("nfe", fields, "", "nope")


def test_input_is_never_mutated(by_type, texts):
    doc, fields = _doc(by_type, "cte")
    fields["issuer_cnpj"] = "123"
    snapshot = copy.deepcopy(fields)
    run_guardrails("cte", fields, texts[doc["doc_id"]], "full")
    assert fields == snapshot


# ----------------------------------------------------------------- never invent
@pytest.mark.parametrize("seed", range(12))
def test_never_invent_and_never_break_a_correct_value(corpus, texts, seed):
    """Under heavy random corruption, every published value is printed in the document
    or derived from the verified access key, and no correct value is ever changed."""
    rates = ErrorRates().scaled(6)
    for doc in corpus:
        if not doc["fields"]:
            continue
        text, truth = texts[doc["doc_id"]], doc["fields"]
        noisy, _ = inject_errors(
            truth,
            doc["doc_type"],
            random.Random(f"{seed}-{doc['doc_id']}"),
            rates,
            filename=doc["filename"],
            party_pool=[
                {"name": "Fictícia Comércio Ltda", "cnpj": generate_cnpj(random.Random(seed))}
            ],
            layout=doc["layout"],
        )
        result = run_guardrails(doc["doc_type"], noisy, text, "full")
        index = DocumentText(text)
        key = (
            parse_access_key(result.record["access_key"])
            if result.record.get("access_key")
            else None
        )
        for name, value in result.record.items():
            if value is None:
                continue
            kind = FIELD_KINDS[name]
            if kind == "access_key":
                assert index.contains_key(value)
            elif kind in ("components", "services"):
                assert all(index.contains_money(c["value"]) for c in value)
            elif result.provenance.get(name) == FROM_KEY:
                assert key is not None
            else:
                assert is_printed(index, kind, value), (doc["doc_id"], name, value)
            if noisy.get(name) is not None and normalize_value(kind, noisy[name]) == truth[name]:
                assert value == truth[name], ("false correction", doc["doc_id"], name)


def test_corrected_values_are_never_new(corpus, texts):
    """A correction always lands on the ground truth in this corpus (it is re-read, not guessed)."""
    for doc in corpus:
        if doc["doc_type"] not in ("nfe", "cte"):
            continue
        fields = copy.deepcopy(doc["fields"])
        fields["issuer_cnpj"] = generate_cnpj(random.Random(1))
        fields["number"] = "1"
        result = run_guardrails(doc["doc_type"], fields, texts[doc["doc_id"]], "full")
        for finding in result.findings:
            if finding.action == Action.CORRECTED:
                assert finding.after == doc["fields"][finding.field]


def test_build_access_key_for_unknown_layout_is_not_reread():
    key = build_access_key(
        cuf="35", year=2025, month=1, cnpj="11222333000181", model="55", series=1, number=10, code=1
    )
    text = f"A: {key}\nB: {key[:-1]}{(int(key[-1]) + 5) % 10}"
    fields = {"access_key": "0" * 44}
    result = run_guardrails("nfe", fields, text, "full")
    assert result.record["access_key"] == key  # exactly one *valid* key is printed
