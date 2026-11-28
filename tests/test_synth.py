import json
from datetime import date

from docai.domain.access_key import MODEL_BY_DOC_TYPE, format_access_key, parse_access_key
from docai.domain.cnpj import format_cnpj, is_valid_cnpj
from docai.domain.schemas import REQUIRED_FIELDS, field_names
from docai.synth import generate_corpus
from docai.synth.render import FOOTER, br_date, brl, dotted_number


def test_corpus_layout(corpus_dir, corpus):
    manifest = json.loads((corpus_dir / "manifest.json").read_text())
    assert manifest["n"] == len(corpus) == 24
    assert manifest["counts"] == {"boleto": 3, "cte": 6, "nfe": 6, "nfse": 6, "other": 3}
    for doc in corpus:
        assert (corpus_dir / "pdf" / doc["filename"]).read_bytes().startswith(b"%PDF")
        assert doc["filename"].startswith(doc["doc_id"] + "__")


def test_ground_truth_matches_schemas(by_type):
    for doc_type in ("nfe", "cte", "nfse"):
        for doc in by_type[doc_type]:
            assert list(doc["fields"]) == field_names(doc_type)
            required = REQUIRED_FIELDS[doc_type]
            assert all(doc["fields"][name] is not None for name in required)
    for doc_type in ("boleto", "other"):
        assert all(doc["fields"] == {} for doc in by_type[doc_type])


def test_access_keys_encode_the_document(by_type):
    for doc_type in ("nfe", "cte"):
        for doc in by_type[doc_type]:
            f = doc["fields"]
            key = parse_access_key(f["access_key"])
            issued = date.fromisoformat(f["issue_date"])
            assert key.model == MODEL_BY_DOC_TYPE[doc_type]
            assert key.cnpj == f["issuer_cnpj"]
            assert str(key.number) == f["number"] and str(key.series) == f["series"]
            assert (key.year, key.month) == (issued.year, issued.month)


def test_all_cnpjs_are_valid(corpus):
    for doc in corpus:
        for name, value in doc["fields"].items():
            if name.endswith("_cnpj"):
                assert is_valid_cnpj(value), (doc["doc_id"], name)


def test_amounts_are_consistent(by_type):
    for doc in by_type["cte"]:
        f = doc["fields"]
        assert round(sum(c["value"] for c in f["freight_components"]), 2) == f["total_value"]
        assert f["freight_components"][0]["name"] == "FRETE PESO"
    for doc in by_type["nfse"]:
        f = doc["fields"]
        assert abs(f["total_value"] * f["iss_rate"] / 100 - f["iss_value"]) <= 0.011


def test_pdf_text_contains_the_printed_values(by_type, texts):
    for doc_type in ("nfe", "cte", "nfse"):
        for doc in by_type[doc_type]:
            f, text = doc["fields"], texts[doc["doc_id"]]
            assert format_cnpj(f["issuer_cnpj"]) in text
            assert format_cnpj(f["recipient_cnpj"]) in text
            assert brl(f["total_value"]) in text
            assert br_date(f["issue_date"]) in text
            assert f["issuer_name"] in text
            assert FOOTER in text  # every page says it is synthetic
            if doc_type != "nfse":
                assert format_access_key(f["access_key"]) in text


def test_generation_is_deterministic(tmp_path):
    a = generate_corpus(6, tmp_path / "a", seed=99)
    b = generate_corpus(6, tmp_path / "b", seed=99)
    assert a == b
    c = generate_corpus(6, tmp_path / "c", seed=100)
    assert a != c


def test_formatting_helpers():
    assert brl(1234.5) == "1.234,50"
    assert brl(0.07) == "0,07"
    assert br_date("2026-03-01") == "01/03/2026"
    assert dotted_number("12345") == "000.012.345"
