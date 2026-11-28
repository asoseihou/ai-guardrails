"""Human review queue: repository, validated corrections, audit trail and the API."""

import sqlite3

import pytest
from fastapi.testclient import TestClient

from docai import container
from docai.classify import KeywordClassifier
from docai.extract.providers import ErrorRates, SimulatedLLMProvider, TruthStore
from docai.guardrails.validate import CorrectionRejected, validate_correction
from docai.pipeline import Pipeline
from docai.review import QueueFull, ReviewConflict, ReviewNotFound, ReviewService
from docai.storage import ReviewRepository, connect

TOKEN = "r" * 32
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture(scope="module")
def noisy_results(corpus_dir, corpus, texts):
    """Pipeline results under heavy noise: plenty of documents need review."""
    provider = SimulatedLLMProvider(TruthStore([corpus_dir]), ErrorRates().scaled(4), seed=3)
    pipeline = Pipeline(provider, KeywordClassifier(), "full")
    out = []
    for doc in corpus:
        if doc["fields"]:
            result = pipeline.run_text(texts[doc["doc_id"]], doc_id=doc["doc_id"])
            out.append((doc, result))
    return out


@pytest.fixture
def service(tmp_path):
    return ReviewService(ReviewRepository(tmp_path / "review.db"), max_pending=50)


def _needing_review(noisy_results):
    return [(d, r) for d, r in noisy_results if r.guardrails.needs_review]


def _enqueue(service, doc, result):
    return service.enqueue(result, text=result.text, document_sha256=doc["doc_id"], actor="test")


# ---------------------------------------------------------------- service + repository
def test_only_documents_that_need_review_are_queued(service, noisy_results):
    clean = next(r for _, r in noisy_results if not r.guardrails.needs_review)
    assert service.enqueue(clean, text="", document_sha256="x", actor="t") == (None, False)
    doc, result = _needing_review(noisy_results)[0]
    item_id, created = _enqueue(service, doc, result)
    assert created and len(item_id) == 32
    assert _enqueue(service, doc, result) == (item_id, False)  # same document: no duplicate
    item = service.get(item_id)
    assert item.status == "pending" and item.pending_fields
    assert item.audit[0].action == "created" and item.audit[0].actor == "test"
    assert "text" not in item.to_dict()  # the document text stays server-side


def test_corrections_are_validated_audited_and_approval_needs_complete_records(
    service, noisy_results
):
    doc, result = next(
        (d, r)
        for d, r in _needing_review(noisy_results)
        if any(r.guardrails.record[f] is None for f in r.guardrails.record if f != "service_order")
    )
    item_id, _ = _enqueue(service, doc, result)
    empty = [
        f for f, v in service.get(item_id).record.items() if v is None and f != "service_order"
    ]
    with pytest.raises(CorrectionRejected, match="correct the empty required fields first"):
        service.approve(item_id, reviewer="ana", reason="looks fine")
    with pytest.raises(CorrectionRejected):
        service.correct(item_id, empty[0], "99999999", reviewer="ana", reason="guess")
    for field in empty:
        service.correct(item_id, field, doc["fields"][field], reviewer="ana", reason="read it")
    item = service.approve(item_id, reviewer="ana", reason="checked against the PDF")
    assert item.status == "approved"
    assert all(item.record[f] == doc["fields"][f] for f in empty)
    actions = [entry.action for entry in item.audit]
    assert actions == ["created", *["corrected"] * len(empty), "approved"]
    corrected = item.audit[1]
    assert corrected.actor == "ana (bearer token)" and corrected.before is None
    assert corrected.after == doc["fields"][empty[0]]
    with pytest.raises(ReviewConflict):
        service.approve(item_id, reviewer="ana", reason="again")


def test_audit_trail_is_append_only(service, noisy_results):
    doc, result = _needing_review(noisy_results)[0]
    _enqueue(service, doc, result)
    with connect(service.repository.db_path) as db, pytest.raises(sqlite3.DatabaseError):
        db.execute("UPDATE review_audit SET actor = 'someone else'")
    with connect(service.repository.db_path) as db, pytest.raises(sqlite3.DatabaseError):
        db.execute("DELETE FROM review_audit")


def test_queries_are_parameterized(service, noisy_results):
    doc, result = _needing_review(noisy_results)[0]
    _enqueue(service, doc, result)
    with pytest.raises(ReviewNotFound):
        service.get("x' OR '1'='1")
    assert service.repository.list("pending' OR '1'='1") == []
    assert len(service.repository.list(None)) == 1


def test_queue_has_a_cap(tmp_path, noisy_results):
    small = ReviewService(ReviewRepository(tmp_path / "r.db"), max_pending=1)
    needing = _needing_review(noisy_results)
    _enqueue(small, *needing[0])
    with pytest.raises(QueueFull):
        _enqueue(small, *needing[1])


# ---------------------------------------------------------------- correction rules
@pytest.fixture
def cte(by_type, texts):
    doc = next(d for d in by_type["cte"] if d["layout"].get("attachment"))
    return doc, texts[doc["doc_id"]]


def test_correction_rules(cte):
    doc, text = cte
    f = doc["fields"]
    record = dict(f)
    ok = validate_correction("cte", "sender_cnpj", f["sender_cnpj"], record, text)
    assert ok == f["sender_cnpj"]
    cases = [
        ("sender_cnpj", f["sender_cnpj"][:-1] + str((int(f["sender_cnpj"][-1]) + 1) % 10), "check"),
        ("number", "123", "not printed"),
        ("issuer_cnpj", f["sender_cnpj"], "contradicts the access key"),
        ("gross_weight", doc["layout"]["attachment"]["gross_weight"], "main page"),
        ("service_order", "4492", "reference label"),
        ("access_key", doc["layout"]["referenced_keys"][0], "model 55"),
        ("total_value", 0, "positive"),
        ("nope", "1", "not a field"),
        ("issue_date", "31/02/2026", "invalid date"),
        ("cfop", None, "needs a value"),
    ]
    for field, value, message in cases:
        with pytest.raises(CorrectionRejected, match=message):
            validate_correction("cte", field, value, record, text)


# ---------------------------------------------------------------- API
@pytest.fixture
def review_client(api_env):
    from docai.api.app import create_app

    api_env.setenv("DOCAI_API_TOKEN", TOKEN)
    container.reset()
    return TestClient(create_app())


def _queued_pdf(corpus_dir, by_type):
    """A corpus PDF that the default pipeline sends to review."""
    pipeline = container.get_pipeline()
    for docs in by_type.values():
        for doc in docs:
            data = (corpus_dir / "pdf" / doc["filename"]).read_bytes()
            if (
                doc["fields"]
                and pipeline.run_pdf(data, filename=doc["filename"]).guardrails.needs_review
            ):
                return doc, data
    raise AssertionError("no document needs review")


def test_review_api_flow(review_client, corpus_dir, by_type):
    client = review_client
    doc, pdf = _queued_pdf(corpus_dir, by_type)
    files = {"file": (doc["filename"], pdf, "application/pdf")}
    assert client.post("/review/items", files=files).status_code == 401  # no token
    created = client.post("/review/items", files=files, headers=AUTH)
    assert created.status_code == 201
    item_id = created.json()["item_id"]
    assert client.post("/review/items", files=files, headers=AUTH).status_code == 200  # dedup
    assert client.get("/review/items").status_code == 401  # reads are private by default
    listing = client.get("/review/items", headers=AUTH).json()
    assert listing["pending"] == 1 and listing["items"][0]["id"] == item_id
    item = client.get(f"/review/items/{item_id}", headers=AUTH).json()
    field = item["pending_fields"][0]
    body = {"field": field, "value": doc["fields"][field], "reason": "re-read", "reviewer": "ana"}
    assert client.post(f"/review/items/{item_id}/corrections", json=body).status_code == 401
    corrected = client.post(f"/review/items/{item_id}/corrections", json=body, headers=AUTH)
    assert corrected.status_code == 200
    assert corrected.json()["field_status"][field] == "reviewed"
    bad = {**body, "value": "not printed anywhere 123"}
    rejected = client.post(f"/review/items/{item_id}/corrections", json=bad, headers=AUTH)
    assert rejected.status_code == 422 and rejected.json()["detail"]["reasons"]
    for name in client.get(f"/review/items/{item_id}", headers=AUTH).json()["pending_fields"]:
        fix = {**body, "field": name, "value": doc["fields"][name]}
        if fix["value"] is not None:
            client.post(f"/review/items/{item_id}/corrections", json=fix, headers=AUTH)
    approval = {"reviewer": "ana", "reason": "checked"}
    approved = client.post(f"/review/items/{item_id}/approve", json=approval, headers=AUTH)
    assert approved.status_code == 200 and approved.json()["status"] == "approved"
    again = client.post(f"/review/items/{item_id}/approve", json=approval, headers=AUTH)
    assert again.status_code == 409
    audit = client.get(f"/review/items/{item_id}", headers=AUTH).json()["audit"]
    assert audit[-1]["action"] == "approved" and audit[-1]["actor"] == "ana (bearer token)"


def test_review_api_input_validation(review_client):
    client = review_client
    assert client.get("/review/items/not-an-id", headers=AUTH).status_code == 422
    assert client.get(f"/review/items/{'a' * 32}", headers=AUTH).status_code == 404
    assert client.get("/review/items?status=drop", headers=AUTH).status_code == 422
    body = {"field": "number", "value": "1", "reason": "x", "reviewer": "<script>"}
    response = client.post(f"/review/items/{'a' * 32}/corrections", json=body, headers=AUTH)
    assert response.status_code == 422  # reason too short, reviewer pattern
    huge = {"field": "number", "value": "1" * 6000, "reason": "long", "reviewer": "ana"}
    response = client.post(f"/review/items/{'a' * 32}/corrections", json=huge, headers=AUTH)
    assert response.status_code == 422


def test_mutations_disabled_without_token(api_env, corpus_dir, by_type):
    from docai.api.app import create_app

    client = TestClient(create_app())
    doc = by_type["nfe"][0]
    files = {
        "file": ("a.pdf", (corpus_dir / "pdf" / doc["filename"]).read_bytes(), "application/pdf")
    }
    response = client.post("/review/items", files=files, headers=AUTH)
    assert response.status_code == 503
    assert client.get("/review/items").status_code == 503  # the queue is closed without a token


def test_reads_can_be_opened_for_a_demo(api_env):
    from docai.api.app import create_app

    api_env.setenv("DOCAI_API_TOKEN", TOKEN)
    api_env.setenv("DOCAI_REVIEW_PUBLIC_READ", "true")
    container.reset()
    client = TestClient(create_app())
    assert client.get("/review/items").status_code == 200
    assert (
        client.post(
            "/review/items", files={"file": ("a.pdf", b"%PDF", "application/pdf")}
        ).status_code
        == 401
    )


def test_cli_review(api_env, corpus_dir, by_type, capsys):
    from docai.cli import main

    pdfs = [str(corpus_dir / "pdf" / d["filename"]) for d in by_type["cte"][:3]]
    assert main(["review", "enqueue", *pdfs, "--with-xml"]) == 0
    out = capsys.readouterr().out
    assert out.count("\n") == 3
    assert main(["review", "list", "--status", "all"]) == 0
