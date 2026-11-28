import pytest

from docai import container

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient


@pytest.fixture
def client(api_env):
    from docai.api.app import create_app

    return TestClient(create_app())


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["provider"]["simulated"] is True
    assert body["classifier"] == "keyword-rules"
    assert body["ground_truth_documents"] == 24
    assert body["knowledge_chunks"] > 10


def test_index_page(client):
    response = client.get("/")
    assert response.status_code == 200 and "ai-guardrails" in response.text
    assert client.get("/static/app.js").status_code == 200


def test_extract_clean_document(client, corpus_dir, by_type):
    doc = by_type["cte"][0]
    pdf = (corpus_dir / "pdf" / doc["filename"]).read_bytes()
    response = client.post(
        "/extract", files={"file": (doc["filename"], pdf, "application/pdf")}, data={"noise": "0"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["doc_type"] == "cte" and body["extracted"]
    assert body["needs_review"] is False
    assert body["record"] == doc["fields"]
    assert {f["status"] for f in body["fields"]} == {"ok"}
    assert body["usage"]["simulated"] is True


def test_extract_noisy_document_reports_findings(client, corpus_dir, corpus):
    reported = 0
    for doc in corpus:
        if not doc["fields"]:
            continue
        pdf = (corpus_dir / "pdf" / doc["filename"]).read_bytes()
        body = client.post(
            "/extract",
            files={"file": (doc["filename"], pdf, "application/pdf")},
            data={"noise": "4"},
        ).json()
        reported += len(body["findings"])
        for finding in body["findings"]:
            assert finding["action"] in {"corrected", "nulled", "flagged"}
    assert reported > 0


def test_extract_non_fiscal_document(client, corpus_dir, by_type):
    doc = by_type["boleto"][0]
    pdf = (corpus_dir / "pdf" / doc["filename"]).read_bytes()
    body = client.post("/extract", files={"file": ("x.pdf", pdf, "application/pdf")}).json()
    assert body["doc_type"] == "boleto" and body["extracted"] is False and body["fields"] == []


def test_extract_unknown_pdf_uses_regex_baseline(client, corpus_dir, by_type):
    doc = by_type["nfse"][0]
    pdf = (corpus_dir / "pdf" / doc["filename"]).read_bytes()
    body = client.post(
        "/extract", files={"file": ("renamed.pdf", pdf, "application/pdf")}, data={"noise": "0"}
    ).json()
    assert body["meta"]["source"] == "regex_baseline"
    assert body["record"] == doc["fields"]


def test_extract_rejects_bad_input(client):
    assert (
        client.post("/extract", files={"file": ("a.txt", b"hello", "text/plain")}).status_code
        == 415
    )
    bad_noise = client.post(
        "/extract", files={"file": ("a.pdf", b"%PDF-1.4", "application/pdf")}, data={"noise": "9"}
    )
    assert bad_noise.status_code == 422
    broken = client.post("/extract", files={"file": ("a.pdf", b"%PDF-broken", "application/pdf")})
    assert broken.status_code == 422
    assert broken.json()["detail"] == "PDF rejected: the file could not be parsed as a PDF"
    assert broken.json()["request_id"] == broken.headers["x-request-id"]


def test_ask(client):
    response = client.post("/ask", json={"question": "What does a CFOP starting with 6 mean?"})
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "extractive" and len(body["passages"]) == 3
    assert any(p["section"].startswith("cfop.md") for p in body["passages"])
    invalid = client.post("/ask", json={"question": "x"})
    assert invalid.status_code == 422
    assert "input" not in invalid.text  # validation errors never echo the submitted value


def test_samples(client, corpus):
    samples = client.get("/samples").json()
    assert len(samples) == 24
    names = [s["name"] for s in samples]
    assert client.get(f"/samples/{names[0]}").content.startswith(b"%PDF")
    assert client.get("/samples/../../etc/passwd").status_code == 404
    with_xml = [s["name"] for s in samples if s["has_xml"]]
    assert with_xml and len(with_xml) == sum(1 for d in corpus if d["xml_filename"])
    assert client.get(f"/samples/{with_xml[0]}/xml").content.startswith(b"<?xml")
    without = next(s["name"] for s in samples if not s["has_xml"])
    assert client.get(f"/samples/{without}/xml").status_code == 404


def test_container_wires_the_settings_into_the_pipeline(api_env):
    api_env.setenv("DOCAI_XML_FIRST", "false")
    api_env.setenv("DOCAI_MAX_XML_BYTES", "2048")
    api_env.setenv("DOCAI_MAX_PDF_PAGES", "3")
    container.reset()
    pipeline = container.get_pipeline()
    assert pipeline.xml_first is False and pipeline.max_xml_bytes == 2048
    assert pipeline.limits.max_pages == 3
    assert pipeline.registry  # CNPJ -> registered name, from the synthetic company pool
