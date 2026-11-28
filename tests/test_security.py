"""Security controls: headers, limits, rate limiting, error shielding, auth, sandboxing, logging."""

import io
import json
import logging
import os

import pytest

pytest.importorskip("fastapi")
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from reportlab.lib import pdfencrypt
from reportlab.pdfgen.canvas import Canvas

from docai import container, tools
from docai.api.middleware import UI_CSP, security_headers
from docai.api.ratelimit import TokenBucketLimiter
from docai.api.security import require_token, token_matches
from docai.extract import pdf as pdf_module
from docai.extract.pdf import PdfLimits, PdfRejected, read_pdf_text
from docai.observability import JsonFormatter, RedactingFilter, request_id_var, short_hash
from docai.settings import Settings

TOKEN = "t" * 32


def _app():
    from docai.api.app import create_app

    return TestClient(create_app(), raise_server_exceptions=False)


@pytest.fixture
def client(api_env):
    return _app()


@pytest.fixture
def pdf_bytes(corpus_dir, by_type):
    doc = by_type["nfe"][0]
    return (corpus_dir / "pdf" / doc["filename"]).read_bytes()


def _encrypted_pdf() -> bytes:
    buf = io.BytesIO()
    canvas = Canvas(buf, encrypt=pdfencrypt.StandardEncryption("user-pw", "owner-pw"))
    canvas.drawString(100, 100, "secret")
    canvas.showPage()
    canvas.save()
    return buf.getvalue()


def _pages_pdf(n: int) -> bytes:
    buf = io.BytesIO()
    canvas = Canvas(buf)
    for i in range(n):
        canvas.drawString(100, 100, f"page {i}")
        canvas.showPage()
    canvas.save()
    return buf.getvalue()


# ---------------------------------------------------------------- headers / request ids
def test_security_headers_on_pages_api_and_errors(client):
    for response in (client.get("/"), client.get("/health"), client.get("/samples/nope")):
        headers = response.headers
        assert headers["content-security-policy"] == UI_CSP
        assert "'unsafe-inline'" not in headers["content-security-policy"]
        assert "frame-ancestors 'none'" in headers["content-security-policy"]
        assert headers["x-content-type-options"] == "nosniff"
        assert headers["x-frame-options"] == "DENY"
        assert headers["referrer-policy"] == "no-referrer"
        assert "camera=()" in headers["permissions-policy"]
        assert len(headers["x-request-id"]) == 32
    assert "access-control-allow-origin" not in client.get("/health").headers  # no CORS


def test_ui_has_no_inline_script_or_style(client):
    html = client.get("/").text
    assert "<script>" not in html and "<style" not in html and "style=" not in html
    assert 'src="/static/app.js"' in html


def test_request_id_is_echoed_only_when_safe(client):
    good = client.get("/health", headers={"X-Request-ID": "trace-1234abcd"})
    assert good.headers["x-request-id"] == "trace-1234abcd"
    bad = client.get("/health", headers={"X-Request-ID": "x\r\nSet-Cookie: a=b"})
    assert bad.headers["x-request-id"] != "x\r\nSet-Cookie: a=b"
    assert len(bad.headers["x-request-id"]) == 32


def test_docs_page_uses_pinned_cdn_assets_with_integrity(client):
    page = client.get("/docs")
    assert page.status_code == 200
    assert page.text.count('integrity="sha384-') == 2 and "swagger-ui-dist@5.34.1" in page.text
    assert (
        page.headers["content-security-policy"]
        == security_headers("/docs")["content-security-policy"]
    )
    assert "<script>" not in page.text  # bootstrap lives in /static/docs.js


def test_docs_can_be_disabled(api_env):
    api_env.setenv("DOCAI_DOCS_ENABLED", "false")
    container.reset()
    client = _app()
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404


# ---------------------------------------------------------------- body size
def test_oversized_declared_body_is_rejected_before_reading(client):
    huge = str(50 * 1024 * 1024)
    response = client.post("/extract", content=b"x", headers={"Content-Length": huge})
    assert response.status_code == 413
    assert response.json()["request_id"] == response.headers["x-request-id"]


def test_oversized_streamed_body_is_cut_off(api_env):
    api_env.setenv("DOCAI_MAX_UPLOAD_BYTES", "2048")
    api_env.setenv("DOCAI_MAX_XML_BYTES", "1024")
    container.reset()
    client = _app()

    def chunks():  # no Content-Length: the middleware must count the streamed bytes
        for _ in range(200):
            yield b"x" * 4096

    response = client.post("/ask", content=chunks(), headers={"Content-Type": "application/json"})
    assert response.status_code == 413


def test_body_limit_stops_reading_early():
    """Raw ASGI: the app stops pulling chunks as soon as the limit is crossed."""
    import asyncio

    from docai.api.middleware import BodySizeLimitMiddleware

    pulled, sent = [], []

    async def app(scope, receive, send):
        while True:  # an app that reads the whole body
            message = await receive()
            if not message.get("more_body"):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def receive():
        pulled.append(1)
        return {"type": "http.request", "body": b"x" * 1000, "more_body": len(pulled) < 100}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": "POST", "path": "/x", "headers": []}
    asyncio.run(BodySizeLimitMiddleware(app, max_bytes=5000)(scope, receive, send))
    assert len(pulled) == 6  # 6,000 bytes > 5,000: stop, never read the other 94 chunks
    assert sent[0]["status"] == 413 and len(sent) == 2


def test_file_over_the_upload_limit_is_rejected_in_the_handler(api_env, pdf_bytes):
    api_env.setenv("DOCAI_MAX_UPLOAD_BYTES", "2048")
    container.reset()
    response = _app().post("/extract", files={"file": ("a.pdf", pdf_bytes, "application/pdf")})
    assert response.status_code == 413


def test_invalid_content_length(client):
    response = client.post("/extract", content=b"x", headers={"Content-Length": "abc"})
    assert response.status_code == 400


# ---------------------------------------------------------------- PDF limits
def test_pdf_limits_direct(pdf_bytes):
    with pytest.raises(PdfRejected, match="encrypted"):
        read_pdf_text(_encrypted_pdf())
    with pytest.raises(PdfRejected, match="could not be parsed"):
        read_pdf_text(b"%PDF-1.7 garbage")
    with pytest.raises(PdfRejected, match="not a PDF"):
        read_pdf_text(b"GIF89a")
    with pytest.raises(PdfRejected, match="larger than"):
        read_pdf_text(pdf_bytes, PdfLimits(max_bytes=100))
    with pytest.raises(PdfRejected, match="21 pages; the limit is 20"):
        read_pdf_text(_pages_pdf(21), PdfLimits(isolate=False))
    with pytest.raises(PdfRejected, match="exceeds 1,000 characters"):
        read_pdf_text(pdf_bytes, PdfLimits(max_chars=1000, isolate=False))
    with pytest.raises(PdfRejected, match="timed out"):
        read_pdf_text(pdf_bytes, PdfLimits(timeout_s=0.001))
    isolated = read_pdf_text(pdf_bytes, PdfLimits(isolate=True))
    assert isolated == read_pdf_text(pdf_bytes, PdfLimits(isolate=False))


def test_pdf_errors_return_422_with_a_clear_message(client):
    files = {"file": ("e.pdf", _encrypted_pdf(), "application/pdf")}
    body = client.post("/extract", files=files).json()
    assert body["detail"] == "PDF rejected: encrypted or password-protected PDFs are not supported"
    many = client.post("/extract", files={"file": ("m.pdf", _pages_pdf(25), "application/pdf")})
    assert many.status_code == 422 and "25 pages" in many.json()["detail"]


def test_pdf_worker_protocol_in_process(monkeypatch, capsys, pdf_bytes):
    class Stdin:
        buffer = io.BytesIO(pdf_bytes)

    monkeypatch.setattr(pdf_module, "_limit_resources", lambda seconds: None)
    monkeypatch.setattr(pdf_module.sys, "stdin", Stdin())
    assert pdf_module._worker_main(["20", "200000", "5"]) == 0
    reply = json.loads(capsys.readouterr().out)
    assert reply["status"] == "ok" and reply["pages"] == 1 and "DANFE" in reply["text"]
    Stdin.buffer = io.BytesIO(b"%PDF-broken")
    pdf_module._worker_main(["20", "200000", "5"])
    assert json.loads(capsys.readouterr().out)["status"] == "rejected"


def test_pdf_worker_does_not_inherit_secrets(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    env = pdf_module._worker_env()
    assert "ANTHROPIC_API_KEY" not in env and "sk-should-not-leak" not in json.dumps(env)


# ---------------------------------------------------------------- rate limiting
def test_token_bucket():
    now = [0.0]
    limiter = TokenBucketLimiter(60, 2, max_keys=2, clock=lambda: now[0])
    assert limiter.acquire("a") == 0 and limiter.acquire("a") == 0
    assert limiter.acquire("a") == pytest.approx(1.0)  # 1 token per second
    now[0] += 1.0
    assert limiter.acquire("a") == 0
    limiter.acquire("b")
    limiter.acquire("c")  # evicts "a" (bounded memory)
    assert "a" not in limiter._buckets
    with pytest.raises(ValueError):
        TokenBucketLimiter(0, 1)


def test_rate_limit_applies_to_posts_only(api_env):
    api_env.setenv("DOCAI_RATE_LIMIT_PER_MINUTE", "1")
    api_env.setenv("DOCAI_RATE_LIMIT_BURST", "2")
    container.reset()
    client = _app()
    statuses = [client.post("/ask", json={"question": "CFOP?"}).status_code for _ in range(3)]
    assert statuses == [200, 200, 429]
    limited = client.post("/ask", json={"question": "CFOP?"})
    assert int(limited.headers["retry-after"]) >= 1
    assert limited.json()["detail"] == "too many requests"
    assert all(client.get("/health").status_code == 200 for _ in range(5))


# ---------------------------------------------------------------- error shielding
def test_unexpected_errors_do_not_leak_internals(api_env, monkeypatch, pdf_bytes, caplog):
    def boom(noise=None):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(container, "get_pipeline", boom)
    with caplog.at_level(logging.ERROR, logger="docai.api"):
        response = _app().post("/extract", files={"file": ("a.pdf", pdf_bytes, "application/pdf")})
    assert response.status_code == 500
    assert response.json() == {
        "detail": "internal error",
        "request_id": response.headers["x-request-id"],
    }
    assert "RuntimeError" not in response.text and "secret" not in response.text
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "secret internal detail" in caplog.text  # details stay in the server log


def test_provider_errors_are_generic(api_env, monkeypatch, pdf_bytes):
    from docai.extract.providers import ProviderError

    class Failing:
        def run_pdf(self, data, filename=None, xml_data=None):
            raise ProviderError("upstream said: invalid x-api-key sk-abc")

    monkeypatch.setattr(container, "get_pipeline", lambda noise=None: Failing())
    response = _app().post("/extract", files={"file": ("a.pdf", pdf_bytes, "application/pdf")})
    assert response.status_code == 502 and "sk-abc" not in response.text


# ---------------------------------------------------------------- CORS
def test_cors_allowlist(api_env):
    api_env.setenv("DOCAI_CORS_ORIGINS", "https://ui.example.com")
    container.reset()
    client = _app()
    ok = client.get("/health", headers={"Origin": "https://ui.example.com"})
    assert ok.headers["access-control-allow-origin"] == "https://ui.example.com"
    other = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in other.headers


def test_settings_validation():
    with pytest.raises(ValueError, match="explicit"):
        Settings(cors_origins=["*"])
    with pytest.raises(ValueError, match="at least 24"):
        Settings(api_token="short")
    settings = Settings(api_token=TOKEN, anthropic_api_key="fake-provider-key-123456")
    assert TOKEN not in repr(settings) and "fake-provider-key" not in repr(settings)
    assert settings.mutations_enabled and set(settings.secret_values()) == {
        TOKEN,
        "fake-provider-key-123456",
    }
    assert Settings(api_token="").api_token is None
    empty = Settings(anthropic_api_key="", gemini_api_key="  ")  # compose passes "" when unset
    assert empty.anthropic_api_key is None and empty.gemini_api_key is None


# ---------------------------------------------------------------- authentication
def _protected_app():
    app = FastAPI()

    @app.post("/mutate", dependencies=[Depends(require_token)])
    def mutate() -> dict:
        return {"ok": True}

    return TestClient(app)


def test_mutations_are_disabled_without_a_configured_token(api_env):
    response = _protected_app().post("/mutate", headers={"Authorization": f"Bearer {TOKEN}"})
    assert response.status_code == 503 and "DOCAI_API_TOKEN" in response.json()["detail"]


def test_bearer_token_is_required(api_env):
    api_env.setenv("DOCAI_API_TOKEN", TOKEN)
    container.reset()
    client = _protected_app()
    missing = client.post("/mutate")
    assert missing.status_code == 401 and missing.headers["www-authenticate"] == "Bearer"
    assert client.post("/mutate", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.post("/mutate", headers={"Authorization": f"Basic {TOKEN}"}).status_code == 401
    ok = client.post("/mutate", headers={"Authorization": f"Bearer {TOKEN}"})
    assert ok.status_code == 200
    assert token_matches(TOKEN, TOKEN) and not token_matches(TOKEN, TOKEN[:-1] + "x")


# ---------------------------------------------------------------- MCP file sandbox
@pytest.fixture
def sandbox(tmp_path, monkeypatch, pdf_bytes):
    root = tmp_path / "root"
    root.mkdir()
    (root / "ok.pdf").write_bytes(pdf_bytes)
    (root / "fake.pdf").write_bytes(b"MZ not a pdf")
    (root / "notes.txt").write_bytes(b"%PDF- but wrong suffix")
    (root / "dir.pdf").mkdir()
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(pdf_bytes)
    (root / "link.pdf").symlink_to(outside)
    monkeypatch.setenv("DOCAI_MCP_ROOT", str(root))
    monkeypatch.setenv("DOCAI_RATE_LIMIT_PER_MINUTE", "0")
    container.reset()
    yield root
    container.reset()


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("../outside.pdf", "inside the allowed root"),
        ("/etc/passwd", "inside the allowed root"),
        ("link.pdf", "inside the allowed root"),  # symlink escaping the root
        ("missing.pdf", "not found"),
        ("notes.txt", "only .pdf"),
        ("fake.pdf", "not a .pdf document"),
        ("dir.pdf", "not a regular file"),
        ("", "invalid path"),
        ("a\x00.pdf", "invalid path"),
        ("x" * 2000, "invalid path"),
    ],
)
def test_extract_document_tool_is_sandboxed(sandbox, path, message):
    with pytest.raises(tools.ToolInputError, match=message):
        tools.extract_document(path)


def test_sandbox_size_limit(sandbox):
    with pytest.raises(tools.ToolInputError, match="larger than"):
        tools.read_sandboxed_file("ok.pdf", sandbox, suffix=".pdf", max_bytes=10, magic=b"%PDF-")


def test_extract_document_tool_reads_allowed_files(sandbox):
    result = tools.extract_document("ok.pdf")
    assert result["doc_type"] == "nfe" and result["filename"] == "ok.pdf"
    assert tools.extract_document(str(sandbox / "ok.pdf"))["doc_type"] == "nfe"


def test_mcp_root_defaults_to_the_samples_folder(monkeypatch, tmp_path):
    monkeypatch.delenv("DOCAI_MCP_ROOT", raising=False)
    settings = Settings(samples_dir=tmp_path)
    assert settings.mcp_root == tmp_path.resolve()


def test_search_tool_bounds_the_query():
    with pytest.raises(tools.ToolInputError):
        tools.search_fiscal_rules("x" * 600)


# ---------------------------------------------------------------- logging
def test_redaction_and_json_logs():
    record = logging.makeLogRecord(
        {
            "msg": "calling with %s and Authorization: Bearer abc.def",
            "args": ("sk-live-12345678",),
            "levelname": "INFO",
        }
    )
    RedactingFilter(["sk-live-12345678"]).filter(record)
    assert "sk-live" not in record.getMessage() and "abc.def" not in record.getMessage()
    token = request_id_var.set("req-123")
    try:
        from docai.observability import RequestIdFilter

        RequestIdFilter().filter(record)
        payload = json.loads(JsonFormatter().format(record))
    finally:
        request_id_var.reset(token)
    assert payload["request_id"] == "req-123" and payload["level"] == "INFO"
    assert short_hash(b"document") == short_hash("document") and len(short_hash(b"x")) == 12


def test_exception_tracebacks_are_redacted():
    try:
        raise RuntimeError("key sk-live-12345678 rejected")
    except RuntimeError:
        import sys

        record = logging.makeLogRecord({"msg": "failed", "exc_info": sys.exc_info()})
    RedactingFilter(["sk-live-12345678"]).filter(record)
    assert "sk-live" not in json.loads(JsonFormatter().format(record))["exception"]


def test_configure_logging_is_idempotent():
    from docai.observability import configure_logging

    configure_logging("INFO", json_output=True)
    configure_logging("INFO", json_output=False)
    assert sum(h.get_name() == "docai" for h in logging.getLogger().handlers) == 1


def test_uploads_are_logged_by_hash_only(client, pdf_bytes, caplog):
    with caplog.at_level(logging.INFO):
        client.post("/extract", files={"file": ("Secret Name.pdf", pdf_bytes, "application/pdf")})
    assert short_hash(pdf_bytes) in caplog.text
    assert "Secret Name" not in caplog.text and "DANFE" not in caplog.text


def test_filenames_are_sanitized_before_reaching_the_prompt():
    from docai.api.routers.documents import safe_filename

    assert safe_filename("../../etc/<document>x.pdf") == "documentx.pdf"
    assert safe_filename("C:\\Users\\a\\nota.pdf") == "nota.pdf"
    assert safe_filename("a" * 300 + ".pdf") == "a" * 128
    assert safe_filename(None) is None and safe_filename("\x00\x01") is None
    assert os.sep not in (safe_filename("x/y/z.pdf") or "")
