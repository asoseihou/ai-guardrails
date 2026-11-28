import asyncio
import json

import pytest

from docai import tools
from docai.cli import main


def test_validate_tools():
    ok = tools.validate_cnpj("11.222.333/0001-81")
    assert ok["valid"] and ok["formatted"] == "11.222.333/0001-81"
    bad = tools.validate_cnpj("11.222.333/0001-80")
    assert not bad["valid"] and bad["expected_check_digits"] == "81"
    key = "35250611222333000181550010000123451123456781"
    decoded = tools.validate_access_key(key)
    assert decoded["valid"] and decoded["doc_type"] == "nfe"
    assert decoded["fields"]["number"] == 12345 and decoded["fields"]["issuer_cnpj_valid"]
    assert not tools.validate_access_key(key[:-1] + "0")["valid"]
    assert not tools.validate_access_key("123")["valid"]


def test_search_tool():
    hits = tools.search_fiscal_rules("CFOP first digit", k=2)
    assert len(hits) == 2 and hits[0]["section"].startswith("cfop.md")


def test_mcp_server_registers_tools():
    pytest.importorskip("mcp")
    from docai.mcp_server import build_server

    server = build_server()
    names = {tool.name for tool in asyncio.run(server.list_tools())}
    assert names == {
        "validate_cnpj",
        "validate_access_key",
        "extract_document",
        "search_fiscal_rules",
    }
    result = asyncio.run(server.call_tool("validate_cnpj", {"cnpj": "11222333000181"}))
    content = getattr(result, "content", result)  # mcp 2.x: CallToolResult
    if isinstance(content, tuple):  # mcp 1.x: (content blocks, structured output)
        content = content[0]
    assert json.loads(content[0].text)["valid"] is True


def test_cli_validate_and_synth(tmp_path, capsys):
    assert main(["validate", "11.222.333/0001-81"]) == 0
    assert '"valid": true' in capsys.readouterr().out
    assert main(["synth", "--types", "nfe,cte", "--out", str(tmp_path), "--prefix", "t"]) == 0
    assert len(list((tmp_path / "pdf").glob("*.pdf"))) == 2


@pytest.mark.slow
def test_cli_demo_end_to_end(tmp_path, capsys, api_env):
    pytest.importorskip("torch")
    args = [
        "demo",
        "--n",
        "40",
        "--data",
        str(tmp_path / "data"),
        "--model",
        str(tmp_path / "m.pt"),
        "--report-dir",
        str(tmp_path / "reports"),
    ]
    assert main(args) == 0
    report = json.loads((tmp_path / "reports" / "eval_report.json").read_text())
    assert report["configs"]["full"]["false_corrections"] == 0
    assert report["usage"]["simulated"] is True
    assert set(report["configs"]) == {"off", "dv-only", "text", "full"}
    assert "Silent-error documents" in capsys.readouterr().out
    from docai import container

    assert len(container.get_run_repository().list()) == 1  # the run is tracked


def test_cli_reports_bad_configuration_without_the_value(monkeypatch):
    from docai import container

    secret = "too-short-secret"
    monkeypatch.setenv("DOCAI_API_TOKEN", secret)
    container.reset()
    try:
        with pytest.raises(SystemExit) as exc:
            main(["validate", "11.222.333/0001-81"])
    finally:
        monkeypatch.delenv("DOCAI_API_TOKEN")
        container.reset()
    message = str(exc.value)
    assert message.startswith("invalid configuration: api_token")
    assert secret not in message
