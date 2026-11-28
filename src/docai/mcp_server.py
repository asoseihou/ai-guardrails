"""MCP server exposing the deterministic validators, the pipeline and the RAG search.

Run over stdio::

    python -m docai mcp          # or: python -m docai.mcp_server

Works with the official ``mcp`` SDK: ``MCPServer`` in 2.x, ``FastMCP`` in 1.x.
File-reading tools only accept files under ``DOCAI_MCP_ROOT`` (see ``docai.tools``).
"""

from __future__ import annotations

import importlib
from typing import Any

from docai import tools

INSTRUCTIONS = (
    "Tools for Brazilian fiscal documents (NF-e, CT-e, NFS-e). validate_cnpj and "
    "validate_access_key are deterministic; extract_document runs LLM extraction followed by "
    "guardrails on a PDF inside the configured root folder; search_fiscal_rules returns cited "
    "passages from local notes."
)


def _server_class() -> Any:
    for module_name, attr in (
        ("mcp.server.mcpserver", "MCPServer"),
        ("mcp.server.fastmcp", "FastMCP"),
    ):
        try:
            return getattr(importlib.import_module(module_name), attr)
        except (ImportError, AttributeError):
            continue
    raise RuntimeError("the MCP SDK is not installed: pip install 'ai-guardrails[mcp]'")


def build_server() -> Any:
    """Create the MCP server with the four tools registered."""
    server = _server_class()("docai", instructions=INSTRUCTIONS)

    @server.tool()  # type: ignore[untyped-decorator]
    def validate_cnpj(cnpj: str) -> dict[str, Any]:
        """Validate a CNPJ (14 positions, modulo-11 check digits; alphanumeric supported)."""
        return tools.validate_cnpj(cnpj)

    @server.tool()  # type: ignore[untyped-decorator]
    def validate_access_key(access_key: str) -> dict[str, Any]:
        """Validate a 44-digit NF-e/CT-e access key and decode state, date, CNPJ, model, number."""
        return tools.validate_access_key(access_key)

    @server.tool()  # type: ignore[untyped-decorator]
    def extract_document(pdf_path: str, xml_path: str | None = None) -> dict[str, Any]:
        """Extract a PDF under the allowed root (XML-first when its fiscal XML is given)."""
        return tools.extract_document(pdf_path, xml_path)

    @server.tool()  # type: ignore[untyped-decorator]
    def search_fiscal_rules(query: str, k: int = 3) -> list[dict[str, Any]]:
        """Search notes on access keys, CNPJ, CFOP, CT-e and NFS-e/ISS; returns cited passages."""
        return tools.search_fiscal_rules(query, k)

    return server


def main() -> None:
    """Serve over stdio."""
    build_server().run()


if __name__ == "__main__":
    main()
