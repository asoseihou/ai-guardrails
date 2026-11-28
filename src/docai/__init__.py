"""docai — LLM extraction of Brazilian fiscal documents with deterministic guardrails.

Everything in this package runs offline on synthetic data. Real LLM providers
(Gemini, Anthropic, Bedrock) are optional and only used when credentials exist.

Layers (enforced by ``tests/test_architecture.py``)::

    interfaces   docai.api, docai.cli, docai.mcp_server, docai.tools
    services     docai.pipeline, docai.review, docai.eval, docai.container
    adapters     docai.extract, docai.storage, docai.classify, docai.rag, docai.synth
    core         docai.guardrails (pure), docai.settings, docai.observability
    domain       docai.domain (pure: schemas, findings, fiscal rules)
"""

__version__ = "0.2.0"
