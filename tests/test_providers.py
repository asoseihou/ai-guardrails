import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from docai.domain.schemas import CTeExtraction, NFeExtraction, NFSeExtraction
from docai.extract.providers import (
    ErrorRates,
    ExtractionContext,
    ProviderUnavailable,
    SimulatedLLMProvider,
    TruthStore,
    detect_real_provider,
    doc_id_from_filename,
    get_provider,
)
from docai.extract.providers.anthropic import AnthropicProvider
from docai.extract.providers.base import inline_json_schema
from docai.extract.providers.bedrock import BedrockProvider
from docai.extract.providers.gemini import GeminiProvider
from docai.settings import Settings

EXPECTED = {"number": "42", "series": "1", "issuer_cnpj": "11222333000181"}


def test_real_sdks_are_imported_lazily():
    code = (
        "import sys, docai.extract.providers, docai.pipeline; "
        "print([m for m in ('anthropic', 'google.genai', 'boto3') if m in sys.modules])"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_inline_schema_has_no_refs():
    schema = inline_json_schema(CTeExtraction.model_json_schema())
    assert "$defs" not in json.dumps(schema) and "$ref" not in json.dumps(schema)
    assert schema["properties"]["freight_components"]["anyOf"][0]["items"]["properties"]["value"]


class FakeAnthropic:
    def __init__(self):
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        block = SimpleNamespace(type="tool_use", input=dict(EXPECTED))
        return SimpleNamespace(
            content=[block], usage=SimpleNamespace(input_tokens=120, output_tokens=30)
        )


def test_anthropic_provider_uses_a_forced_tool():
    fake = FakeAnthropic()
    provider = AnthropicProvider("test-model", client=fake)
    assert provider.model == "test-model"
    result = provider.extract_with_usage(
        "DANFE ...", NFeExtraction, context=ExtractionContext(filename="a.pdf")
    )
    assert result.data == EXPECTED
    assert (result.usage.input_tokens, result.usage.output_tokens) == (120, 30)
    assert result.usage.cost_usd > 0 and result.usage.simulated is False
    call = fake.calls[0]
    assert call["tool_choice"] == {"type": "tool", "name": "record_extraction"}
    assert call["tools"][0]["input_schema"]["properties"]["access_key"]
    assert "a.pdf" in call["messages"][0]["content"]


def test_anthropic_without_tool_block_raises():
    class Bad(FakeAnthropic):
        def create(self, **kwargs):
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="hi")],
                usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            )

    with pytest.raises(Exception, match="tool_use"):
        AnthropicProvider("test-model", client=Bad()).extract("x", NFeExtraction)


def test_gemini_provider_parses_json():
    class Models:
        def generate_content(self, **kwargs):
            self.kwargs = kwargs
            meta = SimpleNamespace(
                prompt_token_count=200, candidates_token_count=40, thoughts_token_count=10
            )
            return SimpleNamespace(text=json.dumps(EXPECTED), usage_metadata=meta)

    client = SimpleNamespace(models=Models())
    provider = GeminiProvider(client=client)
    result = provider.extract_with_usage("NFS-e ...", NFSeExtraction)
    assert result.data == EXPECTED
    assert result.usage.output_tokens == 50  # thinking tokens billed as output
    config = client.models.kwargs["config"]
    assert config["response_mime_type"] == "application/json"
    assert "response_json_schema" in config


def test_bedrock_provider_uses_converse_tool():
    class Runtime:
        def converse(self, **kwargs):
            self.kwargs = kwargs
            return {
                "output": {"message": {"content": [{"toolUse": {"input": dict(EXPECTED)}}]}},
                "usage": {"inputTokens": 90, "outputTokens": 20},
                "metrics": {"latencyMs": 850},
            }

    runtime = Runtime()
    provider = BedrockProvider(model="some.model-id", client=runtime)
    result = provider.extract_with_usage("DACTE ...", CTeExtraction)
    assert result.data == EXPECTED and result.usage.latency_s == 0.85
    assert runtime.kwargs["toolConfig"]["toolChoice"] == {"tool": {"name": "record_extraction"}}


def test_missing_sdk_or_credentials(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(ProviderUnavailable):
        AnthropicProvider("test-model", api_key="x").extract("x", NFeExtraction)
    with pytest.raises(ProviderUnavailable, match="DOCAI_ANTHROPIC_MODEL"):
        AnthropicProvider(api_key="x")
    with pytest.raises(ProviderUnavailable):
        BedrockProvider()


def _settings(**kwargs):
    return Settings(_env_file=None, **kwargs)


def test_auto_selection(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "DOCAI_BEDROCK_MODEL_ID"):
        monkeypatch.delenv(var, raising=False)
    assert detect_real_provider(_settings()) is None
    assert isinstance(get_provider("auto", _settings()), SimulatedLLMProvider)
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    assert detect_real_provider(_settings()) == "gemini"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")  # a key without a model id is not enough
    assert detect_real_provider(_settings()) == "gemini"
    assert detect_real_provider(_settings(anthropic_model="test-model")) == "anthropic"
    with pytest.raises(ValueError):
        get_provider("nope", _settings())


def test_factory_passes_settings_to_real_providers(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    settings = _settings(anthropic_model="test-model", llm_max_output_tokens=512)
    provider = get_provider("anthropic", settings)
    assert isinstance(provider, AnthropicProvider)
    assert (provider.model, provider.max_tokens) == ("test-model", 512)
    assert "sk-test" not in repr(settings)  # secrets never show up in repr/logs
    bedrock = get_provider("bedrock", _settings(bedrock_model_id="m-1", aws_region="us-east-1"))
    assert isinstance(bedrock, BedrockProvider) and bedrock.region == "us-east-1"


def test_simulated_provider_is_deterministic_and_honest(corpus_dir, by_type, texts):
    store = TruthStore([corpus_dir])
    doc = by_type["nfe"][0]
    ctx = ExtractionContext(doc_id=doc["doc_id"], filename=doc["filename"])
    text = texts[doc["doc_id"]]
    noisy = SimulatedLLMProvider(store, ErrorRates().scaled(5), seed=1)
    a = noisy.extract_with_usage(text, NFeExtraction, context=ctx)
    b = noisy.extract_with_usage(text, NFeExtraction, context=ctx)
    assert a.data == b.data and a.meta == b.meta
    assert a.usage.simulated and a.meta["source"] == "ground_truth"
    clean = SimulatedLLMProvider(store, ErrorRates.none()).extract(text, NFeExtraction, context=ctx)
    from docai.domain.normalize import normalize_value
    from docai.domain.schemas import FIELD_KINDS

    assert {k: normalize_value(FIELD_KINDS[k], v) for k, v in clean.items()} == doc["fields"]


def test_doc_id_from_filename():
    assert doc_id_from_filename("doc_00012__lote-48213.pdf") == "doc_00012"
    assert doc_id_from_filename("/tmp/x/sample_00001__scan-1.pdf") == "sample_00001"
    assert doc_id_from_filename(None) is None
