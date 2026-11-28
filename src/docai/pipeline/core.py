"""End-to-end pipeline: PDF (+ optional XML) → text → document type → extraction → guardrails.

Decision order for an extractable document:

1. a valid fiscal XML of the same type covers every required field and
   ``xml_first`` is on → the record comes from the XML (``source="xml"``), no
   LLM call;
2. otherwise the LLM extracts from the PDF text; when an XML is present it wins
   on every field it has, each disagreement is recorded as a finding, and the
   LLM only fills what the XML lacks (``source="llm+xml"``);
3. the deterministic guardrails run in every case.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from docai.classify import DocumentClassifier, Prediction
from docai.domain.doc_types import EXTRACTABLE_TYPES
from docai.domain.layout import looks_like_boleto
from docai.extract.pdf import PdfLimits, read_pdf_text
from docai.extract.providers import ExtractionContext, LLMProvider, Usage
from docai.extract.xml_reader import FiscalXml, read_fiscal_xml
from docai.guardrails import GuardrailConfig, GuardrailResult, run_guardrails
from docai.pipeline.recovery import TRUNCATION_REASON, extract_with_recovery


@dataclass
class PipelineResult:
    """Everything known about one processed document (serializable with ``to_dict``)."""

    filename: str | None
    prediction: Prediction
    classifier: str
    provider: dict[str, Any]
    raw: dict[str, Any] | None = None
    guardrails: GuardrailResult | None = None
    usage: Usage | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    text_chars: int = 0
    xml_status: str = "absent"  # absent | used | cross_checked | type_mismatch | inconsistent
    text: str = field(default="", repr=False)  # kept server-side (review), never serialized

    @property
    def extracted(self) -> bool:
        """True when the document went through extraction and guardrails."""
        return self.guardrails is not None

    def _fields(self) -> list[dict[str, Any]]:
        g = self.guardrails
        if g is None:
            return []
        return [
            {
                "name": name,
                "raw": (self.raw or {}).get(name),
                "final": value,
                "status": str(g.status(name)),
                "provenance": g.provenance.get(name),
                "reasons": [f.reason for f in g.findings if f.field == name],
            }
            for name, value in g.record.items()
        ]

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view used by the API, the MCP server and the UI."""
        g = self.guardrails
        return {
            "filename": self.filename,
            "doc_type": self.prediction.label,
            "confidence": self.prediction.confidence,
            "probabilities": self.prediction.probabilities,
            "classifier": self.classifier,
            "provider": self.provider,
            "extracted": self.extracted,
            "source": g.source if g else None,
            "xml_status": self.xml_status,
            "needs_review": g.needs_review if g else None,
            "review_reasons": list(g.review_reasons) if g else [],
            "fields": self._fields(),
            "record": g.record if g else None,
            "findings": [f.to_dict() for f in g.findings] if g else [],
            "usage": self.usage.to_dict() if self.usage else None,
            "meta": self.meta,
            "text_chars": self.text_chars,
        }


def route(prediction: Prediction, text: str) -> Prediction:
    """Deterministic routing rules on top of the classifier.

    A boleto (bank slip) is never an invoice, even when it carries the same amount
    and due date: a digitable line without any fiscal-document marker settles it.
    """
    if prediction.label in EXTRACTABLE_TYPES and looks_like_boleto(text):
        return Prediction("boleto", 1.0, {**prediction.probabilities, "boleto": 1.0})
    return prediction


def xml_decision(prediction_label: str, xml: FiscalXml | None) -> tuple[FiscalXml | None, str]:
    """Which XML (if any) may be trusted for a document classified as ``prediction_label``."""
    if xml is None:
        return None, "absent"
    if xml.doc_type != prediction_label:
        return None, "type_mismatch"
    if xml.problems:
        return None, "inconsistent"
    return xml, "used"


def needs_llm(xml: FiscalXml | None, xml_first: bool) -> bool:
    """Whether the LLM must be called (no usable XML, XML-first off, or the XML is incomplete)."""
    return xml is None or not xml_first or bool(xml.missing)


def guard(
    doc_type: str,
    text: str,
    llm_data: dict[str, Any] | None,
    xml: FiscalXml | None,
    config: str | GuardrailConfig,
    *,
    xml_first: bool,
    registry: Mapping[str, str] | None = None,
) -> GuardrailResult:
    """Run the guardrails on the XML (XML-first) or on the LLM output cross-checked by the XML.

    The single place where the XML-first policy lives; the API, the CLI and the
    evaluation harness all call it.
    """
    if xml is not None and not needs_llm(xml, xml_first):
        return run_guardrails(
            doc_type,
            xml.fields,
            text,
            config,
            xml_fields=xml.fields,
            source="xml",
            registry=registry,
        )
    xml_fields = xml.fields if xml is not None else None
    return run_guardrails(
        doc_type, llm_data or {}, text, config, xml_fields=xml_fields, registry=registry
    )


class Pipeline:
    """Runs one document through classification, extraction and the guardrails."""

    def __init__(
        self,
        provider: LLMProvider,
        classifier: DocumentClassifier,
        guardrails: str | GuardrailConfig = "full",
        *,
        limits: PdfLimits | None = None,
        xml_first: bool = True,
        max_xml_bytes: int = 1024 * 1024,
        registry: Mapping[str, str] | None = None,
    ) -> None:
        self.provider = provider
        self.classifier = classifier
        self.guardrails = guardrails
        self.limits = limits or PdfLimits()
        self.xml_first = xml_first
        self.max_xml_bytes = max_xml_bytes
        self.registry = registry

    def run_text(
        self,
        text: str,
        *,
        filename: str | None = None,
        doc_id: str | None = None,
        xml: FiscalXml | None = None,
    ) -> PipelineResult:
        """Process an already-extracted text layer (and the document's XML, if any)."""
        prediction = route(self.classifier.predict(text), text)
        result = PipelineResult(
            filename=filename,
            prediction=prediction,
            classifier=self.classifier.name,
            provider=self.provider.describe(),
            text_chars=len(text),
            text=text,
        )
        if prediction.label not in EXTRACTABLE_TYPES:
            return result
        trusted_xml, result.xml_status = xml_decision(prediction.label, xml)
        llm_data = None
        failed = False
        if needs_llm(trusted_xml, self.xml_first):
            context = ExtractionContext(doc_id=doc_id, filename=filename)
            recovery = extract_with_recovery(self.provider, text, prediction.label, context)
            result.usage, result.meta = recovery.usage, recovery.meta()
            if recovery.result is not None:
                llm_data = result.raw = recovery.result.data
            failed = recovery.result is None
            if trusted_xml is not None:
                result.xml_status = "cross_checked"
        elif trusted_xml is not None:
            result.raw = dict(trusted_xml.fields)
        result.guardrails = guard(
            prediction.label,
            text,
            llm_data,
            trusted_xml,
            self.guardrails,
            xml_first=self.xml_first,
            registry=self.registry,
        )
        if failed:
            result.guardrails.review_reasons.append(TRUNCATION_REASON)
        if result.xml_status in ("type_mismatch", "inconsistent"):
            problems = "; ".join(xml.problems) if xml and xml.problems else "wrong document type"
            result.guardrails.review_reasons.append(f"the attached XML was ignored: {problems}")
        return result

    def run_pdf(
        self, data: bytes, *, filename: str | None = None, xml_data: bytes | None = None
    ) -> PipelineResult:
        """Process an untrusted PDF (and XML); raises ``PdfRejected`` / ``XmlRejected``."""
        xml = read_fiscal_xml(xml_data, self.max_xml_bytes) if xml_data is not None else None
        return self.run_text(read_pdf_text(data, self.limits).text, filename=filename, xml=xml)
