"""Guardrail configurations and the runner."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from docai.domain.findings import GuardrailResult
from docai.domain.schemas import field_names
from docai.guardrails import anchors, checks
from docai.guardrails.text_index import DocumentText


@dataclass(frozen=True)
class GuardrailConfig:
    """Which groups of checks run (presets below)."""

    name: str
    schema: bool = True
    cnpj_dv: bool = True
    key_dv: bool = True
    key_cross: bool = True
    key_reread: bool = True
    presence: bool = True
    consistency: bool = True
    required: bool = True
    xml_cross: bool = True
    anchors: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Plain dict, for reports."""
        return asdict(self)


PRESETS: dict[str, GuardrailConfig] = {
    # raw model output, no checks at all
    "off": GuardrailConfig(
        "off",
        schema=False,
        cnpj_dv=False,
        key_dv=False,
        key_cross=False,
        key_reread=False,
        presence=False,
        consistency=False,
        required=False,
        xml_cross=False,
        anchors=False,
    ),
    # v1: format + check digits only (what most pipelines start with)
    "dv-only": GuardrailConfig(
        "dv-only",
        key_cross=False,
        key_reread=False,
        presence=False,
        consistency=False,
        xml_cross=False,
        anchors=False,
    ),
    # v2: + access-key cross-checks, literal presence and consistency
    "text": GuardrailConfig("text", anchors=False),
    # v3: + layout anchors (role blocks, main page, labels) and the name registry
    "full": GuardrailConfig("full"),
}


def get_config(config: str | GuardrailConfig) -> GuardrailConfig:
    """Resolve a preset name (``off``, ``dv-only``, ``text``, ``full``) or pass a config through."""
    if isinstance(config, GuardrailConfig):
        return config
    try:
        return PRESETS[config]
    except KeyError:
        raise ValueError(f"unknown guardrail preset {config!r}; choose {list(PRESETS)}") from None


Check = Callable[[checks.State], None]


def _steps(
    cfg: GuardrailConfig,
    names: list[str],
    xml_fields: dict[str, Any] | None,
    registry: Mapping[str, str] | None,
) -> list[tuple[bool, Check]]:
    """The checks in execution order, each with whether ``cfg`` enables it."""
    with_xml = xml_fields is not None and cfg.xml_cross
    return [
        (cfg.schema, lambda s: checks.check_schema(s, names)),
        (with_xml, lambda s: checks.check_xml_cross(s, xml_fields or {})),
        (
            cfg.key_dv,
            lambda s: checks.check_access_key(
                s, model_check=cfg.key_cross, presence=cfg.presence, reread=cfg.key_reread
            ),
        ),
        (cfg.key_cross, checks.check_key_cross),
        (cfg.anchors, anchors.check_role_anchors),
        (cfg.anchors, anchors.check_cte_main_page),
        (cfg.anchors, anchors.check_referenced_keys),
        (cfg.anchors, anchors.check_references),
        (cfg.cnpj_dv, checks.check_cnpj_dv),
        (cfg.presence, checks.check_presence),
        (
            cfg.anchors and registry is not None,
            lambda s: anchors.check_name_registry(s, registry or {}),
        ),
        (cfg.consistency, checks.check_cte_components),
        (cfg.consistency, checks.check_nfse_iss),
        (cfg.anchors, anchors.check_nfse_services),
        (cfg.consistency, checks.check_parties),
        (cfg.consistency and cfg.presence, checks.check_xml_printed),
        (cfg.required, checks.check_required),
    ]


def run_guardrails(
    doc_type: str,
    extracted: dict[str, Any],
    text: str,
    config: str | GuardrailConfig = "full",
    *,
    xml_fields: dict[str, Any] | None = None,
    source: str | None = None,
    registry: Mapping[str, str] | None = None,
) -> GuardrailResult:
    """Apply the configured checks to one extraction. Never mutates ``extracted``.

    ``xml_fields`` are values read from the document's authoritative XML: when
    given (and ``xml_cross`` is on) they win over the extraction, and every
    difference is recorded as a correction. ``registry`` maps known CNPJs to
    their registered names (company master data) for the name-follows-CNPJ check.
    """
    cfg = get_config(config)
    names = field_names(doc_type)
    record = copy.deepcopy({name: extracted.get(name) for name in names})
    state = checks.State(doc_type=doc_type, record=record, doc=DocumentText(text))
    for enabled, check in _steps(cfg, names, xml_fields, registry):
        if enabled:
            check(state)
    if source is None:
        source = "llm+xml" if xml_fields is not None and cfg.xml_cross else "llm"
    return GuardrailResult(
        doc_type=doc_type,
        record=state.record,
        findings=state.findings,
        provenance=state.provenance,
        config=cfg.name,
        source=source,
    )
