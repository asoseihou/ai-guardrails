"""Enforce the layering documented in docs/ARCHITECTURE.md.

Every ``import docai.x`` inside ``src/docai`` is checked (including lazy imports
inside functions): a module may only import from its own layer or below, the
domain and the guardrails are pure, and adapters do not depend on each other
except for the explicitly listed edges.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "docai"

# top-level package/module -> layer (higher may import lower, never the reverse)
LAYERS = {
    "domain": 0,
    "guardrails": 1,
    "settings": 1,
    "observability": 1,
    "extract": 2,
    "storage": 2,
    "classify": 2,
    "rag": 2,
    "synth": 2,
    "pipeline": 3,
    "review": 3,
    "eval": 3,
    "container": 3,
    "api": 4,
    "cli": 4,
    "mcp_server": 4,
    "tools": 4,
    "__main__": 4,
}
PURE = {
    "domain": {"domain"},
    "guardrails": {"domain", "guardrails"},
    "settings": {"settings"},
    "observability": {"observability"},
}
# adapter -> other adapters it may use (the simulated LLM reuses the synthetic-data helpers)
ADAPTER_EDGES = {("extract", "synth")}


def _component(module: str) -> str | None:
    parts = module.split(".")
    if parts[0] != "docai":
        return None
    return parts[1] if len(parts) > 1 else ""  # "" = the root package (version only)


def _imports(path: Path) -> list[tuple[int, str]]:
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.append((node.lineno, node.module))
    return found


def _source_component(path: Path) -> str:
    rel = path.relative_to(SRC)
    return rel.parts[0].removesuffix(".py")


MODULES = [p for p in sorted(SRC.rglob("*.py")) if p.relative_to(SRC).parts[0] != "__init__.py"]


def test_every_component_has_a_layer():
    assert {_source_component(p) for p in MODULES} <= set(LAYERS)


@pytest.mark.parametrize("path", MODULES, ids=lambda p: str(p.relative_to(SRC)))
def test_layering(path):
    source = _source_component(path)
    violations = []
    for lineno, module in _imports(path):
        target = _component(module)
        if target in (None, "", source):
            continue
        if source in PURE and target not in PURE[source]:
            violations.append(f"{lineno}: {source} must stay pure but imports {module}")
        elif LAYERS[target] > LAYERS[source]:
            violations.append(f"{lineno}: {source} (layer {LAYERS[source]}) imports {module}")
        elif LAYERS[target] == LAYERS[source] == 2 and (source, target) not in ADAPTER_EDGES:
            violations.append(f"{lineno}: adapter {source} imports adapter {module}")
    assert not violations, "\n".join(violations)


def test_the_checker_catches_a_violation(tmp_path):
    bad = tmp_path / "x.py"
    bad.write_text("def f():\n    from docai.api.app import app\n")
    assert _imports(bad) == [(2, "docai.api.app")]
    assert LAYERS[_component("docai.api.app")] > LAYERS["domain"]
