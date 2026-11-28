"""A seeded *simulated* LLM.

It reads the ground truth of a document and injects realistic extraction
errors at configurable rates. This is what lets the whole project run offline
and makes the guardrail evaluation reproducible. Metrics produced with it are
**simulated-LLM metrics**, not measurements of a real model.

Error modes (all observed in real document-extraction systems):

* ``cnpj_digit_swap``       two adjacent digits transposed or one digit misread
* ``hallucinated_party``    issuer name + CNPJ copied from another document
* ``number_from_filename``  document number taken from the file name
* ``wrong_decimal``         total value off by a factor of 10 or 100
* ``dropped_field``         a field silently omitted
* ``key_typo``              one digit of the access key misread, or (CT-e) the key
                            of a referenced NF-e returned instead of the CT-e key
* ``date_swap``             day and month swapped
* ``component_label_swap``  CT-e component values attached to the wrong labels
* ``party_role_swap``       two complete parties swapped (CT-e sender/recipient,
                            NFS-e provider/taker). Both names and CNPJs are printed
                            and valid: only the role-anchor check can see it
* ``name_from_neighbour``   issuer name completed with a similar company's name
                            (the CNPJ stays right)
* ``reference_from_address`` service-order reference taken from an address number
* ``weight_from_attachment`` CT-e gross weight read from the attached DANFE page
* ``due_date_from_duplicate`` CT-e due date read from the attached NF-e's duplicates
* ``payer_into_recipient``  CT-e payer (TOMADOR) box copied into the recipient
* ``referenced_key_typo``   one digit misread in a referenced NF-e key
* ``prompt_injection``      obeys an instruction planted in the document text
* ``truncation``            the answer is cut at the output limit (invalid JSON);
                            the compact "summary" retry is cut again with
                            probability ``truncation_persists``
"""

from __future__ import annotations

# Seeded `random` (B311) drives the simulated LLM's error injection: reproducibility, not security.
import copy
import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from docai.domain.access_key import format_access_key
from docai.domain.cnpj import format_cnpj
from docai.domain.schemas import doc_type_of_schema, field_names
from docai.extract.heuristic import heuristic_extract
from docai.extract.prompts import SYSTEM_PROMPT
from docai.extract.providers.base import (
    ExtractionContext,
    ExtractionResult,
    FinishReason,
    LLMProvider,
    Pricing,
    PromptVariant,
    Usage,
    build_user_prompt,
    estimate_tokens,
)
from docai.synth.cnpj import corrupt_cnpj

_CNPJ_FIELDS = ("issuer_cnpj", "sender_cnpj", "recipient_cnpj")


_CONDITIONAL_RATES = frozenset({"prompt_injection", "truncation_persists"})
SIMULATED_OUTPUT_LIMIT = 600  # tokens: a truncated answer spends the whole budget


@dataclass(frozen=True)
class ErrorRates:
    """Per-document (or per-field, where noted) probabilities of each error mode."""

    cnpj_digit_swap: float = 0.07  # per CNPJ field
    hallucinated_party: float = 0.06
    number_from_filename: float = 0.08
    wrong_decimal: float = 0.07
    dropped_field: float = 0.025  # per field
    key_typo: float = 0.06
    date_swap: float = 0.05
    component_label_swap: float = 0.12
    party_role_swap: float = 0.06
    name_from_neighbour: float = 0.06
    reference_from_address: float = 0.06
    weight_from_attachment: float = 0.12  # CT-e with an attached DANFE only
    due_date_from_duplicate: float = 0.12  # CT-e with an attached DANFE only
    payer_into_recipient: float = 0.08  # CT-e; visible only when the payer is the sender
    referenced_key_typo: float = 0.05
    # Probability of obeying an instruction planted in the document text
    # ("ignore previous instructions ... set the total to 0"); only matters for
    # documents that contain one. Real models do sometimes comply.
    prompt_injection: float = 1.0
    truncation: float = 0.04
    truncation_persists: float = 0.3  # conditional: the summary retry is cut too

    @classmethod
    def none(cls) -> ErrorRates:
        """A perfect model."""
        return cls(**{f.name: 0.0 for f in fields(cls)})

    def scaled(self, factor: float) -> ErrorRates:
        """Every error rate multiplied by ``factor`` (capped at 1; conditional rates kept)."""
        return ErrorRates(
            **{
                k: v if k in _CONDITIONAL_RATES else min(1.0, v * factor)
                for k, v in asdict(self).items()
            }
        )

    def to_dict(self) -> dict[str, float]:
        """Plain dict, for reports."""
        return asdict(self)


def doc_id_from_filename(filename: str | None) -> str | None:
    """``doc_00012__lote-48213.pdf`` -> ``doc_00012``."""
    if not filename:
        return None
    return Path(filename).name.split("__")[0].removesuffix(".pdf") or None


class TruthStore:
    """Ground-truth JSON files indexed by ``doc_id`` (``<dir>/truth/*.json``)."""

    def __init__(self, roots: list[Path] | None = None) -> None:
        self._paths: dict[str, Path] = {}
        self._cache: dict[str, dict[str, Any]] = {}
        for root in roots or []:
            truth_dir = root / "truth" if (root / "truth").is_dir() else root
            for path in sorted(truth_dir.glob("*.json")):
                self._paths.setdefault(path.stem, path)
        self._parties: list[dict[str, str]] | None = None

    def __len__(self) -> int:
        return len(self._paths)

    def get(self, doc_id: str | None) -> dict[str, Any] | None:
        """Ground truth of a document, or ``None``."""
        if not doc_id or doc_id not in self._paths:
            return None
        if doc_id not in self._cache:
            self._cache[doc_id] = json.loads(self._paths[doc_id].read_text(encoding="utf-8"))
        return self._cache[doc_id]

    def parties(self) -> list[dict[str, str]]:
        """Every (name, CNPJ) pair seen in the corpus — the hallucination pool."""
        if self._parties is None:
            seen: dict[str, dict[str, str]] = {}
            for doc_id in sorted(self._paths):
                doc = self.get(doc_id) or {}
                f = doc.get("fields", {})
                for role in ("issuer", "sender", "recipient"):
                    cnpj, name = f.get(f"{role}_cnpj"), f.get(f"{role}_name")
                    if cnpj and name:
                        seen.setdefault(cnpj, {"name": name, "cnpj": cnpj})
            self._parties = [seen[k] for k in sorted(seen)]
        return self._parties


class SimulatedLLMProvider(LLMProvider):
    """Ground truth + seeded, realistic noise. Never calls a network."""

    name = "simulated"
    simulated = True

    def __init__(
        self,
        truth: TruthStore | None = None,
        rates: ErrorRates | None = None,
        *,
        seed: int = 7,
        model: str = "simulated-llm-v1",
        pricing: Pricing | None = None,
    ) -> None:
        super().__init__(model=model, pricing=pricing)
        self.truth = truth or TruthStore()
        self.rates = rates if rates is not None else ErrorRates()
        self.seed = seed

    def complete(
        self,
        text: str,
        schema: type[BaseModel],
        context: ExtractionContext | None,
        variant: PromptVariant = PromptVariant.STANDARD,
    ) -> ExtractionResult:
        """Ground truth (or the regex baseline) plus seeded errors; never touches a network."""
        doc_type = doc_type_of_schema(schema)
        context = context or ExtractionContext()
        doc_id = context.doc_id or doc_id_from_filename(context.filename)
        data, meta = self._answer(text, doc_type, doc_id, context.filename)
        prompt = SYSTEM_PROMPT + build_user_prompt(text, doc_type, context, variant)
        input_tokens = estimate_tokens(prompt)
        output_tokens = estimate_tokens(json.dumps(data, ensure_ascii=False))
        if variant == PromptVariant.SUMMARY:
            output_tokens = max(1, round(output_tokens * 0.6))  # compact JSON
        seed_key = doc_id or hashlib.sha256(text.encode()).hexdigest()[:16]
        if self._truncated(seed_key, doc_type, variant):
            output_tokens = SIMULATED_OUTPUT_LIMIT
            meta = {**meta, "injected_errors": [], "truncated": True}
            finish, error, result_data = FinishReason.LENGTH, "cut at the output limit", None
        else:
            finish, error, result_data = FinishReason.STOP, None, data
        usage = Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_s=round(0.35 + output_tokens / 90 + input_tokens / 9000, 3),
            cost_usd=self.pricing.cost(input_tokens, output_tokens),
            simulated=True,
        )
        return ExtractionResult(
            data=result_data,
            usage=usage,
            provider=self.name,
            model=self.model,
            meta=meta,
            finish_reason=finish,
            error=error,
        )

    def _answer(
        self, text: str, doc_type: str, doc_id: str | None, filename: str | None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """The (corrupted) fields the simulated model would return, and metadata."""
        truth_doc = self.truth.get(doc_id)
        names = field_names(doc_type)
        if truth_doc is not None:
            gold = truth_doc.get("fields", {})
            base = {name: copy.deepcopy(gold.get(name)) for name in names}
            misrouted = truth_doc["doc_type"] != doc_type
            source = "ground_truth_misrouted" if misrouted else "ground_truth"
            layout = truth_doc.get("layout", {})
        else:
            base, source, layout = heuristic_extract(text, doc_type), "regex_baseline", {}
        seed_key = doc_id or hashlib.sha256(text.encode()).hexdigest()[:16]
        rng = random.Random(f"{self.seed}|{seed_key}|{doc_type}")  # nosec B311
        data, injected = inject_errors(
            base,
            doc_type,
            rng,
            self.rates,
            filename=filename,
            party_pool=self.truth.parties(),
            layout=layout,
        )
        data = _llm_style(data, rng)
        injection_rng = random.Random(f"{self.seed}|{seed_key}|{doc_type}|injection")  # nosec B311
        injected += obey_injected_instruction(text, data, injection_rng, self.rates)
        return data, {"source": source, "injected_errors": injected}

    def _truncated(self, seed_key: str, doc_type: str, variant: PromptVariant) -> bool:
        """Deterministic per document: is this call's answer cut at the output limit?"""
        rng = random.Random(f"{self.seed}|{seed_key}|{doc_type}|truncation")  # nosec B311
        cut_first, cut_again = rng.random(), rng.random()
        if cut_first >= self.rates.truncation:
            return False
        return variant == PromptVariant.STANDARD or cut_again < self.rates.truncation_persists


# --------------------------------------------------------------------- errors
class _Injector:
    """Applies each error mode in a fixed order to a copy of the fields.

    Every random draw happens unconditionally and in a fixed order, so changing
    one rate does not reshuffle the errors of the other modes.
    """

    def __init__(self, fields_in: dict[str, Any], rng: random.Random, rates: ErrorRates) -> None:
        self.out = copy.deepcopy(fields_in)
        self.rng = rng
        self.rates = rates
        self.injected: list[dict[str, Any]] = []

    def log(self, kind: str, field: str, before: Any, after: Any) -> None:
        self.injected.append({"type": kind, "field": field, "before": before, "after": after})

    def hallucinated_party(self, party_pool: list[dict[str, str]]) -> None:
        """Issuer name + CNPJ copied from another document of the corpus."""
        out = self.out
        draw, pick = self.rng.random(), self.rng.random()
        if draw >= self.rates.hallucinated_party or not party_pool or not out.get("issuer_cnpj"):
            return
        candidates = [p for p in party_pool if p["cnpj"] != out["issuer_cnpj"]]
        if candidates:
            other = candidates[int(pick * len(candidates))]
            self.log("hallucinated_party", "issuer_name", out.get("issuer_name"), other["name"])
            self.log("hallucinated_party", "issuer_cnpj", out["issuer_cnpj"], other["cnpj"])
            out["issuer_name"], out["issuer_cnpj"] = other["name"], other["cnpj"]

    def number_from_filename(self, filename: str | None) -> None:
        """Document number copied from the file name."""
        draw = self.rng.random()
        tag_numbers = _filename_numbers(filename)
        if draw < self.rates.number_from_filename and tag_numbers and self.out.get("number"):
            candidate = str(int(tag_numbers[-1]))
            if candidate != self.out["number"]:
                self.log("number_from_filename", "number", self.out["number"], candidate)
                self.out["number"] = candidate

    def wrong_decimal(self) -> None:
        """Total off by a factor of 10 or 100."""
        draw, factor = self.rng.random(), self.rng.choice([10, 0.1, 100])
        total = self.out.get("total_value")
        if draw < self.rates.wrong_decimal and total:
            wrong = round(total * factor, 2)
            if wrong not in (total, 0):
                self.log("wrong_decimal", "total_value", total, wrong)
                self.out["total_value"] = wrong

    def cnpj_digit_swap(self) -> None:
        """Per CNPJ field: two digits transposed or one misread."""
        for name in _CNPJ_FIELDS:
            draw = self.rng.random()
            sub_rng = random.Random(self.rng.random())  # nosec B311
            if self.out.get(name) and draw < self.rates.cnpj_digit_swap:
                wrong = corrupt_cnpj(self.out[name], sub_rng)
                self.log("cnpj_digit_swap", name, self.out[name], wrong)
                self.out[name] = wrong

    def key_typo(self, referenced_keys: list[str]) -> None:
        """One digit of the key misread, or (CT-e) a referenced NF-e key returned."""
        rng = self.rng
        draw, mode = rng.random(), rng.random()
        position, digit = rng.randrange(44), rng.randrange(1, 10)
        key = self.out.get("access_key")
        if draw >= self.rates.key_typo or not key:
            return
        if referenced_keys and mode < 0.4:
            wrong = referenced_keys[0]
        else:
            new_digit = str((int(key[position]) + digit) % 10)
            wrong = key[:position] + new_digit + key[position + 1 :]
        self.log("key_typo", "access_key", key, wrong)
        self.out["access_key"] = wrong

    def date_swap(self) -> None:
        """Day and month swapped."""
        draw = self.rng.random()
        issued = self.out.get("issue_date")
        if draw < self.rates.date_swap and issued:
            year, month, day = issued.split("-")
            if int(day) <= 12 and day != month:
                wrong = f"{year}-{day}-{month}"
                self.log("date_swap", "issue_date", issued, wrong)
                self.out["issue_date"] = wrong

    def component_label_swap(self) -> None:
        """CT-e component values attached to the wrong labels."""
        draw, i_pick, j_pick = self.rng.random(), self.rng.random(), self.rng.random()
        comps = self.out.get("freight_components")
        if draw >= self.rates.component_label_swap or not comps or len(comps) < 2:
            return
        i = int(i_pick * len(comps))
        j = (i + 1 + int(j_pick * (len(comps) - 1))) % len(comps)
        if comps[i]["value"] != comps[j]["value"]:
            before = copy.deepcopy(comps)
            comps[i]["value"], comps[j]["value"] = comps[j]["value"], comps[i]["value"]
            self.log("component_label_swap", "freight_components", before, copy.deepcopy(comps))

    def party_role_swap(self, doc_type: str) -> None:
        """Two complete parties swapped (consistent, printed, valid: a blind spot)."""
        draw = self.rng.random()
        pair = {"cte": ("sender", "recipient"), "nfse": ("issuer", "recipient")}.get(doc_type)
        if draw >= self.rates.party_role_swap or not pair:
            return
        a, b = pair
        for suffix in ("name", "cnpj"):
            fa, fb = f"{a}_{suffix}", f"{b}_{suffix}"
            if self.out.get(fa) and self.out.get(fb):
                self.log("party_role_swap", fa, self.out[fa], self.out[fb])
                self.log("party_role_swap", fb, self.out[fb], self.out[fa])
                self.out[fa], self.out[fb] = self.out[fb], self.out[fa]

    def name_from_neighbour(self, party_pool: list[dict[str, str]]) -> None:
        """Issuer name replaced by a plausible neighbour (shares a word), CNPJ kept."""
        draw, pick = self.rng.random(), self.rng.random()
        name = self.out.get("issuer_name")
        if draw >= self.rates.name_from_neighbour or not name or not party_pool:
            return
        words = set(name.split()) - {"Ltda", "S.A.", "ME", "EIRELI"}
        others = [p["name"] for p in party_pool if p["name"] != name]
        neighbours = [n for n in others if words & set(n.split())] or others
        if neighbours:
            wrong = neighbours[int(pick * len(neighbours))]
            self.log("name_from_neighbour", "issuer_name", name, wrong)
            self.out["issuer_name"] = wrong

    def reference_from_address(self, layout: dict[str, Any]) -> None:
        """Service-order reference taken from the number of a street address."""
        draw = self.rng.random()
        party = layout.get("issuer") or {}
        numbers = re.findall(r"\d+", party.get("street", ""))
        if draw < self.rates.reference_from_address and "service_order" in self.out and numbers:
            self.log(
                "reference_from_address", "service_order", self.out["service_order"], numbers[-1]
            )
            self.out["service_order"] = numbers[-1]

    def attachment_traps(self, layout: dict[str, Any]) -> None:
        """CT-e weight / due date taken from the attached DANFE instead of the CT-e box."""
        weight_draw, due_draw = self.rng.random(), self.rng.random()
        attachment = layout.get("attachment")
        if not attachment:
            return
        if weight_draw < self.rates.weight_from_attachment and "gross_weight" in self.out:
            self.log(
                "weight_from_attachment",
                "gross_weight",
                self.out["gross_weight"],
                attachment["gross_weight"],
            )
            self.out["gross_weight"] = attachment["gross_weight"]
        if due_draw < self.rates.due_date_from_duplicate and "due_date" in self.out:
            self.log(
                "due_date_from_duplicate", "due_date", self.out["due_date"], attachment["due_date"]
            )
            self.out["due_date"] = attachment["due_date"]

    def payer_into_recipient(self, layout: dict[str, Any]) -> None:
        """The TOMADOR (payer) box copied into the recipient fields."""
        draw = self.rng.random()
        payer = layout.get(layout.get("taker", ""), {})
        if draw >= self.rates.payer_into_recipient or not payer or "recipient_cnpj" not in self.out:
            return
        if payer.get("cnpj") != self.out.get("recipient_cnpj"):
            for suffix in ("name", "cnpj"):
                name = f"recipient_{suffix}"
                self.log("payer_into_recipient", name, self.out[name], payer[suffix])
                self.out[name] = payer[suffix]

    def referenced_key_typo(self) -> None:
        """One digit misread in the first referenced NF-e key."""
        draw, position, digit = self.rng.random(), self.rng.randrange(44), self.rng.randrange(1, 10)
        keys = self.out.get("referenced_keys")
        if draw < self.rates.referenced_key_typo and keys:
            key = keys[0]
            wrong = key[:position] + str((int(key[position]) + digit) % 10) + key[position + 1 :]
            self.log("referenced_key_typo", "referenced_keys", list(keys), [wrong, *keys[1:]])
            self.out["referenced_keys"] = [wrong, *keys[1:]]

    def dropped_field(self) -> None:
        """Per field: silently omitted."""
        for name in sorted(self.out):
            draw = self.rng.random()
            if self.out[name] is not None and draw < self.rates.dropped_field:
                self.log("dropped_field", name, self.out[name], None)
                self.out[name] = None


def inject_errors(
    fields_in: dict[str, Any],
    doc_type: str,
    rng: random.Random,
    rates: ErrorRates,
    *,
    filename: str | None = None,
    party_pool: list[dict[str, str]] | None = None,
    layout: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return a corrupted copy of ``fields_in`` and the list of injected errors."""
    layout = layout or {}
    injector = _Injector(fields_in, rng, rates)
    injector.hallucinated_party(party_pool or [])
    injector.number_from_filename(filename)
    injector.wrong_decimal()
    injector.cnpj_digit_swap()
    injector.key_typo(layout.get("referenced_keys") or [])
    injector.date_swap()
    injector.component_label_swap()
    injector.party_role_swap(doc_type)
    injector.name_from_neighbour(party_pool or [])
    injector.reference_from_address(layout)
    injector.attachment_traps(layout)
    if doc_type == "cte":
        injector.payer_into_recipient(layout)
    injector.referenced_key_typo()
    injector.dropped_field()
    return injector.out, injector.injected


_INJECTION = re.compile(r"ignore (?:all )?(?:previous|prior|the above) instructions", re.I)
_SET_TOTAL = re.compile(r"set (?:the )?total(?: value)? to (\d[\d.,]*)", re.I)


def obey_injected_instruction(
    text: str, data: dict[str, Any], rng: random.Random, rates: ErrorRates
) -> list[dict[str, Any]]:
    """Simulate a model that follows an instruction planted in the document (prompt injection).

    Mutates ``data`` in place and returns the injected-error log entries. The
    guardrails, not the prompt, are what must catch the effect.
    """
    draw = rng.random()
    if not _INJECTION.search(text) or draw >= rates.prompt_injection or "total_value" not in data:
        return []
    match = _SET_TOTAL.search(text)
    wrong = float(match.group(1).replace(".", "").replace(",", ".")) if match else 0.0
    entry = {
        "type": "prompt_injection",
        "field": "total_value",
        "before": data["total_value"],
        "after": wrong,
    }
    data["total_value"] = wrong
    return [entry]


def _filename_numbers(filename: str | None) -> list[str]:
    if not filename:
        return []
    tag = Path(filename).stem.split("__")[-1]
    return re.findall(r"\d{3,}", tag)


def _llm_style(data: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Representation noise that is *not* an error (formatting a real model varies)."""
    out = dict(data)
    for name in _CNPJ_FIELDS:
        value = out.get(name)
        if value and len(value) == 14 and value.isdigit() and rng.random() < 0.5:
            out[name] = format_cnpj(value)
    key = out.get("access_key")
    if key and len(key) == 44 and rng.random() < 0.2:
        out["access_key"] = format_access_key(key)
    number = out.get("number")
    if number and number.isdigit() and rng.random() < 0.2:
        out["number"] = number.zfill(9)
    return out
