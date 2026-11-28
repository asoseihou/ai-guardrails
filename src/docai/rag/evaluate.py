"""Retrieval evaluation on a small hand-written QA set (hit@k and MRR)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from docai.rag.knowledge import KnowledgeBase

QA_SET = Path(__file__).with_name("qa_set.json")


def load_qa_set(path: Path = QA_SET) -> list[dict[str, Any]]:
    """The hand-written retrieval questions and their expected sections."""
    data: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    return data


def evaluate_retrieval(
    kb: KnowledgeBase, qa_set: list[dict[str, Any]], k: int = 3
) -> dict[str, Any]:
    """hit@1, hit@k and MRR of the retriever on a labelled question set."""
    rows, hits1, hitsk, rr = [], 0, 0, 0.0
    for item in qa_set:
        passages = kb.search(item["question"], k)
        sections = [p.section for p in passages]
        rank = next((i + 1 for i, s in enumerate(sections) if s in item["expected"]), None)
        hits1 += rank == 1
        hitsk += rank is not None
        rr += 1 / rank if rank else 0.0
        rows.append({"question": item["question"], "rank": rank, "top": sections[:1]})
    n = len(qa_set)
    return {
        "n_questions": n,
        "k": k,
        "hit_at_1": round(hits1 / n, 4),
        f"hit_at_{k}": round(hitsk / n, 4),
        "mrr": round(rr / n, 4),
        "rows": rows,
    }
