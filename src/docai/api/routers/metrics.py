"""Run history for the cost & quality dashboard (aggregate metrics only: public)."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Query

from docai import container

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/runs")
def list_runs(limit: Annotated[int, Query(ge=1, le=500)] = 100) -> dict[str, Any]:
    """Recorded evaluation runs, oldest first."""
    return {"runs": [run.to_dict() for run in container.get_run_repository().list(limit)]}
