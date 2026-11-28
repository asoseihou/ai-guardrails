"""Orchestration: PDF → text → document type → extraction → guardrails."""

from docai.pipeline.core import Pipeline, PipelineResult

__all__ = ["Pipeline", "PipelineResult"]
