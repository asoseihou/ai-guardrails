"""Human review queue: list and read, enqueue, correct and approve, all behind the token.

Every endpoint requires ``Authorization: Bearer <DOCAI_API_TOKEN>`` and answers
503 when no token is configured; ``DOCAI_REVIEW_PUBLIC_READ=true`` opens the two
read endpoints for a demo with synthetic data. Every mutation lands in the
append-only audit trail.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, File, Path, Query, Response, UploadFile
from pydantic import BaseModel, Field, JsonValue, field_validator

from docai import container
from docai.api.routers.documents import read_pdf_upload, read_xml_upload, safe_filename
from docai.api.security import require_read_access, require_token
from docai.review import reviewer_actor
from docai.settings import get_settings

router = APIRouter(prefix="/review", tags=["review"])

ItemId = Annotated[str, Path(pattern=r"^[0-9a-f]{32}$", description="review item id")]
REVIEWER_PATTERN = r"^[\w .@'-]{1,64}$"
MAX_VALUE_CHARS = 5000


class CorrectionRequest(BaseModel):
    """Set one field to a value printed in the document."""

    field: str = Field(pattern=r"^[a-z_]{1,40}$", description="field name of the schema")
    value: JsonValue = Field(description="new value; validated by the guardrail rules")
    reason: str = Field(min_length=3, max_length=500)
    reviewer: str = Field(pattern=REVIEWER_PATTERN, description="who is correcting")

    @field_validator("value")
    @classmethod
    def _bounded(cls, value: JsonValue) -> JsonValue:
        if len(json.dumps(value)) > MAX_VALUE_CHARS:
            raise ValueError(f"value larger than {MAX_VALUE_CHARS} characters")
        return value


class ApprovalRequest(BaseModel):
    """Approve the record as it stands."""

    reviewer: str = Field(pattern=REVIEWER_PATTERN)
    reason: str = Field("approved after review", min_length=3, max_length=500)


@router.get("/items", dependencies=[Depends(require_read_access)])
def list_items(
    status: Literal["pending", "approved", "all"] = "pending",
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0, le=100_000)] = 0,
) -> dict[str, Any]:
    """Items in the queue (summaries; the document text is never returned)."""
    repo = container.get_review_service().repository
    items = repo.list(None if status == "all" else status, limit, offset)
    return {"items": [item.summary() for item in items], "pending": repo.count_pending()}


@router.get("/items/{item_id}", dependencies=[Depends(require_read_access)])
def get_item(item_id: ItemId) -> dict[str, Any]:
    """One item: record, field status, findings and the audit trail."""
    return container.get_review_service().get(item_id).to_dict()


@router.post("/items", dependencies=[Depends(require_token)])
def enqueue(
    response: Response,
    file: Annotated[UploadFile, File(description="a PDF document")],
    xml: Annotated[UploadFile | None, File(description="optional fiscal XML")] = None,
) -> dict[str, Any]:
    """Run the pipeline and, when it needs review, add the document to the queue (201)."""
    settings = get_settings()
    data = read_pdf_upload(file, settings.max_upload_bytes)
    xml_data = read_xml_upload(xml, settings.max_xml_bytes)
    result = container.get_pipeline().run_pdf(
        data, filename=safe_filename(file.filename), xml_data=xml_data
    )
    item_id, created = container.get_review_service().enqueue(
        result,
        text=result.text,
        document_sha256=hashlib.sha256(data).hexdigest(),
        actor=reviewer_actor("api client"),
    )
    if item_id is None:
        return {"item_id": None, "created": False, "detail": "no review needed"}
    response.status_code = 201 if created else 200
    return {"item_id": item_id, "created": created}


@router.post("/items/{item_id}/corrections", dependencies=[Depends(require_token)])
def correct(item_id: ItemId, request: CorrectionRequest) -> dict[str, Any]:
    """Correct one field; 422 with the reasons when a guardrail rule rejects the value."""
    item = container.get_review_service().correct(
        item_id, request.field, request.value, reviewer=request.reviewer, reason=request.reason
    )
    return item.to_dict()


@router.post("/items/{item_id}/approve", dependencies=[Depends(require_token)])
def approve(item_id: ItemId, request: ApprovalRequest) -> dict[str, Any]:
    """Approve the item (every required field must have a value)."""
    item = container.get_review_service().approve(
        item_id, reviewer=request.reviewer, reason=request.reason
    )
    return item.to_dict()
