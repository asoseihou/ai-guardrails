"""Bearer-token authentication for the review queue.

Secure by default: when ``DOCAI_API_TOKEN`` is not set, every protected endpoint
answers 503. Reads of the queue need the token too, unless
``DOCAI_REVIEW_PUBLIC_READ=true``. Tokens are compared in constant time
(:func:`hmac.compare_digest`).
A single shared token identifies *a* trusted operator, not a person; the
reviewer name recorded in the audit trail is self-declared (see SECURITY.md).
"""

from __future__ import annotations

import hmac
from typing import Annotated

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from docai.settings import get_settings

_bearer = HTTPBearer(auto_error=False, description="DOCAI_API_TOKEN")


def token_matches(given: str, expected: str) -> bool:
    """Constant-time comparison of two tokens."""
    return hmac.compare_digest(given.encode(), expected.encode())


def require_token(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> None:
    """FastAPI dependency: 503 when no token is configured, 401 on a missing/wrong token."""
    expected = get_settings().api_token
    if expected is None:
        raise HTTPException(
            503, "the review API is disabled: set DOCAI_API_TOKEN on the server to enable it"
        )
    if credentials is None or not token_matches(
        credentials.credentials, expected.get_secret_value()
    ):
        raise HTTPException(
            401, "missing or invalid bearer token", headers={"WWW-Authenticate": "Bearer"}
        )


def require_read_access(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> None:
    """Reads need the token unless ``DOCAI_REVIEW_PUBLIC_READ=true``.

    The queue holds document data, so it is private by default.
    """
    if get_settings().review_public_read:
        return
    require_token(credentials)


TokenRequired = Depends(require_token)
