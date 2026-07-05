"""
Custom exceptions for the Honcho application.
"""

import json
from typing import Any, final

from src.config import settings


class HonchoException(Exception):
    """Base exception for all Honcho-specific errors."""

    status_code: int = 500
    detail: str = "An unexpected error occurred"

    def __init__(self, detail: str | None = None, status_code: int | None = None):
        self.detail = detail or self.detail
        self.status_code = status_code or self.status_code
        super().__init__(self.detail)


@final
class ResourceNotFoundException(HonchoException):
    """Exception raised when a requested resource is not found."""

    status_code = 404
    detail = "Resource not found"


@final
class ObserverException(HonchoException):
    """Exception raised when a request tries to add too many observers to a session"""

    status_code = 400

    def __init__(self, session_name: str, extra_count: int):
        self.detail = (
            f"Cannot create session {session_name} with {extra_count} observers. "
            + f"Maximum allowed is {settings.SESSION_OBSERVERS_LIMIT} observers per session. "
            + "Observers are peers with 'observe_others' set to true."
        )
        super().__init__(self.detail)


@final
class ValidationException(HonchoException):
    """Exception raised when validation fails."""

    status_code = 422
    detail = "Validation error"


@final
class PeerNotAllowedException(HonchoException):
    """Raised when peer auto-creation is gated by the workspace's
    ``allowed_ai_peers`` configuration and a request asks to create a new peer
    whose name is not on the allowlist. Already-existing peers are never
    affected by this guardrail.
    """

    status_code = 422

    def __init__(
        self,
        workspace_name: str,
        rejected: list[str],
        allowed: list[str] | None,
    ):
        # `allowed` should never be None/empty here (the gating code only
        # raises when an allowlist was actually configured and at least one
        # name failed it), but be defensive in case the exception is reused.
        if allowed:
            allowed_str = ", ".join(sorted(allowed))
            rejected_str = ", ".join(sorted(rejected))
            detail = (
                f"Workspace '{workspace_name}' restricts new peer "
                f"auto-creation to: [{allowed_str}]. "
                f"Rejected peer name(s): [{rejected_str}]. "
                f"To allow these peers, add them to the workspace's "
                f"configuration.allowed_ai_peers list, or create the peer "
                f"out-of-band first (peer records survive across allowlist "
                f"changes)."
            )
        else:
            rejected_str = ", ".join(sorted(rejected))
            detail = (
                f"Workspace '{workspace_name}' rejected peer "
                f"auto-creation for: [{rejected_str}]."
            )
        super().__init__(detail)
        self.workspace_name = workspace_name
        self.rejected = rejected
        self.allowed = allowed


@final
class ConflictException(HonchoException):
    """Exception raised when there's a resource conflict."""

    status_code = 409
    detail = "Resource conflict"


@final
class AuthenticationException(HonchoException):
    """Exception raised when authentication fails."""

    status_code = 401
    detail = "Authentication failed"


@final
class AuthorizationException(HonchoException):
    """Exception raised when authorization fails."""

    status_code = 403
    detail = "Not authorized to access this resource"


@final
class DisabledException(HonchoException):
    """Exception raised when a feature is disabled."""

    status_code = 405
    detail = "Feature is disabled"


@final
class FilterError(HonchoException):
    """Exception raised when a filter is misconfigured or invalid."""

    status_code = 422
    detail = "Invalid filter configuration"


@final
class UnsupportedFileTypeError(HonchoException):
    status_code = 415
    detail = "Unsupported file type"


@final
class FileTooLargeError(HonchoException):
    status_code = 413
    detail = "File too large"


@final
class FileProcessingError(HonchoException):
    status_code = 500
    detail = "File processing error"


@final
class SurprisalError(HonchoException):
    """Exception raised when surprisal sampling fails during a dream cycle."""

    status_code = 500
    detail = "Surprisal sampling failed"


@final
class SpecialistExecutionError(HonchoException):
    """Exception raised when a specialist fails during dream orchestration."""

    status_code = 500
    detail = "Specialist execution failed"


@final
class VectorStoreError(HonchoException):
    """Exception raised when a vector store operation fails."""

    status_code = 500
    detail = "Vector store operation failed"


class LLMError(Exception):
    """Exception raised when an LLM call fails.

    Accepts arbitrary positional and keyword inputs, normalizes them into a
    JSON-serializable object, and uses the resulting JSON string as the
    exception message. The normalized object is available via ``to_dict()``
    and the ``data`` attribute.

    Positional and keyword inputs are represented as a JSON object. If a
    single positional argument is a mapping and there are no keyword
    arguments, that mapping is used as the root object; otherwise the shape is
    ``{"args": [...], "kwargs": {...}}``. Values that are not natively
    serializable are converted using ``repr``.
    """

    data: dict[str, Any]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        normalized = {"args": list(args), "kwargs": kwargs}
        message = json.dumps(
            normalized, default=self._json_fallback, ensure_ascii=False
        )
        self.data = normalized
        super().__init__(message)

    @staticmethod
    def _json_fallback(value: Any) -> str:
        """Fallback serializer that returns ``repr(value)`` for unsupported types."""
        return repr(value)

    def to_dict(self) -> dict[str, Any]:
        """Return the normalized JSON object for programmatic access."""
        return self.data
