"""Structured error types for the LitChron MCP server.

Every MCP tool catches generic exceptions and returns an :class:`ErrorResult`
(JSON-serializable) so the LLM can read a code/hint/retryable triple instead
of a stack trace. The underlying exception hierarchy is:

``LitchronError``
    Base class with ``code``, ``message``, ``hint``, ``retryable``.
``BaselineFailure``
    Subclass with ``method`` and ``stderr`` fields for trajectory-baseline
    crashes (R subprocess SIGSEGV, Python baseline RuntimeError, ...).
``PreflightFailure``
    Subclass raised when a required system dep (pandoc, latexmk, Rscript) is
    missing or below the minimum version.
``ComparisonProtocolError``
    Subclass raised by :mod:`litchron.compare` when a comparison cell of
    the (llm_shape, baseline_shape) matrix is undefined.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class LitchronError(Exception):
    """Base structured exception for LitChron tooling.

    Attributes
    ----------
    code
        Short machine-readable identifier (e.g. ``"invalid_run_id"``).
    message
        Human-readable single-sentence description.
    hint
        Actionable next step for the caller (LLM or operator).
    retryable
        Whether the caller can retry the same call and reasonably hope
        for a different outcome.
    """

    def __init__(
        self,
        code: str,
        message: str,
        hint: str,
        retryable: bool,
    ) -> None:
        super().__init__(message)
        self.code: str = code
        self.message: str = message
        self.hint: str = hint
        self.retryable: bool = retryable

    def to_result(self) -> "ErrorResult":
        """Return a JSON-serializable :class:`ErrorResult` for tool returns."""
        return ErrorResult(
            code=self.code,
            message=self.message,
            hint=self.hint,
            retryable=self.retryable,
        )


class BaselineFailure(LitchronError):
    """Raised when a trajectory baseline (R subprocess or Python) fails.

    Extra fields
    ------------
    method
        Baseline name (e.g. ``"monocle3"``, ``"paga"``).
    stderr
        Captured stderr from the failing subprocess, if any. ``None`` for
        in-process failures.
    """

    def __init__(
        self,
        code: str,
        message: str,
        hint: str,
        retryable: bool,
        method: str,
        stderr: Optional[str] = None,
    ) -> None:
        super().__init__(code=code, message=message, hint=hint, retryable=retryable)
        self.method: str = method
        self.stderr: Optional[str] = stderr


class PreflightFailure(LitchronError):
    """Raised when a required system dependency is missing or too old."""


class ComparisonProtocolError(LitchronError):
    """Raised when (llm_shape, baseline_shape) yields an undefined cell."""


# ---------------------------------------------------------------------------
# Pydantic result model
# ---------------------------------------------------------------------------
class ErrorResult(BaseModel):
    """JSON-serializable wrapper for tool error returns."""

    code: str
    message: str
    hint: str
    retryable: bool
