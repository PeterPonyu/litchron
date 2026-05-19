"""Filesystem path + run_id helpers shared across the MCP tool layer.

This module is intentionally tiny: it centralizes the three pieces of
information every MCP tool needs to agree on without dragging in any
analysis dependencies.

* :func:`project_root` returns the LitChron repository root (where the
  ``tex/`` template and ``runs/`` collector both live).
* :func:`run_dir` is the canonical ``project_root() / "runs" / run_id``
  directory for a single run.
* :func:`validate_run_id` enforces the ``[A-Za-z0-9-]+`` charset, which
  keeps the ``\\runDir`` LaTeX macro safe by construction.
* :func:`generate_run_id` produces the default
  ``YYYY-MM-DDTHHMMSS-<8hex>`` identifier used when a caller omits one.

No analysis imports here — every MCP tool entry-point can import this
without pulling in scanpy / anndata / rpy2.
"""
from __future__ import annotations

import re
import secrets
from datetime import datetime, timezone
from pathlib import Path

from mcp_litchron.errors import LitchronError

# ---------------------------------------------------------------------------
# Run ID charset
# ---------------------------------------------------------------------------
_RUN_ID_RE: re.Pattern[str] = re.compile(r"^[A-Za-z0-9-]+$")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def project_root() -> Path:
    """Return the LitChron repository root.

    This module lives at ``<root>/litchron/_runtime.py``, so the project
    root is two parents up. The path is *not* resolved against the
    filesystem (it does not need to exist), but the layout is fixed by
    package structure.
    """
    return Path(__file__).resolve().parent.parent


def run_dir(run_id: str) -> Path:
    """Return the canonical ``<project_root>/runs/<run_id>/`` directory.

    The directory is *not* created here — callers (notably
    :func:`tools.start_run`) decide when to materialize it.
    """
    return project_root() / "runs" / run_id


# ---------------------------------------------------------------------------
# Run ID validation + generation
# ---------------------------------------------------------------------------
def validate_run_id(run_id: str) -> None:
    """Raise :class:`LitchronError` if ``run_id`` is not ``[A-Za-z0-9-]+``.

    The charset is constrained because ``run_id`` is interpolated into the
    LaTeX ``\\runDir`` macro by :func:`litchron.report.compile_pdf`.
    Underscores trigger TeX subscript expansion, whitespace breaks the
    ``\\input`` path, and dots collide with the file-extension heuristic.
    Rejecting them at the MCP boundary is the cheapest fix.
    """
    if not isinstance(run_id, str) or not run_id:
        raise LitchronError(
            code="invalid_run_id",
            message="run_id must be a non-empty string",
            hint=(
                "Provide a run_id matching ^[A-Za-z0-9-]+$, or omit the "
                "argument to auto-generate one."
            ),
            retryable=False,
        )
    if not _RUN_ID_RE.match(run_id):
        raise LitchronError(
            code="invalid_run_id",
            message=(
                f"run_id {run_id!r} contains disallowed characters; "
                "only [A-Za-z0-9-] are accepted"
            ),
            hint=(
                "Avoid underscores, dots, whitespace, and other punctuation; "
                "use hyphens as separators (e.g. 'my-run-2026-05-18')."
            ),
            retryable=False,
        )


def generate_run_id() -> str:
    """Return a default ``YYYY-MM-DDTHHMMSS-<8hex>`` identifier.

    The timestamp is UTC; the random suffix is 8 hex chars from
    :func:`secrets.token_hex(4)`. The colons that ``isoformat`` would
    emit between hours/minutes/seconds are stripped so the result is
    LaTeX-safe (no underscore, no colon, no dot, no whitespace).
    """
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y-%m-%dT%H%M%S")
    return f"{ts}-{secrets.token_hex(4)}"


__all__ = [
    "project_root",
    "run_dir",
    "validate_run_id",
    "generate_run_id",
]
