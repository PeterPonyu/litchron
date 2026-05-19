"""Run-state persistence with atomic, locked read/modify/write.

The state is a single JSON document per run, stored under
``<run_dir>/state.json``. All writes go through :meth:`RunStateStore.update`
which takes an exclusive ``fcntl.flock`` on a sibling ``state.lock`` for the
whole RMW window, then writes the document atomically via ``os.replace``.

This is sufficient for MCP-server-driven runs where multiple tool
invocations may concurrently mutate state on a single host. We do not
attempt to be NFS-safe — local POSIX semantics only.
"""
from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------
class CitationFingerprint(BaseModel):
    """Compact pointer to a verified citation; full record lives in cache."""

    scheme: Literal["doi", "pmid"]
    id: str
    verified_at: str  # ISO-8601
    source: Literal["crossref", "pubmed"]
    confidence: float


class DroppedCitation(BaseModel):
    """A citation the verifier rejected, with a machine-readable reason."""

    raw_id: str
    reason: Literal[
        "title_mismatch",
        "year_mismatch",
        "author_mismatch",
        "crossref_404",
        "pubmed_404",
        "network_error",
        "cosine_below_threshold",
    ]
    detail: str
    dropped_at: str  # ISO-8601


QualityFlag = Literal[
    "no_verified_citations",
    "baseline_disagreement_severe",
    "root_cell_ambiguous",
    "preflight_partial",
    "baseline_failure",
]


class SuggestedTool(BaseModel):
    """Next-step hint emitted by the orchestrator for the LLM caller."""

    tool: str
    args: dict[str, Any]
    rationale: str


class DeltaRef(BaseModel):
    """Pointer to a zarr delta layered on top of an AnnData baseline."""

    baseline: str
    keys: list[str]
    zarr_path: str
    applied_at: str  # ISO-8601


# ---------------------------------------------------------------------------
# Root document
# ---------------------------------------------------------------------------
class RunState(BaseModel):
    """Top-level run document persisted to ``state.json``."""

    schema_version: int = 1
    run_id: str
    h5ad_path: str
    started_at: str
    finished_at: Optional[str] = None
    phase: str = "started"

    baselines_done: list[str] = Field(default_factory=list)
    baselines_all_done: bool = False
    llm_ordering_done: bool = False

    citations_verified: list[CitationFingerprint] = Field(default_factory=list)
    citations_dropped: list[DroppedCitation] = Field(default_factory=list)

    comparison_done: bool = False
    latex_compiled: bool = False
    all_green: bool = False

    quality_flags: list[QualityFlag] = Field(default_factory=list)
    suggested_next_tools: list[SuggestedTool] = Field(default_factory=list)
    adata_deltas: list[DeltaRef] = Field(default_factory=list)


def default_state(run_id: str, h5ad_path: str) -> RunState:
    """Construct a fresh ``RunState`` with sensible defaults.

    ``started_at`` is set to the current UTC ISO timestamp. Callers that
    need a deterministic value (tests) should overwrite the field after
    construction.
    """
    from datetime import datetime, timezone

    return RunState(
        run_id=run_id,
        h5ad_path=h5ad_path,
        started_at=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Locking
# ---------------------------------------------------------------------------
@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """Acquire an exclusive ``fcntl.flock`` on ``path`` for the with-block.

    The lock file is created if missing; the underlying fd is closed on
    exit which releases the lock (POSIX guarantees flocks are released
    on close).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Open in append mode so we never truncate the lock file; O_CREAT is
    # implicit on "a".
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------
class RunStateStore:
    """Atomic, locked persistence of a single :class:`RunState` document."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir: Path = Path(run_dir)
        self.state_path: Path = self.run_dir / "state.json"
        self.lock_path: Path = self.run_dir / "state.lock"

    # -- internal helpers ---------------------------------------------------
    def _read_unlocked(self) -> Optional[RunState]:
        """Return the on-disk state or ``None`` if no state file yet."""
        if not self.state_path.exists():
            return None
        with self.state_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return RunState.model_validate(data)

    def _write_unlocked(self, state: RunState) -> None:
        """Atomically write ``state`` to ``state.json`` (tmp + os.replace)."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        payload = state.model_dump(mode="json")
        # NamedTemporaryFile on the same directory so os.replace is atomic
        # across the same filesystem.
        fd, tmp_name = tempfile.mkstemp(
            prefix=".state-", suffix=".json.tmp", dir=str(self.run_dir)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, self.state_path)
        except Exception:
            # Best-effort cleanup of the temp file on error.
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
            raise

    # -- public API ---------------------------------------------------------
    def read(self) -> RunState:
        """Locked read. Raises ``FileNotFoundError`` if state.json missing."""
        with file_lock(self.lock_path):
            state = self._read_unlocked()
        if state is None:
            raise FileNotFoundError(f"No state.json at {self.state_path}")
        return state

    def write(self, state: RunState) -> None:
        """Locked, atomic write of ``state`` to disk."""
        with file_lock(self.lock_path):
            self._write_unlocked(state)

    def update(self, fn: Callable[[RunState], RunState]) -> RunState:
        """Read-modify-write under a single exclusive lock.

        If no state document exists yet, ``fn`` is invoked with a fresh
        default :class:`RunState` (run_id="" / h5ad_path="" — callers that
        need a different default should pre-seed via :meth:`write`, or
        wrap ``fn`` with :func:`functools.partial`).
        """
        with file_lock(self.lock_path):
            current = self._read_unlocked()
            if current is None:
                current = default_state(run_id="", h5ad_path="")
            new_state = fn(current)
            self._write_unlocked(new_state)
            return new_state
