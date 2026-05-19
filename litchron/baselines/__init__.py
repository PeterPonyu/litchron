"""Baseline trajectory inference dispatch layer for LitChron.

This package exposes a small public surface:

* :data:`BaselineName` — the closed set of method names.
* :data:`R_BACKED` / :data:`PYTHON_BACKED` — partition of methods by runtime.
* :class:`BaselineSpec` — host-availability descriptor.
* :class:`BaselineResult` — uniform return type from every baseline.
* :func:`available_baselines` — filters methods by modality + installed deps.
* :func:`run` — top-level dispatch (subprocess vs. in-process).

**Hybrid-isolation invariant.** R-bridged baselines (``monocle3``,
``slingshot_r``) are always run in a subprocess via
:mod:`litchron.baselines._r_runner`. ``rpy2`` MUST NOT be imported from
any module other than ``_r_runner`` so that a SIGSEGV in libR cannot
take down the MCP server process.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from mcp_litchron.errors import BaselineFailure

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------
BaselineName = Literal[
    "monocle3",
    "slingshot_r",
    "paga",
    "palantir",
    "scvelo",
    "pyslingshot",
]

R_BACKED: set[str] = {"monocle3", "slingshot_r"}
PYTHON_BACKED: set[str] = {"paga", "palantir", "scvelo", "pyslingshot"}


# ---------------------------------------------------------------------------
# Public models
# ---------------------------------------------------------------------------
class BaselineSpec(BaseModel):
    """Host-availability descriptor for a single baseline method."""

    name: str
    runtime: Literal["in_process", "subprocess"]
    requires: list[str] = Field(default_factory=list)
    available: bool = False
    reason: Optional[str] = None


class BaselineResult(BaseModel):
    """Uniform return type from every baseline.

    ``ordering_path`` is an absolute path to a parquet file with at
    minimum columns ``cell_id`` and ``pseudotime``. ``adata_delta_keys``
    enumerates the AnnData keys mutated by the baseline (used by the
    cache replay protocol).
    """

    method: str
    ordering_path: str
    lineage_edges: Optional[list[tuple[str, str]]] = None
    root_cell: Optional[str] = None
    figure_path: Optional[str] = None
    delta_zarr_path: Optional[str] = None
    adata_delta_keys: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Availability detection
# ---------------------------------------------------------------------------
def _try_import(module: str) -> bool:
    """Return True iff ``module`` imports cleanly in the current interpreter."""
    try:
        __import__(module)
        return True
    except Exception:  # ImportError, plus runtime errors during init.
        return False


def _scvelo_layers_ok(adata: Any) -> tuple[bool, Optional[str]]:
    """Return (ok, reason). scVelo requires Ms/Mu or spliced+unspliced."""
    layers = getattr(adata, "layers", None) or {}
    if "Ms" in layers and "Mu" in layers:
        return True, None
    if "spliced" in layers and "unspliced" in layers:
        return True, None
    return False, "scvelo requires Ms/Mu or spliced+unspliced layers"


def available_baselines(adata: Any, preflight: Any) -> list[BaselineSpec]:
    """Return a list of :class:`BaselineSpec`, one per known method.

    The ``available`` field reflects host-side feasibility:

    * Python baselines require the corresponding package to import.
    * R baselines require ``preflight.rscript`` to be non-None.
    * scVelo additionally requires Ms/Mu (or spliced+unspliced) layers.

    Modality filtering: scVelo is gated on spliced/Ms layers; the other
    Python baselines apply to any single-cell modality with clusters and
    embeddings.
    """
    specs: list[BaselineSpec] = []

    # --- Python baselines -------------------------------------------------
    for method, module in (
        ("paga", "scanpy"),
        ("palantir", "palantir"),
        ("pyslingshot", "pyslingshot"),
    ):
        ok = _try_import(module)
        specs.append(
            BaselineSpec(
                name=method,
                runtime="in_process",
                requires=[module],
                available=ok,
                reason=None if ok else f"{module} not importable",
            )
        )

    # scVelo: extra layer gate.
    sv_import = _try_import("scvelo")
    sv_layers_ok, sv_layer_reason = _scvelo_layers_ok(adata)
    sv_ok = sv_import and sv_layers_ok
    sv_reason: Optional[str]
    if not sv_import:
        sv_reason = "scvelo not importable"
    elif not sv_layers_ok:
        sv_reason = sv_layer_reason
    else:
        sv_reason = None
    specs.append(
        BaselineSpec(
            name="scvelo",
            runtime="in_process",
            requires=["scvelo"],
            available=sv_ok,
            reason=sv_reason,
        )
    )

    # --- R baselines ------------------------------------------------------
    rscript = getattr(preflight, "rscript", None)
    r_ok = rscript is not None
    r_reason = None if r_ok else "Rscript not found on PATH"
    for method, pkg in (("monocle3", "monocle3"), ("slingshot_r", "slingshot")):
        specs.append(
            BaselineSpec(
                name=method,
                runtime="subprocess",
                requires=["Rscript", f"R package: {pkg}"],
                available=r_ok,
                reason=r_reason,
            )
        )

    return specs


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
def run(
    method: str,
    run_id: str,
    h5ad_path: str,
    run_dir: Path,
    adata: Any,
) -> BaselineResult:
    """Dispatch ``method`` to its runtime and return the :class:`BaselineResult`.

    Raises
    ------
    BaselineFailure
        On any error (unknown method, import failure inside an in-process
        baseline, subprocess crash / non-zero exit, JSON parse error,
        etc.). The exception's ``method`` attribute records the requested
        baseline name.
    """
    run_dir = Path(run_dir)

    if method in R_BACKED:
        if method == "monocle3":
            from .monocle3 import run_monocle3

            return run_monocle3(run_id=run_id, h5ad_path=h5ad_path, run_dir=run_dir)
        if method == "slingshot_r":
            from .slingshot_r import run_slingshot_r

            return run_slingshot_r(run_id=run_id, h5ad_path=h5ad_path, run_dir=run_dir)

    if method in PYTHON_BACKED:
        if method == "paga":
            from .paga import run_paga

            return run_paga(adata=adata, run_dir=run_dir)
        if method == "palantir":
            from .palantir import run_palantir

            return run_palantir(adata=adata, run_dir=run_dir)
        if method == "scvelo":
            from .scvelo import run_scvelo

            return run_scvelo(adata=adata, run_dir=run_dir)
        if method == "pyslingshot":
            from .pyslingshot import run_pyslingshot

            return run_pyslingshot(adata=adata, run_dir=run_dir)

    raise BaselineFailure(
        code="unknown_method",
        message=f"Unknown baseline method: {method!r}",
        hint=(
            "Use one of: monocle3, slingshot_r, paga, palantir, scvelo, "
            "pyslingshot."
        ),
        retryable=False,
        method=method,
    )


__all__ = [
    "BaselineName",
    "R_BACKED",
    "PYTHON_BACKED",
    "BaselineSpec",
    "BaselineResult",
    "available_baselines",
    "run",
]
