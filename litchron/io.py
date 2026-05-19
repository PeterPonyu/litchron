"""AnnData IO helpers and modality detection.

Centralizes the few touchpoints LitChron has with the on-disk h5ad
format so the rest of the pipeline can stay AnnData-agnostic.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from anndata import AnnData, read_h5ad

Modality = Literal["scrna", "multiome", "citeseq", "spatial", "unknown"]


def load_h5ad(path: str | Path) -> AnnData:
    """Load an ``.h5ad`` file from disk with a clear error on missing path."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"AnnData file not found: {p!s}. "
            "Provide an absolute path to an existing .h5ad file."
        )
    return read_h5ad(str(p))


def detect_modality(adata: AnnData) -> Modality:
    """Best-effort modality classification.

    Order of checks matters — multiome wins over scRNA when both
    spliced/unspliced layers are present, and spatial wins over scRNA
    when spatial coordinates are present.
    """
    # Explicit uns hint always wins (caller-asserted).
    uns_mod = adata.uns.get("modality") if hasattr(adata, "uns") else None
    if isinstance(uns_mod, str):
        m = uns_mod.lower()
        if m in {"scrna", "multiome", "citeseq", "spatial"}:
            return m  # type: ignore[return-value]

    layers = getattr(adata, "layers", None) or {}
    if "spliced" in layers and "unspliced" in layers:
        return "multiome"

    obsm = getattr(adata, "obsm", None) or {}
    if any(("ADT" in k) or ("protein" in k.lower()) for k in obsm.keys()):
        return "citeseq"
    if "spatial" in obsm:
        return "spatial"

    # If X exists and looks numeric, default to scRNA.
    X = getattr(adata, "X", None)
    if X is not None:
        try:
            # Both sparse and dense expose .dtype via np kind.
            kind = np.dtype(X.dtype).kind  # type: ignore[union-attr]
            if kind in {"f", "i", "u"}:
                return "scrna"
        except (TypeError, AttributeError):
            pass

    return "unknown"


def _mean_density(arr: Any) -> float:
    """Compute fraction of nonzero entries; sparse-aware."""
    # Sparse matrices expose .nnz and .shape.
    if hasattr(arr, "nnz") and hasattr(arr, "shape"):
        total = int(arr.shape[0]) * int(arr.shape[1]) if len(arr.shape) >= 2 else int(arr.shape[0])
        if total == 0:
            return 0.0
        return float(arr.nnz) / float(total)
    try:
        a = np.asarray(arr)
        if a.size == 0:
            return 0.0
        return float(np.count_nonzero(a)) / float(a.size)
    except (TypeError, ValueError):
        return 0.0


def summarize_layers(adata: AnnData) -> dict[str, dict[str, Any]]:
    """Return ``{layer: {dtype, shape, mean_density}}`` for each layer."""
    layers = getattr(adata, "layers", None) or {}
    out: dict[str, dict[str, Any]] = {}
    for name, arr in layers.items():
        try:
            dtype = str(np.dtype(arr.dtype))
        except (TypeError, AttributeError):
            dtype = "unknown"
        shape = tuple(getattr(arr, "shape", ()))
        out[name] = {
            "dtype": dtype,
            "shape": shape,
            "mean_density": _mean_density(arr),
        }
    return out
