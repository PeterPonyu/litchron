"""LLM-driven continuous per-cell pseudotime.

The LLM proposes a per-cluster rank (e.g., ``LT-HSC=1, MPP=2, ...``) with
optional ties. To make that ordering comparable to PAGA / scVelo per-cell
pseudotimes — and to overlay it on a UMAP — we lift the discrete ranks to
a continuous per-cell pseudotime in [0, 1] by:

1. **Base term** — collapse the LLM's integer rank to a unit-interval
   coordinate::

       base = (rank - min_rank) / (max_rank - min_rank)

   All cells in a cluster share the same ``base``. Tied clusters share
   ``base`` exactly; their cells are distinguished only by the spread
   term below.

2. **Spread term** — give each cell within a cluster a small position so
   the per-cell vector is not piecewise-constant. The spread is bounded to
   roughly half the gap between adjacent ranks so cells never cross the
   cluster boundary set by ``base``. For ``K`` ranks the inter-rank gap
   is ``1 / (K - 1)``; we cap ``|spread| <= 0.05``, conservative enough
   for any ``K >= 11`` and still tight for smaller ``K``.

   Within a cluster, we project cells onto a 1-D axis:

   * ``spread_method="diffmap"`` and ``X_diffmap`` present → diffmap
     component 1.
   * ``spread_method="pca"`` or diffmap absent → PC1 from ``X_pca``.
   * Neither available → spread is 0 (output is piecewise-constant).

   For each cluster we then compute, with ``c`` = centroid of the cluster
   along that axis, ``sigma`` = mean absolute deviation from ``c``::

       spread_i = sign(x_i - c) * |x_i - c| / max(sigma, eps) * SPREAD_CAP / S

   where ``SPREAD_CAP = 0.05`` and ``S`` is a per-cluster normalizer that
   forces ``max_i |spread_i| <= SPREAD_CAP``. This is sign-preserved so
   cells that are "earlier" in the chosen axis get smaller pseudotime
   than the cluster centroid, "later" cells get larger.

3. **Combination** — ``final = clamp(base + spread, 0, 1)``. The clamp
   matters only at the two extreme clusters where the base is exactly 0
   or 1 — interior clusters always stay inside ``(0, 1)`` by construction.

The function is pure: it does not mutate the input :class:`AnnData`. The
return is a :class:`pandas.Series` indexed by ``adata.obs.index`` with
``name="litchron_pseudotime"``.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from anndata import AnnData

# Cap on per-cell jitter within a cluster (absolute value).
_SPREAD_CAP: float = 0.05


def _pick_axis(adata: AnnData, spread_method: str) -> np.ndarray | None:
    """Return a 1-D numpy array (length n_obs) for the spread axis, or None.

    * ``spread_method == "diffmap"`` prefers ``obsm["X_diffmap"]`` column 1
      (component 0 is usually the constant eigenvector for connected
      graphs, so component 1 is the first informative coordinate).
    * Falls back to ``obsm["X_pca"]`` column 0 when diffmap is absent or
      ``spread_method == "pca"``.
    * Returns ``None`` when neither embedding exists; the caller drops
      the spread term.
    """
    obsm = getattr(adata, "obsm", None) or {}
    if spread_method == "diffmap":
        x = obsm.get("X_diffmap")
        if x is not None and getattr(x, "ndim", 0) == 2 and x.shape[1] >= 2:
            return np.asarray(x[:, 1], dtype=np.float64)
        # fall through to PCA fallback
    x = obsm.get("X_pca")
    if x is not None and getattr(x, "ndim", 0) == 2 and x.shape[1] >= 1:
        return np.asarray(x[:, 0], dtype=np.float64)
    return None


def _normalize_within_cluster(coords: np.ndarray) -> np.ndarray:
    """Sign-preserved, magnitude-capped within-cluster spread in [-CAP, CAP].

    Steps for ``coords`` (length ``n``):

    1. Subtract the cluster centroid: ``d = coords - mean(coords)``.
    2. Scale so ``max |d| == CAP`` (no scaling when all coords are equal).
    3. Return zero if the cluster has fewer than 2 cells.
    """
    n = coords.shape[0]
    if n < 2:
        return np.zeros(n, dtype=np.float64)
    centered = coords - float(coords.mean())
    peak = float(np.max(np.abs(centered)))
    if peak <= 0.0 or not np.isfinite(peak):
        return np.zeros(n, dtype=np.float64)
    return (centered / peak) * _SPREAD_CAP


def compute_llm_pseudotime(
    adata: AnnData,
    per_cell_type_rank: list[Mapping[str, Any]],
    cell_type_col: str = "leiden",
    spread_method: str = "diffmap",
) -> pd.Series:
    """Lift the LLM's per-cluster ranks to a continuous per-cell pseudotime.

    Parameters
    ----------
    adata
        AnnData providing the per-cell cluster assignment in
        ``adata.obs[cell_type_col]`` and an optional embedding in
        ``adata.obsm`` for the spread term.
    per_cell_type_rank
        List of dicts of the same shape as the persisted LLM proposal::

            [{"cell_type": "0", "rank": 1, "confidence": 0.9,
              "tied_with": None, "cell_type_label": "LT-HSC"}, ...]

        Only ``cell_type`` and ``rank`` are required; other fields are
        ignored here (they affect bookkeeping, not the math).
    cell_type_col
        Column in ``adata.obs`` that holds cluster IDs. Falls back to
        ``"cell_type"`` when ``leiden`` is absent — this matches the
        synthetic fixtures shipped with the test suite.
    spread_method
        ``"diffmap"`` (default) or ``"pca"``. See module docstring for
        fallback semantics.

    Returns
    -------
    pandas.Series
        Length ``adata.n_obs``, index matches ``adata.obs.index``,
        name ``"litchron_pseudotime"``. Values are floats in [0, 1].

    Notes
    -----
    * **Pure function** — does not write to ``adata``.
    * **Ties** — clusters with the same rank share ``base``; their cells
      differ only by the spread term, so the per-cell ordering inside a
      tie set is well-defined and stable.
    * **Missing clusters** — clusters present in ``adata.obs`` but absent
      from ``per_cell_type_rank`` are assigned ``rank = max_rank + 1``,
      i.e. parked at the end. This keeps the output length equal to
      ``adata.n_obs`` even when the LLM forgets a cluster.
    """
    if not per_cell_type_rank:
        raise ValueError("per_cell_type_rank must contain at least one entry")

    # Resolve the cluster column with a single fallback so the routine
    # works on both Wave R2 leiden output and the synthetic 'cell_type'
    # fixture without callers having to know which is present.
    col = cell_type_col
    if col not in adata.obs.columns:
        if "cell_type" in adata.obs.columns:
            col = "cell_type"
        else:
            raise KeyError(
                f"cluster column {cell_type_col!r} (and fallback 'cell_type') "
                "not found in adata.obs"
            )

    # Build rank lookup: {cell_type_id: rank}. ``cell_type`` is normalized
    # to str to match the categorical strings in adata.obs.
    rank_map: dict[str, int] = {}
    for entry in per_cell_type_rank:
        if "cell_type" not in entry or "rank" not in entry:
            raise ValueError(
                "every per_cell_type_rank entry must contain 'cell_type' and 'rank'"
            )
        rank_map[str(entry["cell_type"])] = int(entry["rank"])

    ranks_seen = list(rank_map.values())
    min_rank = min(ranks_seen)
    max_rank = max(ranks_seen)
    rank_span = max_rank - min_rank

    cluster_series = adata.obs[col].astype(str)
    n_obs = adata.n_obs

    # Base pseudotime per cluster, with a fall-back for un-ranked clusters.
    unknown_rank = max_rank + 1
    base_per_cell = np.empty(n_obs, dtype=np.float64)
    cluster_array = cluster_series.to_numpy()
    if rank_span == 0:
        # Degenerate case: every cluster gets the same rank. Pin the base
        # at 0.5 so the spread term still places cells in [0.45, 0.55].
        base_per_cell.fill(0.5)
    else:
        for i, c in enumerate(cluster_array):
            r = rank_map.get(c, unknown_rank)
            base_per_cell[i] = (r - min_rank) / rank_span

    # Spread term — sign-preserved within-cluster normalization on the
    # chosen 1-D embedding axis. When neither embedding exists, the
    # spread is zero everywhere and the output is piecewise-constant
    # (still in [0, 1], still pure).
    spread_per_cell = np.zeros(n_obs, dtype=np.float64)
    axis = _pick_axis(adata, spread_method)
    if axis is not None and axis.shape[0] == n_obs:
        for c in pd.unique(cluster_array):
            mask = cluster_array == c
            if not mask.any():
                continue
            spread_per_cell[mask] = _normalize_within_cluster(axis[mask])

    final = np.clip(base_per_cell + spread_per_cell, 0.0, 1.0)

    return pd.Series(
        final,
        index=adata.obs.index,
        name="litchron_pseudotime",
        dtype=np.float64,
    )


__all__ = ["compute_llm_pseudotime"]
