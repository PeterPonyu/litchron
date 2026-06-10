"""Recompute UMAP + PCA + leiden from raw counts on an incoming AnnData.

LitChron treats every input ``.h5ad`` as "as if new" — pre-existing
``X_pca`` / ``X_umap`` / ``obs['leiden']`` from upstream tooling may
encode a different normalization, gene selection, or clustering
resolution than LitChron expects. :func:`recompute_embeddings` therefore
runs the conventional scanpy pipeline (normalize_total → log1p →
highly_variable_genes → PCA → neighbors → UMAP → leiden) and stamps
the results back onto the AnnData.

The function mutates ``adata`` in place. The :class:`AnnDataCache`
replay protocol records these mutations as a delta when the MCP server
chooses to persist them (see :func:`mcp_litchron.tools.recompute_embeddings`).

This module is intentionally permitted to raise — callers wrap it in
the structured :class:`LitchronError` / :class:`ErrorResult` boundary.
"""
from __future__ import annotations

import scanpy as sc
from anndata import AnnData

from .config import RANDOM_SEED


def _has_existing_embedding(adata: AnnData) -> bool:
    """True iff adata already has PCA, UMAP, and a leiden column."""
    obsm = getattr(adata, "obsm", None) or {}
    obs_cols = set(map(str, adata.obs.columns)) if hasattr(adata, "obs") else set()
    return (
        "X_pca" in obsm
        and "X_umap" in obsm
        and "leiden" in obs_cols
    )


def recompute_embeddings(
    adata: AnnData,
    force: bool = False,
    leiden_resolution: float = 1.0,
    n_neighbors: int = 15,
    n_pcs: int = 30,
    seed: int = RANDOM_SEED,
) -> AnnData:
    """Recompute PCA, UMAP, and leiden clustering on ``adata``.

    Pipeline
    --------
    1. ``sc.pp.normalize_total(target_sum=1e4)``
    2. ``sc.pp.log1p``
    3. ``sc.pp.highly_variable_genes(n_top_genes=2000)`` (subset to HVGs)
    4. ``sc.pp.pca(n_comps=n_pcs)``
    5. ``sc.pp.neighbors(n_neighbors=n_neighbors)``
    6. ``sc.tl.umap``
    7. ``sc.tl.leiden(resolution=leiden_resolution)``

    Parameters
    ----------
    adata
        The AnnData to recompute on; mutated in place.
    force
        If ``False`` and ``adata`` already has ``X_pca``, ``X_umap``, and
        ``obs['leiden']``, the function is a no-op and returns ``adata``
        unchanged. Pass ``force=True`` to always recompute.
    leiden_resolution
        Leiden clustering resolution; higher → more clusters.
    n_neighbors
        Number of nearest neighbors for the kNN graph.
    n_pcs
        Number of principal components retained.
    seed
        Random seed threaded into PCA, neighbors, UMAP, and Leiden so the
        embedding and cluster labels are reproducible across runs. Defaults to
        :data:`litchron.config.RANDOM_SEED` (overridable via ``LITCHRON_SEED``).

    Returns
    -------
    The mutated ``adata`` (in-place edits make the return value
    convenient but not strictly necessary).
    """
    if not force and _has_existing_embedding(adata):
        return adata

    # Defensive copy of the raw counts so log1p doesn't double-log on a
    # second invocation. We only do this when actually recomputing.
    if adata.raw is None:
        adata.raw = adata.copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    # Subset to HVGs in-place (memory + downstream PCA both benefit).
    if "highly_variable" in adata.var.columns:
        adata._inplace_subset_var(adata.var["highly_variable"].values)

    # Cap n_comps to avoid LAPACK errors on tiny datasets (e.g. unit
    # tests with 200 cells). scanpy default is 50; we honor the caller's
    # requested n_pcs but never exceed min(n_obs, n_vars) - 1.
    safe_n_pcs = min(int(n_pcs), max(1, min(int(adata.n_obs), int(adata.n_vars)) - 1))
    sc.pp.pca(adata, n_comps=safe_n_pcs, random_state=seed)

    safe_n_neighbors = min(int(n_neighbors), max(2, int(adata.n_obs) - 1))
    sc.pp.neighbors(adata, n_neighbors=safe_n_neighbors, random_state=seed)

    sc.tl.umap(adata, random_state=seed)
    sc.tl.leiden(adata, resolution=float(leiden_resolution), random_state=seed)

    return adata


__all__ = ["recompute_embeddings"]
