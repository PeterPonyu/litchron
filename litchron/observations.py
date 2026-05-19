"""Compute and serialize LitChron observations from an AnnData.

An :class:`Observations` document is the LLM's first read of the dataset:
cluster identities, top marker genes, modality, and size summary. The LLM
uses it to ground its pseudotime proposal in cell-type biology.

The :func:`observations_to_markdown` helper renders the model into a
YAML-front-matter + markdown-table document suitable for writing to
``runs/<run_id>/observations.md``. The on-disk write itself is performed by
the caller (typically the MCP server) so this module stays side-effect-free.
"""
from __future__ import annotations

from typing import Any

import scanpy as sc
from anndata import AnnData
from pydantic import BaseModel, Field

from litchron.embeddings import recompute_embeddings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Cluster column candidates, scanned in order. Extended beyond the original
# (leiden/louvain/cell_type) to cover common upstream tutorial AnnData layouts:
# scvelo's pancreas dataset uses "clusters" / "clusters_coarse" / "clusters_fine";
# cellxgene downloads often expose "ClusterName" / "Clusters". Order is
# significant: scanpy-native names win when present so the LLM oracle's
# proposal IDs (which assume leiden/louvain numbering) keep matching.
_CLUSTER_CANDIDATES: tuple[str, ...] = (
    "leiden",
    "louvain",
    "cell_type",
    "celltype",
    "annotation",
    "clusters",
    "Clusters",
    "cluster",
    "ClusterName",
    "cluster_label",
)
_TOP_N_MARKERS: int = 8


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class Observations(BaseModel):
    """Structured summary of an AnnData for LLM consumption."""

    clusters: list[str]
    markers_per_cluster: dict[str, list[str]]
    modality: str
    n_cells: int
    n_genes: int
    layer_summary: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
def pick_cluster_column(adata: AnnData) -> str | None:
    """Return the first known cluster column found in ``adata.obs``, or None.

    Public because the figure builders and audit driver need the same
    resolution logic to support runs bootstrapped from upstream AnnDatas
    that don't use scanpy's default ``leiden`` column name.
    """
    for candidate in _CLUSTER_CANDIDATES:
        if candidate in adata.obs.columns:
            return candidate
    return None


# Internal alias preserved for backward compatibility with existing call sites.
_pick_cluster_column = pick_cluster_column


# 2D embedding-key candidates, scanned in order. Scanpy writes "X_umap" by
# default; some pipelines/tutorials use the upper-case or t-SNE variant. We
# stop at the first match; figures.py expects exactly 2D coordinates.
_EMBEDDING_CANDIDATES: tuple[str, ...] = ("X_umap", "X_UMAP", "umap", "X_tsne", "X_TSNE")


def pick_embedding_key(adata: AnnData) -> str | None:
    """Return the first known 2D-embedding key found in ``adata.obsm``, or None.

    Used by figure builders that need scatter coordinates. Callers should
    handle ``None`` with a clear "run recompute_embeddings first" error
    rather than letting a downstream ``KeyError`` propagate.
    """
    obsm_keys = getattr(adata, "obsm", {}) or {}
    for candidate in _EMBEDDING_CANDIDATES:
        if candidate in obsm_keys:
            return candidate
    return None


def _modality_hint(adata: AnnData) -> str:
    """Best-effort modality string derived from layers / var.

    Pure inspection — does not mutate the adata.
    """
    layers = set(getattr(adata, "layers", {}).keys() or [])
    if {"spliced", "unspliced"} <= layers:
        return "scrna_velocity"
    var_names = getattr(adata, "var", None)
    if var_names is not None and "feature_types" in adata.var.columns:
        types = set(map(str, adata.var["feature_types"].unique()))
        if {"Gene Expression", "Peaks"} <= types:
            return "multiome"
        if {"Gene Expression", "Antibody Capture"} <= types:
            return "citeseq"
    return "scrna"


def _layer_summary(adata: AnnData) -> dict[str, Any]:
    """Build a small JSON-safe summary of available adata layers."""
    layers = getattr(adata, "layers", {}) or {}
    summary: dict[str, Any] = {"X_dtype": str(adata.X.dtype) if adata.X is not None else None}
    summary["layers"] = sorted(map(str, layers.keys()))
    summary["obs_columns"] = sorted(map(str, adata.obs.columns))
    return summary


def compute_observations(adata: AnnData) -> Observations:
    """Compute :class:`Observations` from an AnnData.

    Side effects
    ------------
    Mutates ``adata`` in place:

    * If adata lacks ``X_pca``, ``X_umap``, or ``leiden``, recomputes via
      :func:`litchron.embeddings.recompute_embeddings` (with ``force=False``
      so a pre-embedded adata is returned unchanged — preserves cache hits).
    * If no cluster column is present after the recompute, runs
      ``sc.tl.leiden(adata, resolution=0.8)`` (writes
      ``adata.obs['leiden']``).
    * If ``adata.uns`` lacks ``rank_genes_groups`` for the chosen cluster
      column, runs ``sc.tl.rank_genes_groups(adata, ..., method='wilcoxon',
      n_genes=15)``.

    These are conventional scanpy mutations expected by every downstream
    analysis; the AnnDataCache replay protocol records them as deltas if
    the caller chooses to persist them.
    """
    recompute_embeddings(adata, force=False)

    cluster_col = _pick_cluster_column(adata)
    if cluster_col is None:
        sc.tl.leiden(adata, resolution=0.8)
        cluster_col = "leiden"

    rgg = adata.uns.get("rank_genes_groups")
    needs_recompute = (
        rgg is None
        or not isinstance(rgg, dict)
        or rgg.get("params", {}).get("groupby") != cluster_col
    )
    if needs_recompute:
        sc.tl.rank_genes_groups(
            adata,
            groupby=cluster_col,
            method="wilcoxon",
            n_genes=_TOP_N_MARKERS,
        )

    names = adata.uns["rank_genes_groups"]["names"]
    # ``names`` is a structured numpy recarray: columns are cluster ids,
    # rows are ranked gene names. Iterate columns to collect top-N markers.
    cluster_ids: list[str] = [str(c) for c in names.dtype.names]
    markers_per_cluster: dict[str, list[str]] = {}
    for cid in cluster_ids:
        col = names[cid]
        top = [str(g) for g in col[:_TOP_N_MARKERS]]
        markers_per_cluster[cid] = top

    return Observations(
        clusters=cluster_ids,
        markers_per_cluster=markers_per_cluster,
        modality=_modality_hint(adata),
        n_cells=int(adata.n_obs),
        n_genes=int(adata.n_vars),
        layer_summary=_layer_summary(adata),
    )


# ---------------------------------------------------------------------------
# Markdown serializer
# ---------------------------------------------------------------------------
def observations_to_markdown(obs: Observations) -> str:
    """Render an :class:`Observations` as YAML+markdown.

    The output begins with a YAML front-matter block (between ``---``
    fences) carrying machine-readable metadata, followed by a markdown
    table with one row per cluster listing the top marker genes.
    """
    clusters_yaml = "\n".join(f"  - {c}" for c in obs.clusters) if obs.clusters else "  []"
    front_matter = (
        "---\n"
        f"n_cells: {obs.n_cells}\n"
        f"n_genes: {obs.n_genes}\n"
        f"modality: {obs.modality}\n"
        "clusters:\n"
        f"{clusters_yaml}\n"
        "---\n"
    )

    lines: list[str] = [front_matter, "# Observations", ""]
    lines.append("| cluster_id | top markers |")
    lines.append("|---|---|")
    for cid in obs.clusters:
        markers = obs.markers_per_cluster.get(cid, [])
        markers_str = ", ".join(markers) if markers else "_none_"
        lines.append(f"| {cid} | {markers_str} |")
    lines.append("")
    return "\n".join(lines)
