"""Pure helper: generate a small synthetic AnnData for deterministic testing.

Three clusters with simulated marker expression patterns and a
``true_pseudotime`` column in ``.obs``. Used by ``tests/conftest.py`` and
integration tests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData


def make_synthetic_adata(
    n_cells: int = 200,
    n_genes: int = 50,
    n_clusters: int = 3,
    seed: int = 42,
) -> AnnData:
    """Return a synthetic :class:`AnnData` with known linear pseudotime ordering.

    Layout
    ------
    * Cells are split evenly across ``n_clusters`` clusters.
    * Each cluster has a distinct set of "marker" genes with elevated mean
      expression; the rest of the matrix is low-level Poisson noise.
    * ``adata.obs["cluster"]`` holds integer cluster labels as strings.
    * ``adata.obs["true_pseudotime"]`` is a float in [0, 1] that increases
      linearly from cluster 0 → cluster 1 → cluster 2 (the ground truth the
      tests compare recovered orderings against).
    * ``adata.obsm["X_pca"]`` holds a trivial 2-D embedding (PCA not run —
      just the first two principal-component-like dimensions from the data
      matrix) so PAGA and neighbors have something to work with.
    """
    rng = np.random.default_rng(seed)

    cells_per_cluster = n_cells // n_clusters
    remainder = n_cells % n_clusters

    cell_ids: list[str] = []
    cluster_labels: list[str] = []
    pseudotime_vals: list[float] = []

    X_parts: list[np.ndarray] = []

    for c in range(n_clusters):
        n_c = cells_per_cluster + (1 if c < remainder else 0)

        # Marker genes: a disjoint band per cluster.
        marker_start = (c * n_genes) // n_clusters
        marker_end = ((c + 1) * n_genes) // n_clusters

        # Base expression: moderate Poisson noise everywhere (higher lam so
        # clusters are not perfectly isolated in PCA space — PAGA needs
        # inter-cluster connections in the neighbor graph to function).
        block = rng.poisson(lam=1.5, size=(n_c, n_genes)).astype(np.float32)

        # Moderate elevation in the marker band (not so strong that clusters
        # become completely disconnected in the neighbor graph).
        marker_signal = rng.poisson(lam=3.0, size=(n_c, marker_end - marker_start)).astype(
            np.float32
        )
        block[:, marker_start:marker_end] += marker_signal

        X_parts.append(block)

        for i in range(n_c):
            cell_idx = sum(
                cells_per_cluster + (1 if ci < remainder else 0) for ci in range(c)
            ) + i
            cell_ids.append(f"cell_{cell_idx:04d}")
            cluster_labels.append(str(c))
            # Linear pseudotime: cluster 0 → [0, 1/3), cluster 1 → [1/3, 2/3), etc.
            pt = (c + (i / n_c)) / n_clusters
            pseudotime_vals.append(float(pt))

    X = np.vstack(X_parts)

    obs = pd.DataFrame(
        {
            # Use "cell_type" so litchron.baselines.paga._pick_cluster_column()
            # finds it without needing to run leiden (which requires neighbors).
            "cell_type": pd.Categorical(cluster_labels, categories=[str(c) for c in range(n_clusters)]),
            "true_pseudotime": pseudotime_vals,
        },
        index=cell_ids,
    )

    var = pd.DataFrame(index=[f"gene_{i:03d}" for i in range(n_genes)])

    adata = AnnData(X=X, obs=obs, var=var)

    # Trivial 2-D embedding: use the first two gene columns as a stand-in for
    # PCA so downstream tools that require obsm["X_pca"] have it.
    adata.obsm["X_pca"] = X[:, :2].copy()

    return adata
