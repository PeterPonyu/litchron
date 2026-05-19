"""Smoke tests for :mod:`litchron.llm_pseudotime`."""
from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from litchron.llm_pseudotime import compute_llm_pseudotime


def _build_three_cluster_adata(n_cells: int = 100, seed: int = 0) -> AnnData:
    """100 cells × 5 genes, three clusters labeled "0"/"1"/"2".

    Provides ``X_diffmap`` and ``X_pca`` so the spread term can use either
    embedding without hitting the fallback path.
    """
    rng = np.random.default_rng(seed)

    cluster_ids: list[str] = []
    diffmap_axis: list[float] = []
    pca_axis: list[float] = []
    per_cluster = n_cells // 3
    remainder = n_cells - 3 * per_cluster
    for c in range(3):
        n_c = per_cluster + (1 if c < remainder else 0)
        cluster_ids.extend([str(c)] * n_c)
        # Cluster-local axis values centered on c with bounded jitter so
        # within-cluster sign is deterministic and the spread term is
        # non-zero.
        diffmap_axis.extend(rng.normal(loc=float(c), scale=0.1, size=n_c).tolist())
        pca_axis.extend(rng.normal(loc=float(c), scale=0.1, size=n_c).tolist())

    X = rng.poisson(lam=1.0, size=(n_cells, 5)).astype(np.float32)
    obs = pd.DataFrame(
        {"cell_type": pd.Categorical(cluster_ids, categories=["0", "1", "2"])},
        index=[f"cell_{i:04d}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(5)])
    adata = AnnData(X=X, obs=obs, var=var)

    # diffmap needs at least 2 columns (we use column 1).
    diffmap = np.column_stack(
        [np.ones(n_cells, dtype=np.float64), np.asarray(diffmap_axis)]
    )
    adata.obsm["X_diffmap"] = diffmap
    adata.obsm["X_pca"] = np.column_stack(
        [np.asarray(pca_axis), np.zeros(n_cells, dtype=np.float64)]
    )
    return adata


def _fake_proposal() -> list[dict[str, object]]:
    return [
        {"cell_type": "0", "rank": 1, "confidence": 0.9, "tied_with": None,
         "cell_type_label": "Early"},
        {"cell_type": "1", "rank": 2, "confidence": 0.8, "tied_with": None,
         "cell_type_label": "Mid"},
        {"cell_type": "2", "rank": 3, "confidence": 0.95, "tied_with": None,
         "cell_type_label": "Late"},
    ]


def test_compute_llm_pseudotime_returns_series_in_unit_interval() -> None:
    adata = _build_three_cluster_adata(n_cells=100)
    out = compute_llm_pseudotime(
        adata=adata,
        per_cell_type_rank=_fake_proposal(),
        cell_type_col="leiden",  # exercise the cell_type fallback
        spread_method="diffmap",
    )

    assert isinstance(out, pd.Series)
    assert len(out) == 100
    assert list(out.index) == list(adata.obs.index)
    assert out.name == "litchron_pseudotime"
    # Values in [0, 1] after clamping.
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_compute_llm_pseudotime_per_cluster_means_track_rank() -> None:
    adata = _build_three_cluster_adata(n_cells=100)
    out = compute_llm_pseudotime(
        adata=adata,
        per_cell_type_rank=_fake_proposal(),
        cell_type_col="leiden",
        spread_method="diffmap",
    )

    means = out.groupby(adata.obs["cell_type"].astype(str)).mean()
    # Rank 1 → base 0, Rank 2 → 0.5, Rank 3 → 1.0; spread is symmetric so
    # the cluster mean lands near the base value.
    assert means["0"] < means["1"] < means["2"]
    assert abs(means["0"] - 0.0) < 0.06
    assert abs(means["1"] - 0.5) < 0.06
    assert abs(means["2"] - 1.0) < 0.06


def test_compute_llm_pseudotime_pca_fallback_when_diffmap_absent() -> None:
    adata = _build_three_cluster_adata(n_cells=100)
    # Drop diffmap; the function should silently fall back to PC1.
    del adata.obsm["X_diffmap"]
    out = compute_llm_pseudotime(
        adata=adata,
        per_cell_type_rank=_fake_proposal(),
        cell_type_col="leiden",
        spread_method="diffmap",
    )
    assert len(out) == 100
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_compute_llm_pseudotime_no_embedding_returns_base_only() -> None:
    adata = _build_three_cluster_adata(n_cells=100)
    del adata.obsm["X_diffmap"]
    del adata.obsm["X_pca"]
    out = compute_llm_pseudotime(
        adata=adata,
        per_cell_type_rank=_fake_proposal(),
        cell_type_col="leiden",
        spread_method="diffmap",
    )
    # With no spread axis, every cell in a cluster shares the cluster's
    # base pseudotime exactly.
    by_cluster = out.groupby(adata.obs["cell_type"].astype(str)).nunique()
    assert (by_cluster == 1).all()


def test_compute_llm_pseudotime_ties_share_base() -> None:
    """Two clusters at the same rank should share their base value."""
    adata = _build_three_cluster_adata(n_cells=100)
    proposal = [
        {"cell_type": "0", "rank": 1},
        {"cell_type": "1", "rank": 2, "tied_with": ["2"]},
        {"cell_type": "2", "rank": 2, "tied_with": ["1"]},
    ]
    # Disable the spread term so we observe base alone.
    del adata.obsm["X_diffmap"]
    del adata.obsm["X_pca"]
    out = compute_llm_pseudotime(
        adata=adata,
        per_cell_type_rank=proposal,
        cell_type_col="leiden",
    )
    means = out.groupby(adata.obs["cell_type"].astype(str)).mean()
    # Ranks {1, 2, 2} → bases {0.0, 1.0, 1.0}.
    assert means["0"] == 0.0
    assert means["1"] == means["2"] == 1.0
