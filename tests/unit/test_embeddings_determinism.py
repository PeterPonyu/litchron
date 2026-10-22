"""Determinism test for recompute_embeddings (issue #20).

PCA / UMAP / Leiden are stochastic; without a pinned seed the cluster labels the
LLM ranks (and the baselines consume) change run-to-run, so the "numerical
comparators" are not reproducible. These tests pin that a fixed seed yields
identical clustering and PCA across independent runs, and that the default seed
comes from the config constant.
"""
from __future__ import annotations

import numpy as np
import pytest
from anndata import AnnData

from litchron.config import RANDOM_SEED
from litchron.embeddings import recompute_embeddings


def _synthetic_counts() -> AnnData:
    """Two separable groups of Poisson counts -> a clusterable AnnData.

    Data generation is itself seeded so the *input* is identical across calls;
    the test then checks that the *pipeline* is deterministic.
    """
    rng = np.random.default_rng(0)
    n_per, n_genes = 100, 60
    base = rng.poisson(2.0, size=(2 * n_per, n_genes)).astype(np.float32)
    base[n_per:, :15] += 8.0  # elevate the first 15 genes in the second group
    return AnnData(base)


def test_recompute_is_deterministic_with_fixed_seed():
    a1 = recompute_embeddings(_synthetic_counts(), force=True, seed=123)
    a2 = recompute_embeddings(_synthetic_counts(), force=True, seed=123)

    # Cluster labels (the thing the LLM ranks) must be identical.
    assert list(a1.obs["leiden"]) == list(a2.obs["leiden"])
    # And the embedding it derives from.
    np.testing.assert_allclose(a1.obsm["X_pca"], a2.obsm["X_pca"], atol=1e-6)
    np.testing.assert_allclose(a1.obsm["X_umap"], a2.obsm["X_umap"], atol=1e-6)


def test_default_seed_is_config_random_seed():
    a_default = recompute_embeddings(_synthetic_counts(), force=True)
    a_explicit = recompute_embeddings(_synthetic_counts(), force=True, seed=RANDOM_SEED)
    assert list(a_default.obs["leiden"]) == list(a_explicit.obs["leiden"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
