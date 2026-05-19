"""Shared pytest fixtures for the LitChron test suite."""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.fixtures.synthetic import make_synthetic_adata


@pytest.fixture
def synthetic_adata():
    """200-cell × 50-gene AnnData with three clusters and a known linear pseudotime.

    Seed is fixed at 42 for determinism across all tests that use this fixture.
    """
    return make_synthetic_adata(n_cells=200, n_genes=50, n_clusters=3, seed=42)


@pytest.fixture
def tmp_run_dir(tmp_path: Path) -> Path:
    """Return a fresh run directory under tmp_path and create it on disk."""
    d = tmp_path / "litchron_runs" / "test-run-abc123"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def project_root() -> Path:
    """Return the absolute path to the litchron repository root."""
    # tests/ is one level below the repo root.
    return Path(__file__).resolve().parent.parent
