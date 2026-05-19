"""Spec §5.12 / L93-97: compare() across all three matrix cells."""
from __future__ import annotations

import pandas as pd
import pytest

from litchron.compare import compare
from mcp_litchron.errors import ComparisonProtocolError

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _series(values: dict[str, float]) -> pd.Series:
    return pd.Series(values, dtype=float)


# ---------------------------------------------------------------------------
# Cell 1: flat × flat
# ---------------------------------------------------------------------------

def test_flat_flat_returns_spearman_and_kendall() -> None:
    llm = _series({"A": 0.1, "B": 0.4, "C": 0.7, "D": 0.9})
    baseline = _series({"A": 0.2, "B": 0.3, "C": 0.8, "D": 1.0})

    row = compare(
        llm_ordering=llm,
        baseline_ordering=baseline,
        baseline_name="test-flat",
    )

    assert row.llm_shape == "flat_ranking"
    assert row.baseline_shape == "flat_ranking"
    assert row.spearman is not None
    assert row.kendall_tau is not None
    assert row.jaccard_edges is None


# ---------------------------------------------------------------------------
# Cell 2: flat × tree
# ---------------------------------------------------------------------------

def test_flat_tree_returns_spearman_and_bin_jaccard() -> None:
    llm = _series({"A": 0.1, "B": 0.5, "C": 0.9})
    baseline_order = _series({"A": 0.0, "B": 0.0, "C": 0.0})  # unused; tree takes over
    tree_edges = [("A", "B"), ("B", "C")]

    row = compare(
        llm_ordering=llm,
        baseline_ordering=baseline_order,
        baseline_edges=tree_edges,
        baseline_name="test-tree",
    )

    assert row.llm_shape == "flat_ranking"
    assert row.baseline_shape == "tree"
    assert row.spearman is not None
    assert row.rank_bin_jaccard is not None


# ---------------------------------------------------------------------------
# Cell 3: tree × tree
# ---------------------------------------------------------------------------

def test_tree_tree_returns_jaccard_edges() -> None:
    llm_order = _series({"A": 0.0, "B": 0.0, "C": 0.0})
    baseline_order = _series({"A": 0.0, "B": 0.0, "C": 0.0})
    llm_edges = [("A", "B"), ("B", "C")]
    baseline_edges = [("A", "B"), ("B", "C")]

    row = compare(
        llm_ordering=llm_order,
        baseline_ordering=baseline_order,
        llm_edges=llm_edges,
        baseline_edges=baseline_edges,
        llm_root="A",
        baseline_root="A",
        baseline_name="test-tree-tree",
    )

    assert row.llm_shape == "tree"
    assert row.baseline_shape == "tree"
    assert row.jaccard_edges == pytest.approx(1.0)
    assert row.root_cell_agreement is True


def test_tree_tree_partial_overlap() -> None:
    llm_order = _series({"A": 0.0, "B": 0.0, "C": 0.0})
    baseline_order = _series({"A": 0.0, "B": 0.0, "C": 0.0})
    llm_edges = [("A", "B"), ("B", "C")]
    baseline_edges = [("A", "B")]

    row = compare(
        llm_ordering=llm_order,
        baseline_ordering=baseline_order,
        llm_edges=llm_edges,
        baseline_edges=baseline_edges,
        baseline_name="partial-overlap",
    )

    assert row.jaccard_edges is not None
    assert 0.0 < row.jaccard_edges < 1.0


# ---------------------------------------------------------------------------
# Undefined cell: both empty → ComparisonProtocolError
# ---------------------------------------------------------------------------

def test_undefined_cell_raises_comparison_protocol_error() -> None:
    """Force the undefined-cell branch by monkey-patching _shape.

    The compare() matrix covers flat×flat, flat×tree/tree×flat, and tree×tree.
    The only way to reach the final ``raise ComparisonProtocolError`` is to
    have both _shape calls return values outside those three defined cells —
    which the current Literal types prevent in normal usage, but the test
    verifies the guard exists by injecting two unexpected shape strings.

    We patch _shape at the module level (temporarily) to return sentinel
    values that don't match any of the three branches, then restore it.
    """
    import litchron.compare as _compare_mod

    original_shape = _compare_mod._shape

    def _always_unknown(_edges):
        return "unknown_shape"  # type: ignore[return-value]

    _compare_mod._shape = _always_unknown  # type: ignore[assignment]
    try:
        with pytest.raises(ComparisonProtocolError) as exc_info:
            compare(
                llm_ordering=pd.Series({"A": 0.1, "B": 0.5}),
                baseline_ordering=pd.Series({"A": 0.2, "B": 0.6}),
                baseline_name="undefined-cell",
            )
        assert exc_info.value.code == "undefined_cell"
    finally:
        _compare_mod._shape = original_shape
