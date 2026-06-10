"""Direction-invariance of the pseudotime comparison (issue #18).

A correct-but-reversed baseline must not read as total disagreement: signed
Spearman is ~ -1, but abs_spearman recovers the true agreement, and the p-value
(n-aware) decides significance instead of a magic |rho| cutoff.
"""
from __future__ import annotations

import pandas as pd
import pytest

from litchron.compare import compare


def _ranks(values: dict[str, float]) -> pd.Series:
    return pd.Series(values, dtype=float)


def test_reversed_ordering_is_perfect_agreement_under_abs():
    a = _ranks({f"ct{i}": float(i) for i in range(12)})
    reversed_b = _ranks({f"ct{i}": float(11 - i) for i in range(12)})
    row = compare(a, reversed_b, baseline_name="reversed")
    # Signed correlation looks like maximal disagreement...
    assert row.spearman == pytest.approx(-1.0)
    # ...but the direction-invariant magnitude shows perfect agreement,
    assert row.abs_spearman == pytest.approx(1.0)
    # and it is highly significant (n=12), so NOT "no better than chance".
    assert row.spearman_pvalue is not None and row.spearman_pvalue < 0.05
    assert row.agreement_no_better_than_chance is False


def test_identical_ordering_is_perfect_agreement():
    a = _ranks({f"ct{i}": float(i) for i in range(12)})
    row = compare(a, a.copy(), baseline_name="identical")
    assert row.spearman == pytest.approx(1.0)
    assert row.abs_spearman == pytest.approx(1.0)
    assert row.agreement_no_better_than_chance is False


def test_chance_flag_tracks_pvalue():
    a = _ranks({f"ct{i}": float(i) for i in range(6)})
    weak = _ranks({"ct0": 2.0, "ct1": 5.0, "ct2": 0.0, "ct3": 4.0, "ct4": 1.0, "ct5": 3.0})
    row = compare(a, weak, baseline_name="weak")
    assert row.spearman_pvalue is not None
    # the chance flag is exactly "not significant at alpha=0.05"
    assert row.agreement_no_better_than_chance == (row.spearman_pvalue > 0.05)
