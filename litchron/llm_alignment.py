"""Robust time-order alignment between competing per-cell pseudotimes.

This module exists because the user wants order-alignment metrics that do
**not** depend on the specific distance metric PAGA/scVelo use. The
canonical pairwise concordance counts are agnostic to the absolute
pseudotime values — they only ask "do the two orderings agree on the
direction of each cell pair?" — which is the cleanest cross-method
comparison.

Metrics computed pairwise per ``(name_a, name_b)``:

* **spearman** — Spearman rank correlation. Linear monotone agreement.
* **kendall** — Kendall tau-b. Pairwise concordance with tie correction.
* **gamma** — Goodman–Kruskal gamma::

      (C - D) / (C + D)

  where ``C`` is the number of concordant cell pairs and ``D`` the number
  of discordant pairs. Ties on either side are excluded from the
  denominator, so gamma is defined exactly when there is at least one
  non-tied pair on each side. Returns ``None`` when ``C + D == 0``.
* **concordance** — fraction of cell pairs ``(a, b)`` with ``a != b``
  where ``a < b`` under both orderings::

      |{(i, j) : x_i < x_j and y_i < y_j}|
      ------------------------------------
            |{(i, j) : x_i != x_j and y_i != y_j}|

  This is the "monotonic concordance" the user asked for; it is invariant
  under any strictly monotone transform of either side.

All four metrics are computed on the intersection index after dropping
any row with a NaN in either series.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr  # type: ignore[import-untyped]


def _concordant_discordant_counts(
    x: np.ndarray, y: np.ndarray
) -> tuple[int, int, int, int]:
    """Return (concordant, discordant, ties_x, ties_y) over all pairs.

    Uses the closed-form identity that ``kendalltau`` (variant ``"b"``)
    computes internally; counting concordant/discordant pairs directly
    is :math:`O(n^2)` but adequate for the typical 10^4..10^5 cell
    populations LitChron sees in practice. Implemented vectorized via
    pairwise sign products to keep the inner loop in numpy.
    """
    n = x.shape[0]
    if n < 2:
        return 0, 0, 0, 0

    # Build pairwise sign matrices; we only need the upper triangle.
    iu = np.triu_indices(n, k=1)
    dx = np.sign(x[iu[0]] - x[iu[1]])
    dy = np.sign(y[iu[0]] - y[iu[1]])

    concordant = int(np.sum((dx * dy) > 0))
    discordant = int(np.sum((dx * dy) < 0))
    ties_x = int(np.sum((dx == 0) & (dy != 0)))
    ties_y = int(np.sum((dy == 0) & (dx != 0)))
    return concordant, discordant, ties_x, ties_y


def _goodman_kruskal_gamma(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Return (C - D) / (C + D); ``None`` when no non-tied pairs exist."""
    c, d, _tx, _ty = _concordant_discordant_counts(x, y)
    denom = c + d
    if denom == 0:
        return None
    return float(c - d) / float(denom)


def _monotonic_concordance(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Fraction of non-tied pairs that agree in direction.

    Distinct from ``gamma`` only in normalization: gamma is on
    ``[-1, 1]`` (sign-tracking), concordance is on ``[0, 1]`` (fraction
    of agreements). They share the same numerator-of-non-tied-pairs
    denominator.
    """
    c, d, _tx, _ty = _concordant_discordant_counts(x, y)
    denom = c + d
    if denom == 0:
        return None
    return float(c) / float(denom)


def _clean_pair(a: pd.Series, b: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Align two series on their shared index and drop rows with any NaN."""
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    aa = a.loc[common].astype(np.float64)
    bb = b.loc[common].astype(np.float64)
    mask = ~(aa.isna() | bb.isna())
    return aa[mask].to_numpy(), bb[mask].to_numpy()


def align_orderings(
    orderings: dict[str, pd.Series],
) -> dict[tuple[str, str], dict[str, Optional[float]]]:
    """Compute pairwise alignment metrics across N named pseudotime series.

    Parameters
    ----------
    orderings
        Mapping from ordering name (e.g. ``"litchron"``, ``"paga"``,
        ``"scvelo"``) to a :class:`pandas.Series` of per-cell pseudotime
        indexed by cell ID.

    Returns
    -------
    dict
        Keyed by ``(name_a, name_b)`` with ``name_a < name_b`` lexically.
        Each value is::

            {"spearman": float|None,
             "kendall":  float|None,
             "gamma":    float|None,
             "concordance": float|None,
             "n":         int}

        ``n`` is the size of the cleaned intersection (after NaN drop).

    Notes
    -----
    * Pairs are emitted with a stable lexical order on the name keys to
      keep the output deterministic across Python sessions.
    * When the intersection has fewer than 2 cells, every metric is
      ``None`` and ``n`` reports the actual cleaned size.
    * Goodman-Kruskal gamma abstains (returns ``None``) when *all* pairs
      are tied on at least one side, e.g. constant input series.
    """
    names = sorted(orderings.keys())
    out: dict[tuple[str, str], dict[str, Optional[float]]] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n_a, n_b = names[i], names[j]
            x, y = _clean_pair(orderings[n_a], orderings[n_b])
            if x.shape[0] < 2:
                out[(n_a, n_b)] = {
                    "spearman": None,
                    "kendall": None,
                    "gamma": None,
                    "concordance": None,
                    "n": int(x.shape[0]),
                }
                continue
            sp = spearmanr(x, y).statistic
            kt = kendalltau(x, y).statistic
            gamma = _goodman_kruskal_gamma(x, y)
            conc = _monotonic_concordance(x, y)
            out[(n_a, n_b)] = {
                "spearman": float(sp) if sp == sp else None,
                "kendall": float(kt) if kt == kt else None,
                "gamma": gamma,
                "concordance": conc,
                "n": int(x.shape[0]),
            }
    return out


def alignment_to_markdown(
    result: dict[tuple[str, str], dict[str, Optional[float]]],
) -> str:
    """Render the alignment dict as a markdown table.

    Columns: ``method_a | method_b | n | spearman | kendall | gamma |
    concordance``. ``None`` values render as ``"—"``; floats render with
    three decimals.
    """

    def fmt(v: Optional[float]) -> str:
        if v is None:
            return "—"
        return f"{v:.3f}"

    lines: list[str] = [
        "# Pseudotime Alignment",
        "",
        "| method_a | method_b | n | spearman | kendall | gamma | concordance |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for (a, b), metrics in sorted(result.items()):
        lines.append(
            "| "
            + " | ".join(
                [
                    a,
                    b,
                    str(metrics.get("n", 0)),
                    fmt(metrics.get("spearman")),
                    fmt(metrics.get("kendall")),
                    fmt(metrics.get("gamma")),
                    fmt(metrics.get("concordance")),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "align_orderings",
    "alignment_to_markdown",
]
