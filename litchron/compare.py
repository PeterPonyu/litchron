"""Compare LLM-proposed cell-type orderings against baseline trajectories.

The comparison matrix is a 2x2 over (llm_shape, baseline_shape) where each
shape is ``flat_ranking`` or ``tree``:

* ``(flat, flat)`` — Spearman + Kendall tau over the intersection of indices.
* ``(flat, tree)`` — linearize the tree by edge-depth from the root, then
  Spearman + Kendall tau over the intersection; if both sides expose edges,
  also compute edge Jaccard; otherwise fall back to a 10-bin rank Jaccard.
* ``(tree, tree)`` — edge-set Jaccard + root agreement.

Undefined cells (e.g. tree-shaped LLM output paired with a flat-shaped
baseline that has no edges and no rankable surrogate) raise
:class:`ComparisonProtocolError` rather than silently returning ``None``.
"""
from __future__ import annotations

from typing import Literal, Optional

import pandas as pd
from pydantic import BaseModel
from scipy.stats import kendalltau, spearmanr  # type: ignore[import-untyped]

from mcp_litchron.errors import ComparisonProtocolError

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
Shape = Literal["flat_ranking", "tree"]


class ComparisonRow(BaseModel):
    """One row of the comparison table: LLM vs. one baseline."""

    baseline: str
    llm_shape: Shape
    baseline_shape: Shape
    spearman: Optional[float] = None
    kendall_tau: Optional[float] = None
    jaccard_edges: Optional[float] = None
    rank_bin_jaccard: Optional[float] = None
    root_cell_agreement: Optional[bool] = None
    notes: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _shape(edges: Optional[list[tuple[str, str]]]) -> Shape:
    return "tree" if edges else "flat_ranking"


def _looks_like_cell_index(series: pd.Series) -> bool:
    """Heuristic: a series is "per-cell" when its index resembles cell IDs.

    Cell IDs in scanpy/anndata are typically strings starting with ``cell_``,
    a barcode (``A...T-1``), or simply unique non-numeric labels in numbers
    far larger than a typical cluster count. We treat any index with more
    than 50 unique entries as per-cell — the proposal layer never emits
    more than a few dozen cell types in practice.
    """
    return len(series.index) > 50


def _intersect_align(
    a: pd.Series, b: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """Return ``a`` and ``b`` reduced to their shared index, in matched order."""
    common = a.index.intersection(b.index)
    return a.loc[common], b.loc[common]


def _spearman_kendall(a: pd.Series, b: pd.Series) -> tuple[Optional[float], Optional[float]]:
    a2, b2 = _intersect_align(a, b)
    if len(a2) < 2:
        return None, None
    sp = spearmanr(a2.values, b2.values).statistic
    kt = kendalltau(a2.values, b2.values).statistic
    sp_f = float(sp) if sp == sp else None  # NaN guard
    kt_f = float(kt) if kt == kt else None
    return sp_f, kt_f


def _edge_jaccard(
    e1: list[tuple[str, str]],
    e2: list[tuple[str, str]],
) -> float:
    """Jaccard similarity of two edge sets (undirected treatment)."""
    s1 = {tuple(sorted(e)) for e in e1}
    s2 = {tuple(sorted(e)) for e in e2}
    if not s1 and not s2:
        return 1.0
    union = s1 | s2
    if not union:
        return 1.0
    return len(s1 & s2) / len(union)


def _linearize_tree(
    edges: list[tuple[str, str]], root: Optional[str]
) -> pd.Series:
    """Rank nodes by BFS depth from ``root`` (v1 heuristic).

    If ``root`` is None, pick the node with no incoming edge (or fall back
    to ``edges[0][0]``). Ties broken alphabetically for stability.
    """
    # Build adjacency (directed parent → child if root supplied).
    children: dict[str, list[str]] = {}
    incoming: dict[str, int] = {}
    nodes: set[str] = set()
    for p, c in edges:
        nodes.add(p)
        nodes.add(c)
        children.setdefault(p, []).append(c)
        incoming[c] = incoming.get(c, 0) + 1
        incoming.setdefault(p, incoming.get(p, 0))

    if root is None:
        roots = [n for n in nodes if incoming.get(n, 0) == 0]
        root = sorted(roots)[0] if roots else (edges[0][0] if edges else "")

    depths: dict[str, int] = {root: 0} if root else {}
    frontier: list[str] = [root] if root else []
    while frontier:
        nxt: list[str] = []
        for parent in frontier:
            for child in sorted(children.get(parent, [])):
                if child not in depths:
                    depths[child] = depths[parent] + 1
                    nxt.append(child)
        frontier = nxt

    # Any disconnected nodes get a max-depth+1 sentinel so they still rank.
    if depths:
        sentinel = max(depths.values()) + 1
    else:
        sentinel = 0
    for n in nodes:
        depths.setdefault(n, sentinel)
    return pd.Series(depths, name="depth").sort_index()


def _rank_bin_jaccard(a: pd.Series, b: pd.Series, n_bins: int = 10) -> Optional[float]:
    """Bin-and-Jaccard rank similarity over the intersection."""
    a2, b2 = _intersect_align(a, b)
    if len(a2) < 2:
        return None
    a_bins = pd.qcut(a2.rank(method="average"), q=min(n_bins, len(a2)), labels=False, duplicates="drop")
    b_bins = pd.qcut(b2.rank(method="average"), q=min(n_bins, len(b2)), labels=False, duplicates="drop")
    agree = (a_bins == b_bins).sum()
    return float(agree) / float(len(a2))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compare(
    llm_ordering: pd.Series,
    baseline_ordering: pd.Series,
    llm_edges: Optional[list[tuple[str, str]]] = None,
    baseline_edges: Optional[list[tuple[str, str]]] = None,
    llm_root: Optional[str] = None,
    baseline_root: Optional[str] = None,
    baseline_name: str = "unknown",
) -> ComparisonRow:
    """Return a :class:`ComparisonRow` for the (LLM, baseline) pair.

    See module docstring for the comparison matrix. Undefined cells raise
    :class:`ComparisonProtocolError`.
    """
    llm_shape: Shape = _shape(llm_edges)
    baseline_shape: Shape = _shape(baseline_edges)

    # -- per-cell × per-cell shortcut ------------------------------------
    # When both sides are per-cell (large index, no edges), the natural
    # comparison is direct Spearman/Kendall on the cell-ID intersection.
    # No cluster aggregation, no tree linearization. This is the path the
    # LitChron continuous pseudotime uses when paired with a baseline's
    # per-cell ordering.parquet.
    if (
        llm_shape == "flat_ranking"
        and baseline_shape == "flat_ranking"
        and _looks_like_cell_index(llm_ordering)
        and _looks_like_cell_index(baseline_ordering)
    ):
        sp, kt = _spearman_kendall(llm_ordering, baseline_ordering)
        return ComparisonRow(
            baseline=baseline_name,
            llm_shape=llm_shape,
            baseline_shape=baseline_shape,
            spearman=sp,
            kendall_tau=kt,
            notes="per-cell vs per-cell: Spearman + Kendall tau over cell-ID intersection",
        )

    # -- (tree, tree) -----------------------------------------------------
    if llm_shape == "tree" and baseline_shape == "tree":
        assert llm_edges is not None and baseline_edges is not None
        j = _edge_jaccard(llm_edges, baseline_edges)
        root_agree: Optional[bool] = None
        if llm_root is not None and baseline_root is not None:
            root_agree = llm_root == baseline_root
        return ComparisonRow(
            baseline=baseline_name,
            llm_shape=llm_shape,
            baseline_shape=baseline_shape,
            jaccard_edges=j,
            root_cell_agreement=root_agree,
            notes="tree-vs-tree: edge Jaccard + root agreement",
        )

    # -- (flat, flat) -----------------------------------------------------
    if llm_shape == "flat_ranking" and baseline_shape == "flat_ranking":
        sp, kt = _spearman_kendall(llm_ordering, baseline_ordering)
        return ComparisonRow(
            baseline=baseline_name,
            llm_shape=llm_shape,
            baseline_shape=baseline_shape,
            spearman=sp,
            kendall_tau=kt,
            notes="flat-vs-flat: Spearman + Kendall tau over intersection",
        )

    # -- (flat, tree) or (tree, flat) -------------------------------------
    if {llm_shape, baseline_shape} == {"flat_ranking", "tree"}:
        if llm_shape == "tree":
            assert llm_edges is not None
            llm_linear = _linearize_tree(llm_edges, llm_root)
            flat = baseline_ordering
        else:
            assert baseline_edges is not None
            baseline_linear = _linearize_tree(baseline_edges, baseline_root)
            llm_linear = llm_ordering  # type: ignore[assignment]
            flat = baseline_linear

        # We want Spearman/Kendall between the linearized tree and the flat.
        if llm_shape == "tree":
            sp, kt = _spearman_kendall(llm_linear, flat)
        else:
            sp, kt = _spearman_kendall(llm_ordering, flat)

        # Edge Jaccard requires both sides expose edges → not the case here.
        # Fall back to 10-bin rank Jaccard over the intersection.
        if llm_shape == "tree":
            rbj = _rank_bin_jaccard(llm_linear, flat)
        else:
            rbj = _rank_bin_jaccard(llm_ordering, flat)

        return ComparisonRow(
            baseline=baseline_name,
            llm_shape=llm_shape,
            baseline_shape=baseline_shape,
            spearman=sp,
            kendall_tau=kt,
            rank_bin_jaccard=rbj,
            notes="mixed-shape: linearize tree by edge depth, then Spearman/Kendall + 10-bin rank Jaccard",
        )

    # -- Undefined -------------------------------------------------------
    raise ComparisonProtocolError(
        code="undefined_cell",
        message=(
            f"No comparison rule for (llm_shape={llm_shape!r}, "
            f"baseline_shape={baseline_shape!r}) against baseline "
            f"{baseline_name!r}"
        ),
        hint=(
            "Provide either flat orderings on both sides or edges on at least "
            "one side. See litchron.compare module docstring."
        ),
        retryable=False,
    )


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def comparison_to_markdown(rows: list[ComparisonRow]) -> str:
    """Render a list of comparison rows as a markdown table."""
    header = (
        "| baseline | llm_shape | baseline_shape | spearman | kendall_tau "
        "| jaccard_edges | rank_bin_jaccard | root_agreement | notes |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|"

    def fmt(v: Optional[float | bool | str]) -> str:
        if v is None:
            return "—"
        if isinstance(v, bool):
            return "yes" if v else "no"
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    lines: list[str] = ["# Comparison", "", header, sep]
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    r.baseline,
                    r.llm_shape,
                    r.baseline_shape,
                    fmt(r.spearman),
                    fmt(r.kendall_tau),
                    fmt(r.jaccard_edges),
                    fmt(r.rank_bin_jaccard),
                    fmt(r.root_cell_agreement),
                    r.notes.replace("|", "\\|"),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)
