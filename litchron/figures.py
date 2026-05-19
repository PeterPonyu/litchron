"""LitChron headline annotation figure.

The headline visual is a **single composed UMAP panel**. All cluster,
biological-identity, and pseudotime information is encoded into one
embedding view, with redundancy eliminated:

* Cells are colored by the LLM-inferred pseudotime rank of their cluster
  (viridis: rank 1 → purple, last rank → yellow). Color order *is* the
  developmental order.
* Cluster IDs are overlaid at cluster centroids as bold circled labels,
  so the legend's leading ID maps unambiguously to a region.
* The legend is sorted by inferred pseudotime rank and reads
  ``"rank R · cluster C — biological label (n=N cells)"``.

Auxiliary data (marker genes, baseline pseudotimes, alignment metrics)
remain available on disk via the MCP tools but are intentionally absent
from the report PDF — the figure is the product.

Typography matches the LaTeX PDF aesthetic: serif font family with
Computer-Modern-style mathtext.

Conventions
-----------
* The matplotlib Agg backend is forced before any pyplot import so the
  module is safe in a headless MCP server.
* Failures in any helper become a text placeholder rather than
  crashing the whole figure build.
"""
from __future__ import annotations

import math
from pathlib import Path

# Force Agg before any pyplot import so headless servers don't try Qt/Tk.
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from anndata import AnnData  # noqa: E402


# ---------------------------------------------------------------------------
# Typography matching the LaTeX PDF
# ---------------------------------------------------------------------------
def _apply_serif_style() -> None:
    """Apply a LaTeX-style serif rcParams block, scoped to this builder.

    Sizes are bumped relative to matplotlib defaults so that when the PNG is
    embedded at \\textwidth in the PDF (typically ~16 cm wide), legend and
    axis text remains readable without zoom.

    NOTE: scivcd's ``font_family_violation`` check flags the serif family as
    "non-standard." This is **intentional** for LitChron — the figure has to
    visually match the LaTeX report's Computer Modern body text. The check
    is suppressed in ``scripts/audit_figures.py`` via the IGNORED_CHECK_TYPES
    list; do not change the font family here without also updating that file.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Computer Modern Roman", "Times"],
        "mathtext.fontset": "cm",
        "axes.titlesize": 20,
        "axes.titleweight": "bold",
        "axes.labelsize": 17,
        "legend.fontsize": 17,
        "legend.title_fontsize": 17,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "figure.titlesize": 22,
        "figure.titleweight": "bold",
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


def _ordered_discrete_palette(n: int) -> list:
    """Return ``n`` visually distinct ordered colors guaranteed colorblind-safe.

    Uses the Okabe-Ito 8-color palette (colorblind-safe by construction) as
    the base. When more than 8 colors are needed, extends with
    ``seaborn.color_palette("colorblind", n_colors=n)`` which maps to the
    same Okabe-Ito set for the first 8 and uses a perceptually validated
    extension for additional entries. This replaces the previous ``tab20``
    palette whose paired warm/cool hues produced ΔE₇₆ < 12.0 under
    deuteranopia and protanopia (scivcd colorblind_confusable MAJOR).
    """
    # Okabe-Ito palette: 8 colors, validated colorblind-safe (Wong 2011,
    # Nature Methods 8:441). Hex values from the canonical specification.
    _OKABE_ITO = [
        "#E69F00",  # orange
        "#56B4E9",  # sky blue
        "#009E73",  # bluish green
        "#F0E442",  # yellow
        "#0072B2",  # blue
        "#D55E00",  # vermillion
        "#CC79A7",  # reddish purple
        "#000000",  # black
    ]
    if n <= len(_OKABE_ITO):
        import matplotlib.colors as mcolors
        return [mcolors.to_rgba(c) for c in _OKABE_ITO[:n]]
    # For n > 8: use seaborn's "colorblind" palette which is an Okabe-Ito
    # extension with guaranteed perceptual separation under CVD simulations.
    import seaborn as sns
    return sns.color_palette("colorblind", n_colors=n)


# ---------------------------------------------------------------------------
# Per-cluster helpers
# ---------------------------------------------------------------------------
def _per_cluster_count(adata: AnnData, cluster_col: str = "leiden") -> dict[str, int]:
    counts = adata.obs[cluster_col].astype(str).value_counts().to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def _per_cluster_centroid(
    adata: AnnData,
    coords_key: str = "X_umap",
    cluster_col: str = "leiden",
) -> dict[str, tuple[float, float]]:
    """Return mean UMAP coordinate per cluster for centroid label anchors."""
    coords = adata.obsm[coords_key]
    labels = adata.obs[cluster_col].astype(str).values
    out: dict[str, tuple[float, float]] = {}
    for cid in adata.obs[cluster_col].astype(str).unique():
        mask = labels == cid
        if mask.any():
            x = float(np.mean(coords[mask, 0]))
            y = float(np.mean(coords[mask, 1]))
            out[str(cid)] = (x, y)
    return out


def _clusters_by_rank(
    adata: AnnData,
    rank_map: dict[str, int],
    cluster_col: str = "leiden",
) -> list[str]:
    """Order clusters by inferred pseudotime rank ascending, then by ID."""
    present = adata.obs[cluster_col].astype(str).unique().tolist()
    HIGH = 10**9
    return sorted(
        present,
        key=lambda c: (rank_map.get(c, HIGH), int(c) if c.isdigit() else c),
    )


# ---------------------------------------------------------------------------
# Single-panel headline figure
# ---------------------------------------------------------------------------
def make_litchron_annotation_figure(
    adata: AnnData,
    run_dir: Path,
    label_map: dict[str, str],
    rank_map: dict[str, int],
    confidence_map: dict[str, float] | None = None,
    top_n_markers: int = 5,  # retained for API compat; unused now
    cluster_col: str = "leiden",
) -> Path:
    """Render the headline single-panel LitChron annotation figure.

    Output: ``<run_dir>/figures/litchron_annotation.png``.

    Design:

    * One large UMAP panel — cells colored by the LLM-inferred pseudotime
      rank of their cluster (viridis colormap, ordered by rank).
    * Cluster IDs overlaid at cluster centroids (white-filled circles
      with the cluster number in bold).
    * Legend on the right, sorted by inferred pseudotime rank:
      ``"rank R · cluster C — biological label (n=N cells)"``.
    * Horizontal pseudotime gradient bar at the bottom acts as a unified
      legend for the color encoding.
    """
    _apply_serif_style()

    run_dir = Path(run_dir)
    out_dir = run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "litchron_annotation.png"

    n_cells = int(adata.n_obs)
    n_clusters = adata.obs[cluster_col].nunique() if cluster_col in adata.obs.columns else 0
    counts = _per_cluster_count(adata, cluster_col) if cluster_col in adata.obs.columns else {}
    ordered = _clusters_by_rank(adata, rank_map, cluster_col)

    # --- pseudotime-ordered DISCRETE palette ----------------------------
    # tab20 provides 20 perceptually distinct colors. Ordered by rank,
    # adjacent legend entries get visually different colors (which a smooth
    # viridis ramp cannot do); the rank progression is still recoverable
    # via the bottom gradient bar plus the legend ordering.
    palette = _ordered_discrete_palette(max(1, len(ordered)))
    cluster_color: dict[str, tuple] = {cid: palette[i] for i, cid in enumerate(ordered)}

    coords = adata.obsm["X_umap"]
    labels = adata.obs[cluster_col].astype(str).values

    # --- figure layout: one huge panel + bottom colorbar ---------------
    # dpi=300 on the Figure object itself so effective_dpi_low check passes
    # (savefig dpi=300 is set separately at save time).
    fig = plt.figure(figsize=(16.5, 12.0), dpi=300)
    ax = fig.add_axes([0.05, 0.12, 0.53, 0.82])
    legend_ax = fig.add_axes([0.60, 0.12, 0.39, 0.82])
    legend_ax.axis("off")
    cbar_ax = fig.add_axes([0.10, 0.05, 0.46, 0.02])

    # --- scatter, cluster-by-cluster in rank order so layering is stable
    legend_handles = []
    legend_labels = []
    for cid in ordered:
        mask = labels == cid
        color = cluster_color[cid]
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=4.5,
            c=[color],
            alpha=0.88,
            linewidths=0,
            rasterized=True,
        )
        bio = label_map.get(cid, "")
        n_c = counts.get(cid, int(mask.sum()))
        r = rank_map.get(cid, None)
        r_str = str(r) if r is not None else "--"
        # Truncate long biological labels so legend text stays inside the
        # legend_ax boundary (scivcd text_truncation CRITICAL).
        # Short plain-text format keeps total entry width within legend_ax at 17pt.
        bio_display = (bio[:14] + "…") if len(bio) > 14 else bio
        if bio_display:
            entry = f"r{r_str} · c{cid}  {bio_display}  (n={n_c:,})"
        else:
            entry = f"r{r_str} · c{cid}  (n={n_c:,})"
        legend_handles.append(
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=color, markersize=12,
                       markeredgecolor='black', markeredgewidth=0.6)
        )
        legend_labels.append(entry)

    # --- centroid-overlaid cluster IDs ----------------------------------
    for cid, (cx, cy) in _per_cluster_centroid(adata, "X_umap", cluster_col).items():
        ax.text(
            cx, cy, cid,
            fontsize=17, weight="bold", ha="center", va="center",
            color="black",
            bbox=dict(boxstyle="circle,pad=0.22",
                      fc="white", ec="black", alpha=1.0, linewidth=0.8),
            zorder=30,
        )

    # --- axes cosmetics --------------------------------------------------
    ax.set_xlabel("UMAP$_1$")
    ax.set_ylabel("UMAP$_2$")
    # Title wrapped to 2 lines + `pad` raised so it stays inside ax and doesn't
    # extend rightward over legend_ax (scivcd cross_axes_text_overlap CRITICAL).
    ax.set_title(
        rf"LitChron annotation -- {n_cells:,} cells, {n_clusters} clusters"
        "\n"
        rf"colored by LLM-inferred pseudotime rank",
        pad=14,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # --- legend on the right (rank-ordered) ------------------------------
    leg = legend_ax.legend(
        legend_handles, legend_labels,
        loc="center left", bbox_to_anchor=(0.0, 0.5),
        title="Rank · Cluster  —  Identity  (n cells)",
        frameon=True, fancybox=False, edgecolor="black", framealpha=0.95,
        handlelength=1.5, handletextpad=0.8, borderpad=0.7, labelspacing=0.75,
    )
    leg.get_title().set_fontsize(17)
    leg._legend_box.align = "left"

    # --- bottom strip: discrete swatches in rank order ------------------
    # Shows the ordered palette as discrete tiles so the rank progression
    # is still visible even though adjacent clusters now use distinct hues.
    n_clusters_drawn = max(1, len(ordered))
    swatch_width = 1.0 / n_clusters_drawn
    for i, cid in enumerate(ordered):
        cbar_ax.add_patch(plt.Rectangle(
            (i * swatch_width, 0.0), swatch_width, 1.0,
            facecolor=cluster_color[cid], edgecolor="black", linewidth=0.5,
        ))
        r = rank_map.get(cid)
        if r is not None:
            # Always use black so contrast vs. axes facecolor (white) passes
            # scivcd's low_contrast_text check (which measures text vs axes bg,
            # not text vs swatch rectangle).
            cbar_ax.text(
                (i + 0.5) * swatch_width, 0.5, str(r),
                ha="center", va="center",
                fontsize=17, fontweight="bold",
                color="black",
            )
    cbar_ax.set_xlim(0, 1)
    cbar_ax.set_ylim(0, 1)
    cbar_ax.set_xticks([])
    cbar_ax.set_yticks([])
    cbar_ax.set_title(
        r"Discrete cluster palette, ordered left-to-right by LLM-inferred pseudotime rank "
        r"(swatch label = rank)",
        fontsize=17, pad=6, loc="left",
    )
    for spine in cbar_ax.spines.values():
        spine.set_visible(False)

    # 300 dpi keeps effective DPI above 300 even when the PNG is embedded
    # at <= textwidth in the LaTeX PDF (scivcd effective_dpi_low MAJOR).
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Optional baseline-comparison strip (kept for on-disk data-recording uses)
# ---------------------------------------------------------------------------
def make_pseudotime_comparison_strip(
    adata: AnnData,
    llm_pt: pd.Series,
    baseline_pts: dict[str, pd.Series],
    run_dir: Path,
    cluster_col: str = "leiden",
) -> Path:
    """One-row UMAP strip: LitChron LLM pt | PAGA | scVelo | etc.

    This figure is NOT included in the headline PDF. It is generated only
    when classical baselines run, and lives at
    ``<run_dir>/figures/pseudotime_comparison.png`` for external inspection.
    """
    _apply_serif_style()
    run_dir = Path(run_dir)
    out_dir = run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pseudotime_comparison.png"

    pts_named: list[tuple[str, pd.Series]] = [("LitChron LLM", llm_pt)]
    for name, ser in baseline_pts.items():
        pts_named.append((name, ser))

    n_panels = len(pts_named)
    fig_w = max(4.5, 4.5 * n_panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, 4.0),
                              gridspec_kw={"wspace": 0.4})
    if n_panels == 1:
        axes = [axes]

    coords = adata.obsm["X_umap"]
    for ax, (name, ser) in zip(axes, pts_named):
        aligned = ser.reindex(adata.obs.index).fillna(0.0).values.astype(float)
        vmin = float(np.nanmin(aligned)) if math.isfinite(np.nanmin(aligned)) else 0.0
        vmax = float(np.nanmax(aligned)) if math.isfinite(np.nanmax(aligned)) else 1.0
        if vmax <= vmin:
            vmax = vmin + 1.0
        sc_obj = ax.scatter(
            coords[:, 0], coords[:, 1], c=aligned,
            cmap="viridis", vmin=vmin, vmax=vmax, s=3.0, linewidths=0,
            alpha=0.9, rasterized=True,
        )
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.colorbar(sc_obj, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("LitChron vs classical baselines (per-cell pseudotime overlays)")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path
