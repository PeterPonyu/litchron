"""PAGA + DPT in-process baseline.

Runs Scanpy's PAGA on the existing or freshly-computed cluster column,
then a diffusion-map / DPT pseudotime with a heuristic root cell. The
artifact layout is the common LitChron baseline layout:

    <run_dir>/baselines/paga/
        ordering.parquet  -- cell_id, pseudotime, cell_type
        plot.png          -- PAGA graph (matplotlib Agg backend)
        log.txt           -- run timestamps + parameters
        adata_delta.zarr  -- minimal delta proving replay (dpt_pseudotime)

Any third-party exception is caught and re-raised as
:class:`BaselineFailure` so the MCP server can return a structured error.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive backend (must come before pyplot).
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from mcp_litchron.errors import BaselineFailure  # noqa: E402

from ..config import RANDOM_SEED  # noqa: E402
from . import BaselineResult  # noqa: E402

_METHOD = "paga"
_CLUSTER_CANDIDATES: tuple[str, ...] = ("leiden", "louvain", "cell_type")


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _pick_cluster_column(adata: Any) -> str:
    """Return an existing cluster column, otherwise run leiden and return 'leiden'."""
    for cand in _CLUSTER_CANDIDATES:
        if cand in adata.obs.columns:
            return cand
    import scanpy as sc

    sc.tl.leiden(adata, resolution=0.8, random_state=RANDOM_SEED)
    return "leiden"


def _ensure_neighbors(adata: Any) -> None:
    """Run sc.pp.neighbors if there is no precomputed neighbor graph."""
    if "neighbors" in adata.uns:
        return
    import scanpy as sc

    sc.pp.neighbors(adata, n_neighbors=15, random_state=RANDOM_SEED)


def _heuristic_root_cell(adata: Any) -> str:
    """Pick a root cell as the one with the minimum value on diffmap[:,1].

    Component 0 of the diffusion map is conventionally trivial (the
    constant eigenvector), so we project on component 1.
    """
    X_dm = adata.obsm["X_diffmap"]
    if X_dm.shape[1] < 2:
        idx = 0
    else:
        idx = int(np.argmin(X_dm[:, 1]))
    return str(adata.obs_names[idx])


def _extract_lineage_edges(adata: Any, threshold: float = 0.1) -> list[tuple[str, str]]:
    """Threshold the PAGA connectivities to a list of (group_a, group_b) edges."""
    paga = adata.uns.get("paga") or {}
    conn = paga.get("connectivities")
    if conn is None:
        return []
    # PAGA stores a sparse symmetric matrix; the groups are in obs[groupby].cat.categories
    # but PAGA also stamps the group order on uns['paga']['groups_key'].
    groups_key = paga.get("groups") or paga.get("groups_key")
    if isinstance(groups_key, str) and groups_key in adata.obs.columns:
        try:
            categories = [str(c) for c in adata.obs[groups_key].cat.categories]
        except AttributeError:
            categories = sorted(map(str, set(adata.obs[groups_key])))
    else:
        # Fall back to integer indices as labels.
        n = conn.shape[0]
        categories = [str(i) for i in range(n)]

    coo = conn.tocoo() if hasattr(conn, "tocoo") else None
    edges: list[tuple[str, str]] = []
    if coo is None:
        # Dense fallback.
        arr = np.asarray(conn)
        n = arr.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if arr[i, j] > threshold:
                    edges.append((categories[i], categories[j]))
        return edges

    seen: set[tuple[int, int]] = set()
    for i, j, v in zip(coo.row, coo.col, coo.data):
        if i >= j:
            continue
        if v <= threshold:
            continue
        key = (int(i), int(j))
        if key in seen:
            continue
        seen.add(key)
        edges.append((categories[int(i)], categories[int(j)]))
    return edges


def _write_ordering(
    out_path: Path,
    cell_ids: list[str],
    pseudotime: np.ndarray,
    cell_types: list[str],
) -> None:
    """Write ordering.parquet with stable schema."""
    table = pa.table(
        {
            "cell_id": pa.array(cell_ids, type=pa.string()),
            "pseudotime": pa.array(pseudotime.astype(float), type=pa.float64()),
            "cell_type": pa.array(cell_types, type=pa.string()),
        }
    )
    pq.write_table(table, str(out_path))


def _write_delta_zarr(out_path: Path, pseudotime: np.ndarray) -> None:
    """Persist a minimal delta proving replay (v1 simplification).

    A full delta would round-trip ``uns/paga``, ``uns/diffmap_evals``,
    ``obs/dpt_pseudotime`` and ``obsm/X_diffmap``. For v1 we persist just
    ``obs/dpt_pseudotime`` — enough to verify the cache-replay protocol
    without depending on the unstable scanpy uns layout.
    """
    import zarr

    root = zarr.open(str(out_path), mode="w")
    obs = root.create_group("obs")
    arr = np.asarray(pseudotime, dtype=np.float64)
    obs.create_dataset("dpt_pseudotime", data=arr, shape=arr.shape, dtype="f8")


def _save_plot(adata: Any, out_path: Path) -> None:
    """Render a PAGA graph and save to PNG via the Agg backend."""
    import scanpy as sc

    fig, ax = plt.subplots(figsize=(6, 6))
    try:
        sc.pl.paga(adata, show=False, ax=ax)
    except Exception:
        # Fallback: scatter diffmap[:,1] vs [:,2] colored by pseudotime.
        X_dm = adata.obsm.get("X_diffmap")
        pst = adata.obs.get("dpt_pseudotime")
        if X_dm is not None and X_dm.shape[1] >= 3 and pst is not None:
            ax.scatter(X_dm[:, 1], X_dm[:, 2], c=np.asarray(pst), s=2)
            ax.set_xlabel("DC1")
            ax.set_ylabel("DC2")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_paga(adata: Any, run_dir: Path) -> BaselineResult:
    """Run PAGA + DPT on ``adata`` and persist the LitChron baseline artifacts."""
    out_dir = Path(run_dir) / "baselines" / _METHOD
    out_dir.mkdir(parents=True, exist_ok=True)
    ordering_path = out_dir / "ordering.parquet"
    plot_path = out_dir / "plot.png"
    log_path = out_dir / "log.txt"
    delta_path = out_dir / "adata_delta.zarr"

    log_lines: list[str] = [f"[{_now()}] paga.run_paga start"]

    try:
        import scanpy as sc

        cluster_col = _pick_cluster_column(adata)
        log_lines.append(f"[{_now()}] cluster_col={cluster_col}")

        _ensure_neighbors(adata)
        log_lines.append(f"[{_now()}] neighbors ready (n_neighbors=15)")

        sc.tl.paga(adata, groups=cluster_col)
        log_lines.append(f"[{_now()}] sc.tl.paga done")

        sc.tl.diffmap(adata, n_comps=15)
        log_lines.append(f"[{_now()}] sc.tl.diffmap n_comps=15 done")

        root_cell_name = _heuristic_root_cell(adata)
        root_iloc = int(adata.obs_names.get_loc(root_cell_name))
        adata.uns["iroot"] = root_iloc
        log_lines.append(
            f"[{_now()}] iroot={root_iloc} root_cell={root_cell_name!r}"
        )

        sc.tl.dpt(adata, n_dcs=15)
        log_lines.append(f"[{_now()}] sc.tl.dpt n_dcs=15 done")

        pst = np.asarray(adata.obs["dpt_pseudotime"], dtype=float)
        cell_ids = [str(x) for x in adata.obs_names]
        cell_types = [str(x) for x in adata.obs[cluster_col]]

        _write_ordering(ordering_path, cell_ids, pst, cell_types)
        log_lines.append(f"[{_now()}] wrote {ordering_path.name}")

        _save_plot(adata, plot_path)
        log_lines.append(f"[{_now()}] wrote {plot_path.name}")

        edges = _extract_lineage_edges(adata, threshold=0.1)
        log_lines.append(f"[{_now()}] extracted {len(edges)} lineage edges")

        _write_delta_zarr(delta_path, pst)
        log_lines.append(f"[{_now()}] wrote {delta_path.name} (v1 minimal)")

        log_lines.append(f"[{_now()}] paga.run_paga done")
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

        return BaselineResult(
            method=_METHOD,
            ordering_path=str(ordering_path),
            lineage_edges=edges or None,
            root_cell=root_cell_name,
            figure_path=str(plot_path),
            delta_zarr_path=str(delta_path),
            adata_delta_keys=[
                "uns/paga",
                "uns/diffmap_evals",
                "obs/dpt_pseudotime",
                "obsm/X_diffmap",
            ],
        )
    except BaselineFailure:
        raise
    except Exception as exc:  # noqa: BLE001 -- catch-and-wrap by design.
        log_lines.append(f"[{_now()}] FAIL {type(exc).__name__}: {exc}")
        try:
            log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        except OSError:
            pass
        raise BaselineFailure(
            code="paga_runtime_error",
            message=f"PAGA baseline failed: {type(exc).__name__}: {exc}",
            hint=(
                "Check that adata has a clusterable embedding and that scanpy "
                "is installed."
            ),
            retryable=False,
            method=_METHOD,
        ) from exc


__all__ = ["run_paga"]
