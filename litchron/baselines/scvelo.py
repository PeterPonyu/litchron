"""scVelo in-process baseline.

Computes a stochastic-mode velocity pseudotime via ``scvelo``. Requires
``Ms``/``Mu`` layers (or ``spliced``/``unspliced`` from which moments
are computed). Same artifact layout as the other Python baselines.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from mcp_litchron.errors import BaselineFailure  # noqa: E402

from . import BaselineResult  # noqa: E402

_METHOD = "scvelo"
_CLUSTER_CANDIDATES: tuple[str, ...] = ("leiden", "louvain", "cell_type")


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _cluster_col(adata: Any) -> str:
    for cand in _CLUSTER_CANDIDATES:
        if cand in adata.obs.columns:
            return cand
    return ""


def _write_ordering(
    out_path: Path,
    cell_ids: list[str],
    pseudotime: np.ndarray,
    cell_types: list[str],
) -> None:
    table = pa.table(
        {
            "cell_id": pa.array(cell_ids, type=pa.string()),
            "pseudotime": pa.array(pseudotime.astype(float), type=pa.float64()),
            "cell_type": pa.array(cell_types, type=pa.string()),
        }
    )
    pq.write_table(table, str(out_path))


def _write_delta_zarr(out_path: Path, pseudotime: np.ndarray) -> None:
    import zarr

    root = zarr.open(str(out_path), mode="w")
    obs = root.create_group("obs")
    arr = np.asarray(pseudotime, dtype=np.float64)
    obs.create_dataset("velocity_pseudotime", data=arr, shape=arr.shape, dtype="f8")


def _save_plot(adata: Any, pseudotime: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    obsm = adata.obsm
    proj = None
    for key in ("X_umap", "X_pca", "X_diffmap"):
        if key in obsm and obsm[key].shape[1] >= 2:
            proj = obsm[key]
            break
    if proj is not None:
        ax.scatter(proj[:, 0], proj[:, 1], c=pseudotime, s=2)
        ax.set_xlabel("dim0")
        ax.set_ylabel("dim1")
    else:
        ax.plot(np.sort(pseudotime))
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_scvelo(adata: Any, run_dir: Path) -> BaselineResult:
    """Run scVelo stochastic velocity + pseudotime on ``adata``."""
    out_dir = Path(run_dir) / "baselines" / _METHOD
    out_dir.mkdir(parents=True, exist_ok=True)
    ordering_path = out_dir / "ordering.parquet"
    plot_path = out_dir / "plot.png"
    log_path = out_dir / "log.txt"
    delta_path = out_dir / "adata_delta.zarr"

    log_lines: list[str] = [f"[{_now()}] scvelo.run_scvelo start"]

    layers = getattr(adata, "layers", None) or {}
    has_ms_mu = "Ms" in layers and "Mu" in layers
    has_spliced = "spliced" in layers and "unspliced" in layers
    if not has_ms_mu and not has_spliced:
        log_lines.append(f"[{_now()}] FAIL preprocessing layers missing")
        try:
            log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        except OSError:
            pass
        raise BaselineFailure(
            code="scvelo_preprocessing_missing",
            message=(
                "scVelo requires Ms/Mu layers or spliced/unspliced to "
                "compute moments."
            ),
            hint=(
                "Pre-process the AnnData with scvelo.pp.filter_and_normalize "
                "and scvelo.pp.moments, or load a dataset with spliced/"
                "unspliced layers."
            ),
            retryable=False,
            method=_METHOD,
        )

    try:
        import scvelo as scv

        log_lines.append(f"[{_now()}] scvelo import ok")

        if not has_ms_mu and has_spliced:
            scv.pp.moments(adata)
            log_lines.append(f"[{_now()}] scvelo.pp.moments computed Ms/Mu")

        scv.tl.velocity(adata, mode="stochastic")
        log_lines.append(f"[{_now()}] scv.tl.velocity stochastic done")

        scv.tl.velocity_graph(adata)
        log_lines.append(f"[{_now()}] scv.tl.velocity_graph done")

        scv.tl.velocity_pseudotime(adata)
        log_lines.append(f"[{_now()}] scv.tl.velocity_pseudotime done")

        pst = np.asarray(adata.obs["velocity_pseudotime"], dtype=float)
        cell_ids = [str(x) for x in adata.obs_names]
        cluster_col = _cluster_col(adata)
        cell_types = (
            [str(x) for x in adata.obs[cluster_col]] if cluster_col else [""] * len(cell_ids)
        )

        _write_ordering(ordering_path, cell_ids, pst, cell_types)
        log_lines.append(f"[{_now()}] wrote {ordering_path.name}")

        _save_plot(adata, pst, plot_path)
        log_lines.append(f"[{_now()}] wrote {plot_path.name}")

        _write_delta_zarr(delta_path, pst)
        log_lines.append(f"[{_now()}] wrote {delta_path.name} (v1 minimal)")

        log_lines.append(f"[{_now()}] scvelo.run_scvelo done")
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

        return BaselineResult(
            method=_METHOD,
            ordering_path=str(ordering_path),
            lineage_edges=None,
            root_cell=None,
            figure_path=str(plot_path),
            delta_zarr_path=str(delta_path),
            adata_delta_keys=[
                "obs/velocity_pseudotime",
                "layers/velocity",
                "uns/velocity_graph",
            ],
        )
    except BaselineFailure:
        raise
    except Exception as exc:  # noqa: BLE001
        log_lines.append(f"[{_now()}] FAIL {type(exc).__name__}: {exc}")
        try:
            log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        except OSError:
            pass
        raise BaselineFailure(
            code="scvelo_runtime_error",
            message=f"scVelo baseline failed: {type(exc).__name__}: {exc}",
            hint=(
                "Install scvelo (`pip install scvelo`) and confirm spliced/"
                "unspliced layers are present."
            ),
            retryable=False,
            method=_METHOD,
        ) from exc


__all__ = ["run_scvelo"]
