"""pyslingshot in-process baseline.

Wraps the pure-Python ``pyslingshot`` port of the R Slingshot algorithm.
Operates on the existing PCA embedding and cluster column.
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

_METHOD = "pyslingshot"
_CLUSTER_CANDIDATES: tuple[str, ...] = ("leiden", "louvain", "cell_type")


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _cluster_col(adata: Any) -> str:
    for cand in _CLUSTER_CANDIDATES:
        if cand in adata.obs.columns:
            return cand
    raise RuntimeError(
        "pyslingshot requires a cluster column (leiden / louvain / cell_type) "
        "in adata.obs"
    )


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
    obs.create_dataset(
        "pyslingshot_pseudotime", data=arr, shape=arr.shape, dtype="f8"
    )


def _save_plot(embedding: np.ndarray, pseudotime: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    if embedding.shape[1] >= 2:
        ax.scatter(embedding[:, 0], embedding[:, 1], c=pseudotime, s=2)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    else:
        ax.plot(np.sort(pseudotime))
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_pyslingshot(adata: Any, run_dir: Path) -> BaselineResult:
    """Run pyslingshot on ``adata.obsm['X_pca']`` and persist artifacts."""
    out_dir = Path(run_dir) / "baselines" / _METHOD
    out_dir.mkdir(parents=True, exist_ok=True)
    ordering_path = out_dir / "ordering.parquet"
    plot_path = out_dir / "plot.png"
    log_path = out_dir / "log.txt"
    delta_path = out_dir / "adata_delta.zarr"

    log_lines: list[str] = [f"[{_now()}] pyslingshot.run_pyslingshot start"]

    try:
        from pyslingshot import Slingshot

        log_lines.append(f"[{_now()}] pyslingshot import ok")

        if "X_pca" not in adata.obsm:
            raise RuntimeError(
                "pyslingshot requires adata.obsm['X_pca']; run sc.pp.pca first"
            )
        cluster_col = _cluster_col(adata)
        log_lines.append(f"[{_now()}] cluster_col={cluster_col}")

        col = adata.obs[cluster_col]
        try:
            labels = col.cat.codes.values
        except AttributeError:
            # Coerce to categorical on the fly.
            import pandas as pd

            labels = pd.Categorical(col).codes
        labels = np.asarray(labels).astype(np.int64)
        embedding = np.asarray(adata.obsm["X_pca"], dtype=float)

        slingshot = Slingshot(data=embedding, cluster_labels=labels)
        slingshot.fit()
        log_lines.append(f"[{_now()}] Slingshot.fit done")

        pst_raw = getattr(slingshot, "pseudotime", None)
        if pst_raw is None:
            raise RuntimeError(
                "Slingshot.fit produced no `pseudotime` attribute"
            )
        # pyslingshot returns per-lineage pseudotimes; collapse to a per-cell
        # mean for the canonical ordering column.
        pst_arr = np.asarray(pst_raw, dtype=float)
        if pst_arr.ndim == 2:
            pst = np.nanmean(pst_arr, axis=1)
        else:
            pst = pst_arr
        # Replace NaN with the max finite value so the ordering is total.
        if np.isnan(pst).any():
            finite = pst[np.isfinite(pst)]
            fill = float(np.max(finite)) if finite.size else 0.0
            pst = np.where(np.isnan(pst), fill, pst)

        cell_ids = [str(x) for x in adata.obs_names]
        cell_types = [str(x) for x in adata.obs[cluster_col]]

        _write_ordering(ordering_path, cell_ids, pst, cell_types)
        log_lines.append(f"[{_now()}] wrote {ordering_path.name}")

        _save_plot(embedding, pst, plot_path)
        log_lines.append(f"[{_now()}] wrote {plot_path.name}")

        _write_delta_zarr(delta_path, pst)
        log_lines.append(f"[{_now()}] wrote {delta_path.name} (v1 minimal)")

        log_lines.append(f"[{_now()}] pyslingshot.run_pyslingshot done")
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

        return BaselineResult(
            method=_METHOD,
            ordering_path=str(ordering_path),
            lineage_edges=None,
            root_cell=None,
            figure_path=str(plot_path),
            delta_zarr_path=str(delta_path),
            adata_delta_keys=["obs/pyslingshot_pseudotime"],
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
            code="pyslingshot_runtime_error",
            message=f"pyslingshot baseline failed: {type(exc).__name__}: {exc}",
            hint=(
                "Install pyslingshot (`pip install pyslingshot`) and ensure "
                "adata.obsm['X_pca'] and a cluster column exist."
            ),
            retryable=False,
            method=_METHOD,
        ) from exc


__all__ = ["run_pyslingshot"]
