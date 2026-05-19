"""Palantir in-process baseline.

Wraps the upstream ``palantir`` package's diffusion-map + multiscale
pseudotime. Same artifact layout as :mod:`litchron.baselines.paga`:

    <run_dir>/baselines/palantir/
        ordering.parquet
        plot.png
        log.txt
        adata_delta.zarr
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

_METHOD = "palantir"
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
    """Minimal v1 delta — just the pseudotime obs vector."""
    import zarr

    root = zarr.open(str(out_path), mode="w")
    obs = root.create_group("obs")
    arr = np.asarray(pseudotime, dtype=np.float64)
    obs.create_dataset("palantir_pseudotime", data=arr, shape=arr.shape, dtype="f8")


def _pick_early_cell(adata: Any) -> str:
    """Pick the cell with the smallest value along the top diffusion component."""
    obsm = adata.obsm
    for key in ("DM_EigenVectors_multiscaled", "DM_EigenVectors", "X_diffmap", "X_pca"):
        if key in obsm:
            X = obsm[key]
            col = 1 if X.shape[1] > 1 else 0
            idx = int(np.argmin(X[:, col]))
            return str(adata.obs_names[idx])
    return str(adata.obs_names[0])


def _save_plot(adata: Any, pseudotime: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    obsm = adata.obsm
    proj = None
    for key in ("X_umap", "X_diffmap", "X_pca"):
        if key in obsm and obsm[key].shape[1] >= 2:
            proj = obsm[key]
            break
    if proj is not None:
        ax.scatter(proj[:, 0], proj[:, 1], c=pseudotime, s=2)
        ax.set_xlabel("dim0")
        ax.set_ylabel("dim1")
    else:
        ax.plot(np.sort(pseudotime))
        ax.set_xlabel("rank")
        ax.set_ylabel("pseudotime")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_palantir(adata: Any, run_dir: Path) -> BaselineResult:
    """Run Palantir on ``adata`` and persist LitChron baseline artifacts."""
    out_dir = Path(run_dir) / "baselines" / _METHOD
    out_dir.mkdir(parents=True, exist_ok=True)
    ordering_path = out_dir / "ordering.parquet"
    plot_path = out_dir / "plot.png"
    log_path = out_dir / "log.txt"
    delta_path = out_dir / "adata_delta.zarr"

    log_lines: list[str] = [f"[{_now()}] palantir.run_palantir start"]

    try:
        import palantir  # noqa: F401

        log_lines.append(f"[{_now()}] palantir import ok")

        # Diffusion maps + multiscale space.
        try:
            palantir.utils.run_diffusion_maps(adata, n_components=10)
        except AttributeError:
            # Older palantir versions exposed this under palantir.run.
            palantir.run.run_diffusion_maps(adata, n_components=10)
        log_lines.append(f"[{_now()}] run_diffusion_maps done")

        try:
            palantir.utils.determine_multiscale_space(adata)
        except AttributeError:
            palantir.run.determine_multiscale_space(adata)
        log_lines.append(f"[{_now()}] determine_multiscale_space done")

        early_cell = _pick_early_cell(adata)
        log_lines.append(f"[{_now()}] early_cell={early_cell!r}")

        # Core call. Newer Palantir API stores results on adata.obs.
        try:
            res = palantir.core.run_palantir(
                adata,
                early_cell,
                terminal_states=None,
                num_waypoints=500,
            )
        except TypeError:
            # Some versions expect a DataFrame, not an AnnData.
            ms = adata.obsm.get("DM_EigenVectors_multiscaled")
            if ms is None:
                raise
            import pandas as pd

            ms_df = pd.DataFrame(ms, index=adata.obs_names)
            res = palantir.core.run_palantir(
                ms_df, early_cell, terminal_states=None, num_waypoints=500
            )
        log_lines.append(f"[{_now()}] palantir.core.run_palantir done")

        # Extract pseudotime from either obs or the result object.
        if "palantir_pseudotime" in adata.obs.columns:
            pst = np.asarray(adata.obs["palantir_pseudotime"], dtype=float)
        else:
            pst_series = getattr(res, "pseudotime", None)
            if pst_series is None and isinstance(res, dict):
                pst_series = res.get("pseudotime")
            if pst_series is None:
                raise RuntimeError(
                    "Palantir result has no `pseudotime` attribute or obs column"
                )
            pst = np.asarray(pst_series, dtype=float)

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

        log_lines.append(f"[{_now()}] palantir.run_palantir done")
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

        return BaselineResult(
            method=_METHOD,
            ordering_path=str(ordering_path),
            lineage_edges=None,
            root_cell=early_cell,
            figure_path=str(plot_path),
            delta_zarr_path=str(delta_path),
            adata_delta_keys=[
                "obs/palantir_pseudotime",
                "obsm/DM_EigenVectors",
                "obsm/DM_EigenVectors_multiscaled",
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
            code="palantir_runtime_error",
            message=f"Palantir baseline failed: {type(exc).__name__}: {exc}",
            hint=(
                "Install palantir (`pip install palantir`) and ensure the "
                "adata has a clusterable embedding."
            ),
            retryable=False,
            method=_METHOD,
        ) from exc


__all__ = ["run_palantir"]
