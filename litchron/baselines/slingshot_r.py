"""Slingshot (R) baseline wrapper (subprocess-isolated).

Mirror of :mod:`litchron.baselines.monocle3` but ``method="slingshot_r"``.
This module never imports rpy2; the R interpreter lives in the child
process spawned via :mod:`litchron.baselines._r_runner`.
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import subprocess
import sys
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

_METHOD = "slingshot_r"
_TIMEOUT_S: int = 600


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _parse_ordering_csv(path: Path) -> tuple[list[str], np.ndarray]:
    cell_ids: list[str] = []
    pst: list[float] = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cell_ids.append(str(row["cell_id"]))
            pst.append(float(row["pseudotime"]))
    return cell_ids, np.asarray(pst, dtype=float)


def _parse_edges_csv(path: Path) -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            a = str(row.get("source") or row.get("from") or "")
            b = str(row.get("target") or row.get("to") or "")
            if a and b:
                edges.append((a, b))
    return edges


def _write_ordering_parquet(
    out_path: Path,
    cell_ids: list[str],
    pseudotime: np.ndarray,
) -> None:
    table = pa.table(
        {
            "cell_id": pa.array(cell_ids, type=pa.string()),
            "pseudotime": pa.array(pseudotime.astype(float), type=pa.float64()),
            "cell_type": pa.array([""] * len(cell_ids), type=pa.string()),
        }
    )
    pq.write_table(table, str(out_path))


def _save_plot(pseudotime: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.sort(pseudotime))
    ax.set_xlabel("rank")
    ax.set_ylabel("pseudotime")
    ax.set_title("slingshot (R) pseudotime (sorted)")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _write_delta_zarr(out_path: Path, pseudotime: np.ndarray) -> None:
    import zarr

    root = zarr.open(str(out_path), mode="w")
    obs = root.create_group("obs")
    arr = np.asarray(pseudotime, dtype=np.float64)
    obs.create_dataset(
        "slingshot_r_pseudotime", data=arr, shape=arr.shape, dtype="f8"
    )


def run_slingshot_r(run_id: str, h5ad_path: str, run_dir: Path) -> BaselineResult:
    """Spawn the R subprocess runner and persist Slingshot (R) artifacts."""
    out_dir = Path(run_dir) / "baselines" / _METHOD
    out_dir.mkdir(parents=True, exist_ok=True)
    ordering_path = out_dir / "ordering.parquet"
    plot_path = out_dir / "plot.png"
    log_path = out_dir / "log.txt"
    delta_path = out_dir / "adata_delta.zarr"

    log_lines: list[str] = [
        f"[{_now()}] slingshot_r.run_slingshot_r start run_id={run_id}"
    ]

    cmd = [
        sys.executable,
        "-m",
        "litchron.baselines._r_runner",
        _METHOD,
        run_id,
        h5ad_path,
        str(run_dir),
    ]
    log_lines.append(f"[{_now()}] spawn {' '.join(cmd)}")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=_TIMEOUT_S,
            text=True,
        )
    except subprocess.TimeoutExpired as exc:
        log_lines.append(f"[{_now()}] TIMEOUT after {_TIMEOUT_S}s")
        try:
            log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        except OSError:
            pass
        raise BaselineFailure(
            code="slingshot_r_timeout",
            message=f"slingshot_r subprocess exceeded {_TIMEOUT_S}s",
            hint="Retry with a smaller adata or raise the runner timeout.",
            retryable=True,
            method=_METHOD,
            stderr=None,
        ) from exc

    log_lines.append(f"[{_now()}] returncode={proc.returncode}")

    if proc.returncode in (-11, 139):
        try:
            log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        except OSError:
            pass
        raise BaselineFailure(
            code="segfault",
            message="slingshot_r subprocess crashed (SIGSEGV in libR).",
            hint=(
                "Re-run; if reproducible, downgrade slingshot / Bioconductor "
                "or report upstream."
            ),
            retryable=True,
            method=_METHOD,
            stderr=proc.stderr or None,
        )

    stdout = (proc.stdout or "").strip().splitlines()
    payload: dict[str, Any] | None = None
    for line in reversed(stdout):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            break
        except json.JSONDecodeError:
            continue

    if payload is None:
        try:
            log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        except OSError:
            pass
        raise BaselineFailure(
            code="slingshot_r_no_json",
            message="slingshot_r runner produced no JSON status line on stdout.",
            hint="Inspect stderr for the underlying R / rpy2 error.",
            retryable=False,
            method=_METHOD,
            stderr=proc.stderr or None,
        )

    if payload.get("status") != "ok" or proc.returncode != 0:
        try:
            log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        except OSError:
            pass
        raise BaselineFailure(
            code=str(payload.get("code") or "slingshot_r_error"),
            message=str(
                payload.get("message")
                or "slingshot_r runner reported non-ok status."
            ),
            hint="See stderr / runner log for details.",
            retryable=False,
            method=_METHOD,
            stderr=proc.stderr or None,
        )

    ordering_csv = payload.get("ordering_csv")
    if not ordering_csv:
        raise BaselineFailure(
            code="slingshot_r_missing_ordering",
            message="slingshot_r runner did not report an ordering_csv path.",
            hint="See runner log; this is an internal runner contract failure.",
            retryable=False,
            method=_METHOD,
            stderr=proc.stderr or None,
        )

    try:
        cell_ids, pst = _parse_ordering_csv(Path(ordering_csv))
        log_lines.append(f"[{_now()}] parsed ordering {len(cell_ids)} cells")

        _write_ordering_parquet(ordering_path, cell_ids, pst)
        log_lines.append(f"[{_now()}] wrote {ordering_path.name}")

        _save_plot(pst, plot_path)
        log_lines.append(f"[{_now()}] wrote {plot_path.name}")

        edges: list[tuple[str, str]] | None = None
        edges_csv = payload.get("edges_csv")
        if edges_csv:
            edges = _parse_edges_csv(Path(edges_csv))
            log_lines.append(f"[{_now()}] parsed {len(edges)} edges")

        _write_delta_zarr(delta_path, pst)
        log_lines.append(f"[{_now()}] wrote {delta_path.name} (v1 minimal)")

        log_lines.append(f"[{_now()}] slingshot_r.run_slingshot_r done")
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

        return BaselineResult(
            method=_METHOD,
            ordering_path=str(ordering_path),
            lineage_edges=edges,
            root_cell=payload.get("root"),
            figure_path=str(plot_path),
            delta_zarr_path=str(delta_path),
            adata_delta_keys=["obs/slingshot_r_pseudotime"],
        )
    except BaselineFailure:
        raise
    except Exception as exc:  # noqa: BLE001
        log_lines.append(
            f"[{_now()}] FAIL post-runner {type(exc).__name__}: {exc}"
        )
        try:
            log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        except OSError:
            pass
        raise BaselineFailure(
            code="slingshot_r_persist_error",
            message=(
                "Failed to persist slingshot_r artifacts: "
                f"{type(exc).__name__}: {exc}"
            ),
            hint="Check write permissions on the run directory.",
            retryable=False,
            method=_METHOD,
            stderr=proc.stderr or None,
        ) from exc


__all__ = ["run_slingshot_r"]
