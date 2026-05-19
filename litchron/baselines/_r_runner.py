"""Subprocess entry point for R-bridged trajectory baselines.

Usage::

    python -m litchron.baselines._r_runner <method> <run_id> <h5ad_path> <run_dir>

Where ``method`` is one of ``monocle3``, ``slingshot_r``. This is the
**only** place in LitChron permitted to import :mod:`rpy2`. Keeping the
R interpreter in a child process means a SIGSEGV from libR (which is
common with Monocle3 / Bioconductor stacks) cannot take down the MCP
server itself.

Output contract: a **single** JSON line on stdout, either::

    {"status": "ok", "ordering_csv": "<abs path>", "edges_csv": "<abs path|null>", "root": "<cell id|null>"}

or::

    {"status": "error", "code": "<short id>", "message": "<human>"}

Exit codes:

* ``0`` -- success (status: ok).
* ``1`` -- caught error (status: error).
* ``-11`` / ``139`` -- SIGSEGV from libR. The parent process is expected
  to translate that into a structured :class:`BaselineFailure`.

V1 stub mode: if ``LITCHRON_STUB_R=1`` is set in the environment, the
runner short-circuits and emits a deterministic linear pseudotime over
all cells in the supplied h5ad without ever importing rpy2. This lets
the test suite exercise the subprocess plumbing without a working R
installation.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any


def _emit_ok(
    ordering_csv: str,
    edges_csv: str | None = None,
    root: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "status": "ok",
        "ordering_csv": ordering_csv,
        "edges_csv": edges_csv,
        "root": root,
    }
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _emit_error(code: str, message: str) -> None:
    payload = {"status": "error", "code": code, "message": message}
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _stub_run(method: str, h5ad_path: str, run_dir: Path) -> int:
    """Deterministic stub that bypasses rpy2 for CI / tests."""
    try:
        # Local import — anndata is a LitChron dep, but keep the import
        # inside the stub so a missing optional doesn't break the entry.
        from anndata import read_h5ad

        adata = read_h5ad(h5ad_path)
    except Exception as exc:  # noqa: BLE001
        _emit_error("stub_h5ad_load_failed", f"{type(exc).__name__}: {exc}")
        return 1

    out_dir = run_dir / "baselines" / method
    out_dir.mkdir(parents=True, exist_ok=True)
    ordering_csv = out_dir / "ordering.csv"

    n = int(adata.n_obs)
    cell_ids = [str(x) for x in adata.obs_names]
    with ordering_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["cell_id", "pseudotime"])
        for i, cid in enumerate(cell_ids):
            pst = float(i) / max(n - 1, 1)
            writer.writerow([cid, f"{pst:.6f}"])

    root = cell_ids[0] if cell_ids else None
    _emit_ok(ordering_csv=str(ordering_csv), edges_csv=None, root=root)
    return 0


def _real_run(method: str, h5ad_path: str, run_dir: Path) -> int:
    """Real path: import rpy2 and dispatch to the requested method.

    NOTE: rpy2 import is intentionally late so the stub path never touches
    libR. Any exception here is translated into a JSON error line.
    """
    try:
        import rpy2.robjects as ro  # noqa: F401  (late import by design)
    except Exception as exc:  # noqa: BLE001
        _emit_error("rpy2_unavailable", f"{type(exc).__name__}: {exc}")
        return 1

    out_dir = run_dir / "baselines" / method
    out_dir.mkdir(parents=True, exist_ok=True)

    if method == "monocle3":
        # Real Monocle3 dispatch — left as a thin placeholder for the v1
        # ship. The Python wrapper persists artifacts and surfaces the
        # error message verbatim if this branch returns the not-implemented
        # error code.
        _emit_error(
            "monocle3_not_implemented",
            (
                "rpy2 monocle3 bridge not implemented in v1; set "
                "LITCHRON_STUB_R=1 to use the deterministic stub."
            ),
        )
        return 1
    if method == "slingshot_r":
        _emit_error(
            "slingshot_r_not_implemented",
            (
                "rpy2 slingshot bridge not implemented in v1; set "
                "LITCHRON_STUB_R=1 to use the deterministic stub."
            ),
        )
        return 1

    _emit_error("unknown_method", f"unknown method: {method!r}")
    return 1


def main(argv: list[str]) -> int:
    if len(argv) != 5:
        _emit_error(
            "bad_argv",
            (
                "usage: python -m litchron.baselines._r_runner "
                "<method> <run_id> <h5ad_path> <run_dir>"
            ),
        )
        return 1

    _, method, _run_id, h5ad_path, run_dir_str = argv
    run_dir = Path(run_dir_str)
    if method not in {"monocle3", "slingshot_r"}:
        _emit_error("unknown_method", f"unknown method: {method!r}")
        return 1

    if os.environ.get("LITCHRON_STUB_R") == "1":
        return _stub_run(method, h5ad_path, run_dir)
    return _real_run(method, h5ad_path, run_dir)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
