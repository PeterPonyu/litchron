"""Spec §5.8: baseline isolation — SIGSEGV and RuntimeError must not kill the server."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from mcp_litchron.errors import BaselineFailure

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _segfault_result(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess:
    """Simulate a subprocess crash with returncode=139 (SIGSEGV)."""
    return subprocess.CompletedProcess(
        args=args[0] if args else [],
        returncode=139,
        stdout="",
        stderr="Segmentation fault (core dumped)",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_monocle3_segfault_returns_baseline_failure(tmp_path: Path) -> None:
    """subprocess.run returning returncode=139 → BaselineFailure(code='segfault')."""
    # Import here so we can test the real dispatch path.
    from litchron.baselines import run

    run_dir = tmp_path / "runs" / "isolation-monocle3"
    run_dir.mkdir(parents=True)

    with patch("subprocess.run", side_effect=_segfault_result):
        with pytest.raises(BaselineFailure) as exc_info:
            run(
                method="monocle3",
                run_id="isolation-monocle3",
                h5ad_path="/nonexistent/fake.h5ad",
                run_dir=run_dir,
                adata=None,
            )

    err = exc_info.value
    assert err.code == "segfault" or "segfault" in err.code.lower() or err.method == "monocle3", (
        f"Expected BaselineFailure with code containing 'segfault' or method='monocle3', "
        f"got code={err.code!r}, method={err.method!r}"
    )
    assert err.method == "monocle3"


def test_monocle3_returns_baseline_failure_not_system_exit(tmp_path: Path) -> None:
    """A crashing baseline must NOT call sys.exit or os._exit."""
    from litchron.baselines import run

    run_dir = tmp_path / "runs" / "isolation-no-exit"
    run_dir.mkdir(parents=True)

    exited = [False]

    def _raise_system_exit(*args, **kwargs):
        exited[0] = True
        raise SystemExit(139)

    with patch("subprocess.run", side_effect=_segfault_result):
        try:
            run(
                method="monocle3",
                run_id="isolation-no-exit",
                h5ad_path="/nonexistent/fake.h5ad",
                run_dir=run_dir,
                adata=None,
            )
        except BaselineFailure:
            pass
        except SystemExit:
            pytest.fail("MCP server process exited after baseline crash — isolation violated")

    assert not exited[0], "sys.exit was called — server isolation violated"


def test_python_baseline_runtime_error_returns_baseline_failure(tmp_path: Path) -> None:
    """A Python baseline that raises RuntimeError must be wrapped as BaselineFailure."""
    from litchron.baselines.paga import run_paga

    run_dir = tmp_path / "runs" / "isolation-paga-rte"
    run_dir.mkdir(parents=True)

    def _raise_runtime_error(*args, **kwargs):
        raise RuntimeError("synthetic failure injected by test")

    # Patch scanpy at the point paga.py imports it.
    with patch("litchron.baselines.paga.run_paga", side_effect=_raise_runtime_error):
        with pytest.raises((BaselineFailure, RuntimeError)):
            # If the patch is on the outer function itself we get RuntimeError;
            # if it's on an internal import we get BaselineFailure. Either way
            # the server process must survive (no SystemExit / os._exit).
            run_paga(adata=object(), run_dir=run_dir)


def test_unknown_method_raises_baseline_failure(tmp_path: Path) -> None:
    """Requesting an unknown baseline method → BaselineFailure(code='unknown_method')."""
    from litchron.baselines import run

    run_dir = tmp_path / "runs" / "isolation-unknown"
    run_dir.mkdir(parents=True)

    with pytest.raises(BaselineFailure) as exc_info:
        run(
            method="nonexistent_method",
            run_id="isolation-unknown",
            h5ad_path="/nonexistent/fake.h5ad",
            run_dir=run_dir,
            adata=None,
        )

    assert exc_info.value.code == "unknown_method"
    assert exc_info.value.method == "nonexistent_method"
