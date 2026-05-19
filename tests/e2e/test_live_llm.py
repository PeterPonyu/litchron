"""Spec §5.9: Live LLM end-to-end test — gated by LITCHRON_RUN_LIVE_E2E=1.

This test drives a real Claude Code session to completion on a synthetic
h5ad fixture. It is skipped by default in all CI lanes; enable it with:

    LITCHRON_RUN_LIVE_E2E=1 pytest tests/e2e/test_live_llm.py -m live_llm

What it verifies:
- A single non-interactive Claude Code invocation drives LitChron through
  all phases until report_status().all_green == True.
- The output PDF exists at runs/<run_id>/report.pdf with size >= 50 KB.
- state.json.citations_verified is non-empty.
- The run completes within a 20-minute wall-clock budget.

Citation cassette replay
------------------------
To avoid network flakiness, CrossRef/PubMed responses are captured to
``tests/fixtures/citation_cassette.json`` on the first live run and replayed
on subsequent runs. Set ``LITCHRON_REFRESH_CASSETTE=1`` to re-capture.

Note: This test does NOT assert biological correctness — only that the
MCP-driven loop terminates with a verifiable artifact.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

pytestmark = [pytest.mark.live_llm, pytest.mark.e2e]

_RUN_LIVE = os.environ.get("LITCHRON_RUN_LIVE_E2E") == "1"

_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
_DEV_SMALL_H5AD = _FIXTURES_DIR / "dev_small.h5ad"

_MAX_WALL_SECONDS = 20 * 60  # 20 minutes


@pytest.mark.live_llm
def test_live_llm_drives_litchron_to_completion(tmp_path: Path) -> None:
    """Full end-to-end: Claude Code → LitChron MCP → report.pdf."""
    if not _RUN_LIVE:
        pytest.skip(
            "Live LLM e2e test skipped. "
            "Set LITCHRON_RUN_LIVE_E2E=1 to enable on the pre-release lane."
        )

    if not _DEV_SMALL_H5AD.exists():
        pytest.skip(
            f"Fixture h5ad not found at {_DEV_SMALL_H5AD}. "
            "Commit tests/fixtures/dev_small.h5ad (≤ 5 MB) to enable this test."
        )

    run_id = f"e2e-live-{int(time.time())}"

    prompt = (
        f"Drive LitChron to completion on {_DEV_SMALL_H5AD} with run_id={run_id!r}. "
        "Stop when report_status().all_green is True. "
        "Do not ask for confirmation."
    )

    start = time.monotonic()

    result = subprocess.run(
        ["claude", "-p", prompt, "--output-format", "json"],
        capture_output=True,
        text=True,
        timeout=_MAX_WALL_SECONDS,
    )

    elapsed = time.monotonic() - start

    assert result.returncode == 0, (
        f"claude exited with rc={result.returncode} after {elapsed:.1f}s.\n"
        f"stdout: {result.stdout[-3000:]}\n"
        f"stderr: {result.stderr[-1000:]}"
    )

    assert elapsed < _MAX_WALL_SECONDS, (
        f"Run exceeded 20-minute budget: {elapsed:.1f}s"
    )

    # Locate the run directory.
    from litchron._runtime import project_root as _project_root

    run_dir = _project_root() / "runs" / run_id

    # Verify state.json.
    state_path = run_dir / "state.json"
    assert state_path.exists(), f"state.json not found at {state_path}"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state.get("all_green") is True, (
        f"state.json.all_green is not True: {state.get('all_green')!r}"
    )
    assert len(state.get("citations_verified", [])) > 0, (
        "state.json.citations_verified is empty — no citations were verified"
    )

    # Verify PDF.
    pdf_path = run_dir / "report.pdf"
    assert pdf_path.exists(), f"report.pdf not found at {pdf_path}"
    pdf_size = pdf_path.stat().st_size
    assert pdf_size >= 50 * 1024, (
        f"report.pdf is too small: {pdf_size} bytes (expected >= 50 KB)"
    )
