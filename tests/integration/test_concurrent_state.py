"""Spec §5.7b: concurrent-state-write race — fcntl.flock correctness."""
from __future__ import annotations

import multiprocessing
from pathlib import Path

import pytest

from litchron.state import RunState, RunStateStore, default_state

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Worker function (must be module-level for multiprocessing pickling)
# ---------------------------------------------------------------------------

def _increment_counter(run_dir_str: str) -> None:
    """Increment the counter field in state.json by 1 under a lock."""
    run_dir = Path(run_dir_str)
    store = RunStateStore(run_dir)

    def bump(state: RunState) -> RunState:
        current = getattr(state, "phase", "0")
        # We abuse the phase field as a string counter for this test.
        try:
            n = int(current)
        except (ValueError, TypeError):
            n = 0
        return state.model_copy(update={"phase": str(n + 1)})

    store.update(bump)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_concurrent_updates_both_land(tmp_path: Path) -> None:
    """Two concurrent update(fn) calls must both increment the counter.

    Uses multiprocessing.Pool so each worker runs in a separate OS process
    with an independent file descriptor — this exercises the fcntl.flock
    guard across real process boundaries.
    """
    run_dir = tmp_path / "runs" / "concurrent-state-001"
    run_dir.mkdir(parents=True)

    # Seed state with counter = 0.
    store = RunStateStore(run_dir)
    s = default_state(run_id="concurrent-state-001", h5ad_path="/tmp/fake.h5ad")
    s.started_at = "2026-01-01T00:00:00+00:00"
    s.phase = "0"
    store.write(s)

    run_dir_str = str(run_dir)

    # Spawn two processes that each increment the counter.
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(2) as pool:
        pool.map(_increment_counter, [run_dir_str, run_dir_str])

    # Both increments must have landed.
    final = store.read()
    try:
        counter = int(final.phase)
    except (ValueError, TypeError):
        counter = -1

    assert counter == 2, (
        f"Expected counter == 2 after two concurrent increments, got {counter!r}. "
        "This indicates a torn write or missed fcntl.flock acquisition."
    )


def test_state_file_always_valid_json_under_concurrent_writes(tmp_path: Path) -> None:
    """state.json must always be valid JSON after concurrent writes."""
    import json

    run_dir = tmp_path / "runs" / "concurrent-state-002"
    run_dir.mkdir(parents=True)

    store = RunStateStore(run_dir)
    s = default_state(run_id="concurrent-state-002", h5ad_path="/tmp/fake.h5ad")
    s.started_at = "2026-01-01T00:00:00+00:00"
    s.phase = "0"
    store.write(s)

    run_dir_str = str(run_dir)
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(2) as pool:
        pool.map(_increment_counter, [run_dir_str, run_dir_str])

    # The file must parse as valid JSON.
    raw = (run_dir / "state.json").read_text(encoding="utf-8")
    parsed = json.loads(raw)  # must not raise
    assert isinstance(parsed, dict)
    assert "run_id" in parsed
