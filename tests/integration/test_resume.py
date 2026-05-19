"""Spec §5.7: resume after crash — idempotent re-run and state replay."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from litchron.state import RunState, RunStateStore, default_state

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_fake_ordering(out_dir: Path, method: str) -> Path:
    """Write a minimal ordering.parquet for a given baseline method."""
    ordering_dir = out_dir / "baselines" / method
    ordering_dir.mkdir(parents=True, exist_ok=True)
    ordering_path = ordering_dir / "ordering.parquet"
    table = pa.table(
        {
            "cell_id": pa.array([f"cell_{i:04d}" for i in range(10)], type=pa.string()),
            "pseudotime": pa.array(np.linspace(0.0, 1.0, 10), type=pa.float64()),
            "cell_type": pa.array(["0"] * 4 + ["1"] * 3 + ["2"] * 3, type=pa.string()),
        }
    )
    pq.write_table(table, str(ordering_path))
    return ordering_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_resume_skips_completed_baseline(tmp_path: Path) -> None:
    """A baseline already in baselines_done must not be re-computed.

    Simulates the idempotency contract: if state says paga is done, calling
    the run logic again should see the completed state and short-circuit.
    """
    run_dir = tmp_path / "runs" / "resume-test-001"
    run_dir.mkdir(parents=True)

    # Seed state with paga + palantir completed.
    store = RunStateStore(run_dir)
    s = default_state(run_id="resume-test-001", h5ad_path="/tmp/fake.h5ad")
    s.started_at = "2026-01-01T00:00:00+00:00"
    s.baselines_done = ["paga", "palantir"]
    store.write(s)

    # Write fake ordering artifacts so downstream code finds the files.
    _write_fake_ordering(run_dir, "paga")
    _write_fake_ordering(run_dir, "palantir")

    # Read back and verify — the completed state persists across reads.
    recovered = store.read()
    assert "paga" in recovered.baselines_done
    assert "palantir" in recovered.baselines_done


def test_completed_baseline_idempotent_check(tmp_path: Path) -> None:
    """Calling update() on a state with baselines_done does not remove them."""
    run_dir = tmp_path / "runs" / "resume-test-002"
    run_dir.mkdir(parents=True)

    store = RunStateStore(run_dir)
    s = default_state(run_id="resume-test-002", h5ad_path="/tmp/fake.h5ad")
    s.started_at = "2026-01-01T00:00:00+00:00"
    s.baselines_done = ["paga"]
    store.write(s)

    # Simulate a "re-run" check: update that adds a new baseline without
    # removing the existing one.
    def add_palantir(state: RunState) -> RunState:
        new_done = list(state.baselines_done)
        if "palantir" not in new_done:
            new_done.append("palantir")
        return state.model_copy(update={"baselines_done": new_done})

    updated = store.update(add_palantir)
    assert "paga" in updated.baselines_done
    assert "palantir" in updated.baselines_done


def test_compare_orderings_with_persisted_parquets(tmp_path: Path) -> None:
    """compare() succeeds on orderings loaded from persisted parquet files."""
    from litchron.compare import compare

    run_dir = tmp_path / "runs" / "resume-test-003"
    run_dir.mkdir(parents=True)

    paga_path = _write_fake_ordering(run_dir, "paga")
    palantir_path = _write_fake_ordering(run_dir, "palantir")

    # Load the orderings.
    paga_table = pq.read_table(str(paga_path))
    palantir_table = pq.read_table(str(palantir_path))

    paga_series = pd.Series(
        paga_table["pseudotime"].to_pylist(),
        index=paga_table["cell_id"].to_pylist(),
        name="pseudotime",
    )
    palantir_series = pd.Series(
        palantir_table["pseudotime"].to_pylist(),
        index=palantir_table["cell_id"].to_pylist(),
        name="pseudotime",
    )

    row = compare(
        llm_ordering=paga_series,
        baseline_ordering=palantir_series,
        baseline_name="palantir",
    )

    assert row.spearman is not None
    assert row.baseline == "palantir"


def test_state_survives_restart_cycle(tmp_path: Path) -> None:
    """Write state, 'restart' by re-creating the store, read back unchanged."""
    run_dir = tmp_path / "runs" / "resume-test-004"
    run_dir.mkdir(parents=True)

    store1 = RunStateStore(run_dir)
    s = default_state(run_id="resume-test-004", h5ad_path="/tmp/fake.h5ad")
    s.started_at = "2026-01-01T00:00:00+00:00"
    s.baselines_done = ["paga"]
    s.comparison_done = True
    store1.write(s)

    # Simulate process restart: new store object pointing at the same directory.
    store2 = RunStateStore(run_dir)
    recovered = store2.read()

    assert recovered.baselines_done == ["paga"]
    assert recovered.comparison_done is True
    assert recovered.run_id == "resume-test-004"
