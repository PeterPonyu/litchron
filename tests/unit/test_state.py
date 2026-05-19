"""Tests for RunState, RunStateStore, default_state, and fcntl lock semantics."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from litchron.state import RunState, RunStateStore, default_state


def _make_state(run_id: str = "test-run-001") -> RunState:
    s = default_state(run_id=run_id, h5ad_path="/tmp/fake.h5ad")
    s.started_at = "2026-01-01T00:00:00+00:00"
    return s


def test_schema_version_is_1() -> None:
    s = _make_state()
    assert s.schema_version == 1


def test_round_trip(tmp_path: Path) -> None:
    """Write → read → equal."""
    store = RunStateStore(tmp_path / "run-001")
    original = _make_state("round-trip")
    store.write(original)
    recovered = store.read()
    assert recovered == original


def test_update_modifies_state(tmp_path: Path) -> None:
    """update(fn) applies the function and persists the result."""
    store = RunStateStore(tmp_path / "run-002")
    initial = _make_state("update-test")
    store.write(initial)

    updated = store.update(lambda s: s.model_copy(update={"phase": "baselines"}))
    assert updated.phase == "baselines"

    on_disk = store.read()
    assert on_disk.phase == "baselines"


def test_atomicity_incomplete_write_leaves_good_state(tmp_path: Path) -> None:
    """A tmp file that is never renamed must not corrupt the good state."""
    store = RunStateStore(tmp_path / "run-003")
    good = _make_state("atomic-test")
    store.write(good)

    # Simulate a write that fails after creating the tmp file but before
    # os.replace (e.g. disk-full). We write a corrupt tmp file manually and
    # then read — the store must still return the good state.
    import tempfile

    fd, tmp_name = tempfile.mkstemp(
        prefix=".state-", suffix=".json.tmp", dir=str(store.run_dir)
    )
    import os

    with os.fdopen(fd, "w") as fh:
        fh.write("<<<CORRUPT>>>")
    # Do NOT call os.replace — simulates interrupted write.
    # The state.json from the previous good write must still be readable.
    recovered = store.read()
    assert recovered == good

    # Cleanup tmp file.
    try:
        os.unlink(tmp_name)
    except FileNotFoundError:
        pass


def test_read_raises_if_no_state_file(tmp_path: Path) -> None:
    store = RunStateStore(tmp_path / "no-state-yet")
    with pytest.raises(FileNotFoundError):
        store.read()


def test_update_creates_default_when_no_file(tmp_path: Path) -> None:
    """update() seeds a default RunState when no state.json exists yet."""
    store = RunStateStore(tmp_path / "run-004")
    result = store.update(lambda s: s.model_copy(update={"phase": "started"}))
    assert result.phase == "started"
    # Should be readable afterwards.
    on_disk = store.read()
    assert on_disk.phase == "started"


def test_state_serializes_to_valid_json(tmp_path: Path) -> None:
    store = RunStateStore(tmp_path / "run-005")
    state = _make_state("json-check")
    store.write(state)
    raw = (store.run_dir / "state.json").read_text(encoding="utf-8")
    parsed = json.loads(raw)
    assert parsed["schema_version"] == 1
    assert parsed["run_id"] == "json-check"
