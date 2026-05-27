"""Tests for ``mcp_litchron.tools.start_run`` force-restart archive behavior.

When ``force=True`` is passed and the target run directory contains
artifacts from a prior run (proposals.md, figures/, baselines/, etc.),
``start_run`` must sweep those artifacts into
``<run_dir>/_archive/<timestamp>/`` BEFORE rewriting the skeleton +
``state.json``. Otherwise stale content (in particular ``proposals.md``)
silently pollutes the new report.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from litchron import _runtime as runtime_mod
from mcp_litchron import tools as tools_mod
from mcp_litchron.tools import StartRunResult, start_run


@pytest.fixture
def isolated_project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ``litchron._runtime.project_root`` (and the import-time
    binding in :mod:`mcp_litchron.tools`) to ``tmp_path`` so each test gets
    a fresh ``runs/`` directory.
    """
    fake_root = tmp_path / "repo"
    fake_root.mkdir()
    monkeypatch.setattr(runtime_mod, "project_root", lambda: fake_root)
    # tools.py imports run_dir from _runtime — that re-uses project_root
    # via the patched symbol, so no further patching is required for the
    # tools-side path. But start_run also imports `project_root` directly
    # from _runtime; the lambda above is the canonical override.
    monkeypatch.setattr(tools_mod, "run_dir", runtime_mod.run_dir)
    return fake_root


def _make_h5ad(tmp_path: Path) -> Path:
    """Create a minimal AnnData h5ad file on disk and return its path.

    ``start_run`` only requires the path to be a regular file; it does
    not load the contents (that's ``load_h5ad``'s job).
    """
    path = tmp_path / "data.h5ad"
    path.write_bytes(b"\x89HDF\r\n\x1a\n")  # not a real h5ad — start_run
    # only checks ``src.is_file()``.
    return path


def test_force_archives_stale_proposal_and_figures(
    isolated_project_root: Path, tmp_path: Path
) -> None:
    """``force=True`` on a non-empty run dir archives prior artifacts."""
    h5ad = _make_h5ad(tmp_path)
    rid = "force-restart-test"

    # First call seeds the run dir.
    first = start_run(h5ad_path=str(h5ad), run_id=rid, force=False)
    assert isinstance(first, StartRunResult)
    run_dir = Path(first.run_dir)

    # Seed stale artifacts that would pollute a new report.
    stale_proposal = run_dir / "proposals.md"
    stale_proposal.write_text(
        "STALE_PROPOSAL_CONTENT: should not survive a force restart"
    )
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    stale_figure = figures_dir / "litchron_annotation.png"
    stale_figure.write_bytes(b"\x89PNG\r\n\x1a\nSTALE_FIGURE_BYTES")
    # A stale baseline directory with an ordering parquet.
    baselines_dir = run_dir / "baselines" / "paga"
    baselines_dir.mkdir(parents=True, exist_ok=True)
    (baselines_dir / "ordering.parquet").write_bytes(b"STALE_BASELINE_BYTES")

    # Force restart: must archive (not delete) the stale contents.
    second = start_run(h5ad_path=str(h5ad), run_id=rid, force=True)
    assert isinstance(second, StartRunResult), f"got {second!r}"
    assert second.run_id == rid

    # 1) The stale files MUST NOT remain at their old paths.
    assert not stale_proposal.exists(), (
        "force=True did not clear stale proposals.md from the run dir"
    )
    assert not stale_figure.exists(), (
        "force=True did not clear stale figures/litchron_annotation.png"
    )
    assert not (run_dir / "baselines" / "paga" / "ordering.parquet").exists(), (
        "force=True did not clear stale baseline ordering parquet"
    )

    # 2) But they MUST be findable under <run_dir>/_archive/<stamp>/.
    archive_root = run_dir / "_archive"
    assert archive_root.is_dir(), "force=True did not create _archive/"
    archives = [p for p in archive_root.iterdir() if p.is_dir()]
    assert len(archives) == 1, f"expected exactly one archive, got {archives}"
    archive_dir = archives[0]
    archived_proposal = archive_dir / "proposals.md"
    assert archived_proposal.is_file(), "stale proposals.md was lost, not archived"
    assert "STALE_PROPOSAL_CONTENT" in archived_proposal.read_text()
    assert (archive_dir / "figures" / "litchron_annotation.png").is_file()
    assert (
        archive_dir / "baselines" / "paga" / "ordering.parquet"
    ).is_file()

    # 3) A fresh skeleton must be present in the now-non-archived part of
    # the run dir.
    assert (run_dir / "baselines").is_dir()
    assert (run_dir / "logs").is_dir()
    assert (run_dir / "tex_sections").is_dir()
    # state.json must be fresh (no llm_ordering_done / no citations etc).
    state = json.loads((run_dir / "state.json").read_text())
    assert state["run_id"] == rid
    assert state.get("llm_ordering_done") in (False, None)


def test_force_on_empty_dir_does_not_create_archive(
    isolated_project_root: Path, tmp_path: Path
) -> None:
    """``force=True`` on a never-used run_id should not create an empty archive."""
    h5ad = _make_h5ad(tmp_path)
    rid = "fresh-force-test"
    res = start_run(h5ad_path=str(h5ad), run_id=rid, force=True)
    assert isinstance(res, StartRunResult)
    assert not (Path(res.run_dir) / "_archive").exists(), (
        "force=True on an empty run dir should not create _archive/"
    )


def test_repeated_force_accumulates_separate_archives(
    isolated_project_root: Path, tmp_path: Path
) -> None:
    """Two ``force=True`` calls produce two separate archive subdirs.

    Crucially, the second archive must NOT contain the first archive
    (no nesting), even though the first archive lives under the run dir.
    """
    h5ad = _make_h5ad(tmp_path)
    rid = "double-force-test"
    start_run(h5ad_path=str(h5ad), run_id=rid, force=False)
    run_dir = Path(runtime_mod.run_dir(rid))
    (run_dir / "proposals.md").write_text("first")
    start_run(h5ad_path=str(h5ad), run_id=rid, force=True)
    (run_dir / "proposals.md").write_text("second")
    start_run(h5ad_path=str(h5ad), run_id=rid, force=True)

    archives = sorted((run_dir / "_archive").iterdir())
    assert len(archives) == 2, f"expected 2 archives, got {archives!r}"
    # The second archive must NOT contain a nested _archive directory
    # (i.e. we never recursed _archive into itself).
    for a in archives:
        assert not (a / "_archive").exists(), (
            f"archive {a} accidentally nests the _archive root"
        )
