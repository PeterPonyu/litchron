"""Spec §5.0: preflight.check_environment and assert_critical_or_raise."""
from __future__ import annotations

import shutil
from unittest.mock import patch

import pytest

from litchron.preflight import PreflightReport, assert_critical_or_raise, check_environment


def test_check_environment_returns_preflight_report() -> None:
    report = check_environment(require_r=False)
    assert isinstance(report, PreflightReport)


def test_report_shape() -> None:
    report = check_environment(require_r=False)
    # Shape: all documented fields exist.
    assert hasattr(report, "pandoc")
    assert hasattr(report, "latexmk")
    assert hasattr(report, "rscript")
    assert hasattr(report, "r_version")
    assert hasattr(report, "mcp_importable")
    assert hasattr(report, "scanpy_importable")
    assert hasattr(report, "embed_model_available")
    assert hasattr(report, "missing")
    assert hasattr(report, "warnings")
    assert hasattr(report, "all_critical_ok")


def _make_patched_lookup(absent: str):
    """Return a fn that mimics _which_with_env_bin but returns None for ``absent``."""
    from litchron.preflight import _which_with_env_bin as real_lookup

    def patched_lookup(name: str):
        if name == absent:
            return None
        return real_lookup(name)

    return patched_lookup


def test_pandoc_none_appears_in_missing_when_absent(monkeypatch) -> None:
    """When the env-aware pandoc lookup returns None, 'pandoc' appears in missing."""
    with patch(
        "litchron.preflight._which_with_env_bin",
        side_effect=_make_patched_lookup("pandoc"),
    ):
        report = check_environment(require_r=False)

    assert report.pandoc is None
    assert "pandoc" in report.missing


def test_assert_critical_raises_with_apt_hint_when_pandoc_none(monkeypatch) -> None:
    """assert_critical_or_raise must raise RuntimeError with apt-install hint."""
    with patch(
        "litchron.preflight._which_with_env_bin",
        side_effect=_make_patched_lookup("pandoc"),
    ):
        report = check_environment(require_r=False)

    # pandoc absent → all_critical_ok must be False.
    assert report.all_critical_ok is False

    with pytest.raises(RuntimeError) as exc_info:
        assert_critical_or_raise(report)

    msg = str(exc_info.value)
    assert "apt install" in msg or "sudo apt" in msg, (
        f"Expected apt-install hint in error message, got:\n{msg}"
    )
    assert "pandoc" in msg


def test_assert_critical_does_not_raise_when_ok() -> None:
    """No exception when all_critical_ok is True."""
    report = PreflightReport(
        pandoc="/usr/bin/pandoc",
        latexmk="/usr/bin/latexmk",
        mcp_importable=True,
        all_critical_ok=True,
    )
    # Must not raise.
    assert_critical_or_raise(report)


def test_require_r_adds_rscript_to_missing_when_absent(monkeypatch) -> None:
    """require_r=True and Rscript absent → 'Rscript' in missing."""
    with patch(
        "litchron.preflight._which_with_env_bin",
        side_effect=_make_patched_lookup("Rscript"),
    ):
        report = check_environment(require_r=True)

    assert "Rscript" in report.missing
