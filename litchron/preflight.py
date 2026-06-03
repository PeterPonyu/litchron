"""Preflight environment checks for LitChron.

Runs on MCP server startup (and on demand) to confirm the host has the
binaries and Python packages required to produce a citation-verified,
LaTeX-compiled report. Critical missing pieces must raise early so the
LLM doesn't begin a multi-hour run that can't finish.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


def _which_with_env_bin(name: str) -> Optional[str]:
    """Like :func:`shutil.which` but also probes the directory of ``sys.executable``.

    MCP clients launch the server with the env's interpreter directly (e.g.
    ``/path/to/env/bin/python -m mcp_litchron.server``); in that mode the
    inherited ``$PATH`` may not include the env's ``bin/``, so a conda-installed
    pandoc won't be visible to bare :func:`shutil.which`. This helper covers
    that case by checking the interpreter's sibling directory first.
    """
    hit = shutil.which(name)
    if hit is not None:
        return hit
    candidate = Path(sys.executable).parent / name
    if candidate.is_file() and candidate.stat().st_mode & 0o111:
        return str(candidate)
    return None

from pydantic import BaseModel, Field

from .config import EMBED_MODEL


class PreflightReport(BaseModel):
    """Structured snapshot of the host's LitChron readiness."""

    pandoc: Optional[str] = None
    latexmk: Optional[str] = None
    biber: Optional[str] = None
    rscript: Optional[str] = None
    r_version: Optional[str] = None

    mcp_importable: bool = False
    scanpy_importable: bool = False
    embed_model_available: bool = False

    missing: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    all_critical_ok: bool = False


def _hf_cache_has_embed_model() -> bool:
    """Return True iff the configured embed model is in the local HF cache.

    We only check that the model directory exists — not that every file
    is downloaded. A more thorough check would inspect the snapshots/
    subtree, but that's brittle when HF caching changes shape. For
    preflight purposes "directory present" is a good enough heuristic;
    the verifier will load lazily and surface concrete errors if files
    are missing.
    """
    # EMBED_MODEL has the form "<org>/<name>"; HF cache uses
    # "models--<org>--<name>".
    safe = EMBED_MODEL.replace("/", "--")
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{safe}"
    return cache_dir.exists()


def _r_version_from_rscript(rscript_path: str) -> Optional[str]:
    """Invoke Rscript to print "<major>.<minor>"; return None on failure."""
    try:
        result = subprocess.run(
            [rscript_path, "-e", "cat(R.version$major, R.version$minor, sep='.')"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    out = result.stdout.strip()
    return out or None


def check_environment(require_r: bool = False) -> PreflightReport:
    """Inspect host tools and Python imports; return a :class:`PreflightReport`."""
    report = PreflightReport()

    # --- binaries ---------------------------------------------------------
    report.pandoc = _which_with_env_bin("pandoc")
    report.latexmk = _which_with_env_bin("latexmk")
    report.biber = _which_with_env_bin("biber")
    report.rscript = _which_with_env_bin("Rscript")

    if report.rscript is not None:
        report.r_version = _r_version_from_rscript(report.rscript)

    # --- Python imports ---------------------------------------------------
    try:
        import mcp  # noqa: F401

        report.mcp_importable = True
    except ImportError:
        report.mcp_importable = False

    try:
        import scanpy  # noqa: F401

        report.scanpy_importable = True
    except ImportError:
        report.scanpy_importable = False

    # --- embed model cache ------------------------------------------------
    report.embed_model_available = _hf_cache_has_embed_model()

    # --- aggregate missing / warnings ------------------------------------
    if report.pandoc is None:
        report.missing.append("pandoc")
    if report.latexmk is None:
        report.missing.append("latexmk")
    if report.biber is None:
        report.warnings.append(
            "biber not found; the tex template uses backend=biber, so the "
            "bibliography (references) will not compile — install "
            "texlive-bibtex-extra and biber"
        )
    if not report.mcp_importable:
        report.missing.append("mcp (python package)")

    if not report.scanpy_importable:
        report.warnings.append("scanpy not importable; baselines will fail")
    if not report.embed_model_available:
        report.warnings.append(
            f"embed model {EMBED_MODEL!r} not found in HF cache; first use will download"
        )
    if require_r:
        if report.rscript is None:
            report.missing.append("Rscript")
        elif report.r_version is None:
            report.warnings.append(
                "Rscript present but version probe failed; R baselines may not run"
            )

    # --- critical gate ----------------------------------------------------
    report.all_critical_ok = (
        report.pandoc is not None
        and report.latexmk is not None
        and report.mcp_importable
    )

    return report


def assert_critical_or_raise(report: PreflightReport) -> None:
    """Raise :class:`RuntimeError` with install hints if critical pieces missing."""
    if report.all_critical_ok:
        return

    lines: list[str] = [
        "LitChron preflight failed — missing critical components.",
        f"Missing: {', '.join(report.missing) or '(none listed)'}",
        "",
        "To install system binaries on Debian/Ubuntu:",
        "    sudo apt install pandoc latexmk texlive-latex-extra texlive-bibtex-extra biber",
        "",
        "To install the MCP Python package:",
        "    pip install 'mcp>=1.0'",
    ]
    if report.warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in report.warnings:
            lines.append(f"  - {w}")
    raise RuntimeError("\n".join(lines))
