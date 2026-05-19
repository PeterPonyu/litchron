"""Spec §5.12 / L54: assert the repository skeleton is complete."""
from __future__ import annotations

from pathlib import Path


def test_required_files_exist(project_root: Path) -> None:
    required = [
        "pyproject.toml",
        "environment.yml",
        "README.md",
        "litchron/__init__.py",
        "mcp_litchron/__init__.py",
        "tex/litchron.tex",
        "litchron/preflight.py",
        "litchron/state.py",
        "litchron/citations.py",
        "mcp_litchron/cache.py",
        "mcp_litchron/tools.py",
        "mcp_litchron/server.py",
    ]
    missing = [p for p in required if not (project_root / p).exists()]
    assert missing == [], f"Missing files: {missing}"
