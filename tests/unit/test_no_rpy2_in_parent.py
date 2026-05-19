"""Spec §5.1.1: Hybrid-isolation invariant.

Importing the parent modules must never drag rpy2 into sys.modules.
The R-backed baselines spawn subprocesses; only _r_runner (run in a child
process) is allowed to import rpy2.
"""
from __future__ import annotations

import sys


def test_rpy2_not_in_sys_modules_after_litchron_import() -> None:
    import litchron  # noqa: F401

    assert "rpy2" not in sys.modules, (
        "rpy2 was imported as a side-effect of 'import litchron'. "
        "Check litchron/__init__.py and its transitive imports."
    )


def test_rpy2_not_in_sys_modules_after_litchron_baselines_import() -> None:
    import litchron.baselines  # noqa: F401

    assert "rpy2" not in sys.modules, (
        "rpy2 was imported as a side-effect of 'import litchron.baselines'. "
        "Only _r_runner (run as a subprocess) may import rpy2."
    )


def test_rpy2_not_in_sys_modules_after_mcp_litchron_import() -> None:
    import mcp_litchron  # noqa: F401

    assert "rpy2" not in sys.modules, (
        "rpy2 was imported as a side-effect of 'import mcp_litchron'."
    )


def test_rpy2_not_in_sys_modules_after_mcp_litchron_cache_import() -> None:
    # mcp_litchron.cache pulls in anndata + litchron.io; neither should touch rpy2.
    import mcp_litchron.cache  # noqa: F401

    assert "rpy2" not in sys.modules, (
        "rpy2 was imported as a side-effect of 'import mcp_litchron.cache'."
    )
