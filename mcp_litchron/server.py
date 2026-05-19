"""LitChron stdio MCP server entrypoint.

This module is the only LitChron file that the parent process executes
directly. Its responsibilities:

1. Run :func:`litchron.preflight.check_environment` and abort with a
   :class:`PreflightFailure` if any critical dependency (pandoc,
   latexmk, the ``mcp`` Python SDK) is missing. The abort is *eager* —
   we refuse to start the server because the LLM has no way to recover
   mid-loop from a missing system binary.
2. Register every tool function in :mod:`mcp_litchron.tools` against a
   :class:`ToolRegistry`. The registry is the in-process source of
   truth for *what tools exist* and is consumed by:
   - the ``--list-tools`` CLI flag (used by §5.3 verification), and
   - the actual MCP stdio handler (when the SDK is importable).
3. When invoked as a module (``python -m mcp_litchron.server``), select
   between the ``--list-tools`` path (SDK-free) and the stdio handler.

Expected MCP SDK
----------------
This server targets the official ``mcp`` Python SDK (`pip install
mcp>=1.0`), specifically the ``mcp.server.Server`` + ``mcp.server.stdio``
surface. The SDK has not yet stabilized its tool-registration API across
all 1.x releases, so :func:`run_stdio_server` imports inside the function
body and surfaces a :class:`PreflightFailure` if the import fails. This
keeps cold-import (and the ``--list-tools`` path) independent of the SDK.

Tool registration contract
--------------------------
Every entry of :data:`mcp_litchron.tools.TOOL_REGISTRY` is wrapped by
``register_tool(name, fn, description)``. The registry stores enough
information for both surfaces (CLI list and SDK handler) to operate
without re-importing.

R-isolation invariant
---------------------
This module never imports ``rpy2`` or any module that eagerly imports
``rpy2`` (the R-bridged baselines are late-imported inside
:func:`mcp_litchron.tools.run_baseline`). Importing ``server`` must not
load R into the parent process — that's the entire point of Option C
hybrid execution.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any, Callable

from litchron.preflight import PreflightReport, check_environment
from mcp_litchron.errors import PreflightFailure
from mcp_litchron.tools import TOOL_REGISTRY

# ---------------------------------------------------------------------------
# Preflight gate (runs on stdio-server startup — NOT on bare module import)
# ---------------------------------------------------------------------------
#
# The plan requires the server to ``check_environment(require_r=False)`` at
# startup and abort with :class:`PreflightFailure` if critical components
# (pandoc, latexmk, the ``mcp`` SDK) are missing. It also requires
# ``python -m mcp_litchron.server --list-tools`` to succeed for §5.3
# verification — that path inspects the in-process registry and never
# touches the missing system binaries, so it must bypass the gate.
#
# Resolution: preflight runs when :func:`run_stdio_server` is invoked.
# ``--list-tools`` is a CLI inspection of the in-process registry and
# does not require server-grade dependencies. Tests can also import
# this module safely without tripping the gate.
_PREFLIGHT_REPORT: PreflightReport | None = None


def _ensure_preflight() -> PreflightReport:
    """Run preflight (idempotent) and abort with :class:`PreflightFailure`."""
    global _PREFLIGHT_REPORT
    if _PREFLIGHT_REPORT is not None:
        return _PREFLIGHT_REPORT
    report = check_environment(require_r=False)
    if not report.all_critical_ok:
        missing = ", ".join(report.missing) or "(unspecified)"
        raise PreflightFailure(
            code="preflight_critical_missing",
            message=(
                f"LitChron preflight aborted: missing critical components: {missing}."
            ),
            hint=(
                "On Debian/Ubuntu: `sudo apt install pandoc latexmk "
                "texlive-latex-extra`. For the Python SDK: "
                "`pip install 'mcp>=1.0'`."
            ),
            retryable=False,
        )
    _PREFLIGHT_REPORT = report
    return report


# ---------------------------------------------------------------------------
# In-process tool registry
# ---------------------------------------------------------------------------
@dataclass
class RegisteredTool:
    """A single tool entry exposed by both the CLI and the SDK handler."""

    name: str
    description: str
    fn: Callable[..., Any]


class ToolRegistry:
    """Holds the (name → callable) mapping for every MCP tool."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, name: str, fn: Callable[..., Any], description: str) -> None:
        """Add a tool to the registry. Re-registration is rejected."""
        if name in self._tools:
            raise ValueError(f"Tool {name!r} already registered")
        self._tools[name] = RegisteredTool(name=name, description=description, fn=fn)

    def list_tools(self) -> list[dict[str, str]]:
        """Return JSON-serializable ``{name, description}`` pairs."""
        return [
            {"name": t.name, "description": t.description}
            for t in self._tools.values()
        ]

    def get(self, name: str) -> RegisteredTool:
        """Look up a registered tool by name."""
        return self._tools[name]

    def names(self) -> list[str]:
        return list(self._tools.keys())


def build_registry() -> ToolRegistry:
    """Construct the populated :class:`ToolRegistry` from :mod:`tools`."""
    from mcp_litchron import tools as _tools_module

    reg = ToolRegistry()
    for spec in TOOL_REGISTRY:
        fn = getattr(_tools_module, spec.callable_name)
        reg.register(name=spec.name, fn=fn, description=spec.description)
    return reg


# ---------------------------------------------------------------------------
# CLI: --list-tools
# ---------------------------------------------------------------------------
def _print_tools_json(registry: ToolRegistry) -> int:
    """Print the registered tools as a JSON array; return exit code."""
    payload = registry.list_tools()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


# ---------------------------------------------------------------------------
# SDK stdio server
# ---------------------------------------------------------------------------
def run_stdio_server(registry: ToolRegistry) -> None:
    """Drive the MCP stdio loop using the installed ``mcp`` Python SDK.

    The MCP SDK is imported inside this function so the cold module
    import doesn't pay the SDK cost, and so the ``--list-tools`` CLI
    path remains usable on hosts where the SDK is absent.

    The implementation targets the ``mcp.server.Server`` + ``stdio_server``
    pair. We do NOT use SDK decorators against module-level callables
    because the SDK's decorator semantics shift across versions; instead,
    we explicitly register a ``list_tools`` handler and a ``call_tool``
    dispatcher that reads from :class:`ToolRegistry`.
    """
    # Hard preflight gate (per Plan Phase 2 §"MCP server"): aborts before
    # any LLM-driven loop begins if pandoc/latexmk/mcp are missing.
    _ensure_preflight()

    try:
        import asyncio

        from mcp.server import Server  # type: ignore[import-not-found]
        from mcp.server.stdio import stdio_server  # type: ignore[import-not-found]
        from mcp.types import TextContent, Tool  # type: ignore[import-not-found]
    except ImportError as e:
        raise PreflightFailure(
            code="mcp_sdk_missing",
            message=f"mcp Python SDK not importable: {e}",
            hint=(
                "Install with `pip install 'mcp>=1.0'`. The --list-tools "
                "CLI path does not require the SDK."
            ),
            retryable=False,
        ) from e

    server = Server("litchron")

    @server.list_tools()  # type: ignore[misc]
    async def _list_tools() -> list[Any]:
        return [
            Tool(
                name=t.name,
                description=t.description,
                inputSchema={"type": "object", "additionalProperties": True},
            )
            for t in [registry.get(n) for n in registry.names()]
        ]

    @server.call_tool()  # type: ignore[misc]
    async def _call_tool(name: str, arguments: dict[str, Any]) -> list[Any]:
        entry = registry.get(name)
        result = entry.fn(**(arguments or {}))
        # Result models expose .model_dump(); dicts pass through; everything
        # else is rendered via repr for the LLM.
        if hasattr(result, "model_dump"):
            payload = result.model_dump(mode="json")
        elif isinstance(result, dict):
            payload = result
        else:
            payload = {"value": repr(result)}
        return [TextContent(type="text", text=json.dumps(payload))]

    async def _amain() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    asyncio.run(_amain())


# ---------------------------------------------------------------------------
# Module entrypoint
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Module entrypoint: dispatches ``--list-tools`` vs stdio loop."""
    args = list(argv) if argv is not None else sys.argv[1:]
    registry = build_registry()

    if "--list-tools" in args:
        return _print_tools_json(registry)

    run_stdio_server(registry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RegisteredTool",
    "ToolRegistry",
    "build_registry",
    "run_stdio_server",
    "main",
    "_ensure_preflight",
]
