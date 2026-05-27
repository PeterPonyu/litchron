"""Tests for the MCP server's input-schema derivation and graceful
malformed-call handling (issue #3).

The MCP server previously advertised every tool with the bland
``{"type": "object", "additionalProperties": true}`` schema and called
``entry.fn(**arguments)`` directly. Missing required kwargs and
unexpected kwargs therefore raised a raw ``TypeError`` across the MCP
boundary before any tool-level error handling ran.

The fixed behavior:

1. ``derive_input_schema(fn)`` returns an object schema with
   ``properties`` keyed by parameter name and ``required`` populated
   from the function's positional-required parameters.
2. ``dispatch_tool_call(registry, name, arguments)`` converts signature
   binding errors (missing required / unexpected keyword) into
   structured :class:`ErrorResult` payloads with code
   ``"invalid_arguments"`` and the tool name in the message.
3. The ``--list-tools`` payload exposes the per-tool ``inputSchema``
   including ``required`` for at least one tool with required params.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from mcp_litchron.errors import ErrorResult
from mcp_litchron.server import (
    ToolRegistry,
    _list_tools_payload,
    build_registry,
    derive_input_schema,
    dispatch_tool_call,
)


# ---------------------------------------------------------------------------
# derive_input_schema
# ---------------------------------------------------------------------------
def test_derive_input_schema_required_for_no_default_params() -> None:
    """A function with only required positional params gets ``required``."""

    def fn(run_id: str, section: str) -> dict[str, Any]:
        return {"run_id": run_id, "section": section}

    schema = derive_input_schema(fn)
    assert schema["type"] == "object"
    assert set(schema["properties"]) == {"run_id", "section"}
    assert schema["properties"]["run_id"] == {"type": "string"}
    assert schema["properties"]["section"] == {"type": "string"}
    assert sorted(schema["required"]) == ["run_id", "section"]
    assert schema["additionalProperties"] is True


def test_derive_input_schema_optionals_excluded_from_required() -> None:
    """Parameters with defaults are listed in properties but NOT required."""

    def fn(run_id: str, force: bool = False, label: str | None = None) -> None:
        return None

    schema = derive_input_schema(fn)
    assert schema["required"] == ["run_id"]
    assert "force" in schema["properties"]
    assert schema["properties"]["force"] == {"type": "boolean"}
    # Optional[str] should map to a string fragment (None is conveyed by
    # required-ness, not by emitting a "null" branch).
    assert schema["properties"]["label"]["type"] == "string"


def test_derive_input_schema_no_required_when_all_have_defaults() -> None:
    """A purely-optional callable does not emit a ``required`` key at all."""

    def fn(force: bool = False) -> None:
        return None

    schema = derive_input_schema(fn)
    assert "required" not in schema


def test_real_tools_advertise_required_run_id() -> None:
    """At least one registered tool exposes a non-empty ``required`` list.

    ``compute_observations`` (and most LitChron tools) require ``run_id``
    with no default; the published schema must reflect that so MCP
    clients can pre-validate calls.
    """
    payload = _list_tools_payload(build_registry())
    by_name = {entry["name"]: entry for entry in payload}
    assert "compute_observations" in by_name
    schema = by_name["compute_observations"]["inputSchema"]
    assert schema["required"] == ["run_id"]
    # And it must still keep additionalProperties open for backward compat.
    assert schema["additionalProperties"] is True


def test_at_least_one_tool_exposes_required_params() -> None:
    """Generic check: the listing surfaces ``required`` for at least one tool."""
    payload = _list_tools_payload(build_registry())
    tools_with_required = [p for p in payload if p["inputSchema"].get("required")]
    assert tools_with_required, (
        "no registered tool publishes a non-empty required list; "
        "schema derivation is producing nothing useful"
    )


# ---------------------------------------------------------------------------
# dispatch_tool_call
# ---------------------------------------------------------------------------
def _registry_with(fn: Any, name: str = "demo") -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(name=name, fn=fn, description="demo tool")
    return reg


def test_missing_required_arg_returns_structured_error() -> None:
    """Calling a tool without a required arg yields an ErrorResult, not an exception."""

    def needs_run_id(run_id: str) -> dict[str, Any]:
        return {"ok": True, "run_id": run_id}

    reg = _registry_with(needs_run_id, name="needs_run_id")
    result = dispatch_tool_call(reg, "needs_run_id", arguments={})
    assert isinstance(result, ErrorResult)
    assert result.code == "invalid_arguments"
    assert "needs_run_id" in result.message
    # The bind error message names the missing parameter.
    assert "run_id" in result.message
    assert result.retryable is False


def test_unexpected_kwarg_returns_structured_error() -> None:
    """Calling a tool with an extra kwarg yields a structured error."""

    def only_run_id(run_id: str) -> dict[str, Any]:
        return {"ok": True}

    reg = _registry_with(only_run_id, name="only_run_id")
    result = dispatch_tool_call(
        reg, "only_run_id", arguments={"run_id": "abc", "bogus": 1}
    )
    assert isinstance(result, ErrorResult)
    assert result.code == "invalid_arguments"
    assert "only_run_id" in result.message
    assert "bogus" in result.message


def test_valid_args_pass_through() -> None:
    """A well-formed call must NOT be intercepted by the dispatcher."""

    def ok(run_id: str, force: bool = False) -> dict[str, Any]:
        return {"run_id": run_id, "force": force}

    reg = _registry_with(ok, name="ok")
    result = dispatch_tool_call(reg, "ok", arguments={"run_id": "abc"})
    assert isinstance(result, dict)
    assert result == {"run_id": "abc", "force": False}


def test_unknown_tool_name_returns_structured_error() -> None:
    """Looking up a non-existent tool returns ErrorResult, not KeyError."""
    reg = ToolRegistry()  # empty
    result = dispatch_tool_call(reg, "nonexistent_tool", arguments={})
    assert isinstance(result, ErrorResult)
    assert result.code == "unknown_tool"
    assert "nonexistent_tool" in result.message


def test_dispatcher_serializes_error_to_json() -> None:
    """The structured error round-trips through JSON (MCP transport requirement)."""

    def needs(run_id: str) -> None:
        return None

    reg = _registry_with(needs, name="needs")
    err = dispatch_tool_call(reg, "needs", arguments={})
    assert isinstance(err, ErrorResult)
    payload = err.model_dump(mode="json")
    s = json.dumps(payload)
    assert "invalid_arguments" in s
    assert "needs" in s


# ---------------------------------------------------------------------------
# Integration with the real registry: dispatcher applies to a real tool
# ---------------------------------------------------------------------------
def test_real_tool_dispatch_missing_arg_returns_error_result(tmp_path: Any) -> None:
    """Dispatching to a real registered tool without required kwargs degrades cleanly."""
    reg = build_registry()
    res = dispatch_tool_call(reg, "compute_observations", arguments={})
    assert isinstance(res, ErrorResult)
    assert res.code == "invalid_arguments"
    assert "run_id" in res.message


def test_real_tool_dispatch_unexpected_kwarg_returns_error_result() -> None:
    """A bogus kwarg on a real tool degrades to invalid_arguments, not a crash."""
    reg = build_registry()
    res = dispatch_tool_call(
        reg, "report_status", arguments={"run_id": "abc", "totally_made_up": True}
    )
    assert isinstance(res, ErrorResult)
    assert res.code == "invalid_arguments"
    assert "totally_made_up" in res.message
