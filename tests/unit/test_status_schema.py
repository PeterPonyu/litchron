"""Spec §5.12: StatusResult shape and all_green → empty suggested_next_tools."""
from __future__ import annotations

from litchron.state import (
    CitationFingerprint,
    QualityFlag,
    RunState,
    SuggestedTool,
    default_state,
)


def _fully_green_state() -> RunState:
    """Construct a RunState that satisfies all all_green preconditions."""
    s = default_state(run_id="green-run", h5ad_path="/tmp/test.h5ad")
    s.started_at = "2026-01-01T00:00:00+00:00"
    s.baselines_done = ["paga", "palantir"]
    s.baselines_all_done = True
    s.llm_ordering_done = True
    s.citations_verified = [
        CitationFingerprint(
            scheme="doi",
            id="10.1000/test.001",
            verified_at="2026-01-01T00:00:00+00:00",
            source="crossref",
            confidence=0.9,
        )
    ]
    s.comparison_done = True
    s.latex_compiled = True
    s.all_green = True
    s.quality_flags = []
    s.suggested_next_tools = []
    return s


def test_status_result_has_all_green_field() -> None:
    s = _fully_green_state()
    assert hasattr(s, "all_green")
    assert isinstance(s.all_green, bool)


def test_status_result_has_suggested_next_tools() -> None:
    s = _fully_green_state()
    assert hasattr(s, "suggested_next_tools")
    assert isinstance(s.suggested_next_tools, list)


def test_status_result_has_quality_flags() -> None:
    s = _fully_green_state()
    assert hasattr(s, "quality_flags")
    assert isinstance(s.quality_flags, list)


def test_suggested_next_tools_empty_when_all_green() -> None:
    """When all_green is True, suggested_next_tools must be empty (spec §5.12)."""
    s = _fully_green_state()
    assert s.all_green is True
    assert s.suggested_next_tools == [], (
        "suggested_next_tools must be empty when all_green == True"
    )


def test_suggested_tool_shape() -> None:
    """SuggestedTool has tool, args, rationale fields."""
    tool = SuggestedTool(
        tool="run_baseline",
        args={"method": "paga"},
        rationale="PAGA not yet run",
    )
    assert tool.tool == "run_baseline"
    assert isinstance(tool.args, dict)
    assert isinstance(tool.rationale, str)


def test_quality_flag_values() -> None:
    """QualityFlag is a Literal covering the documented set."""
    from typing import get_args

    expected = {
        "no_verified_citations",
        "baseline_disagreement_severe",
        "root_cell_ambiguous",
        "preflight_partial",
        "baseline_failure",
    }
    actual = set(get_args(QualityFlag))
    assert actual == expected


def test_state_with_suggested_tools_not_green() -> None:
    """A non-green run can carry suggested_next_tools."""
    s = default_state(run_id="partial-run", h5ad_path="/tmp/test.h5ad")
    s.started_at = "2026-01-01T00:00:00+00:00"
    s.suggested_next_tools = [
        SuggestedTool(
            tool="run_baseline",
            args={"method": "paga"},
            rationale="PAGA baseline not yet complete",
        )
    ]
    assert s.all_green is False
    assert len(s.suggested_next_tools) == 1
