"""Tests for validate_run_id and generate_run_id."""
from __future__ import annotations

import re

import pytest

from litchron._runtime import generate_run_id, validate_run_id
from mcp_litchron.errors import LitchronError

_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9-]+$")


# ---------------------------------------------------------------------------
# validate_run_id: rejection cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_id", [
    "run_with_underscore",
    "run.with.dot",
    "run with space",
    "run\twith\ttab",
    "run\nwith\nnewline",
    "",
    "run!bang",
    "run@at",
    "run/slash",
])
def test_validate_rejects_invalid_ids(bad_id: str) -> None:
    with pytest.raises(LitchronError) as exc_info:
        validate_run_id(bad_id)
    assert exc_info.value.code == "invalid_run_id"


# ---------------------------------------------------------------------------
# validate_run_id: acceptance cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("good_id", [
    "my-run",
    "2026-05-18T123456-abcd1234",
    "RunABC",
    "run-001",
    "ABC123",
    "a",
    "Z",
    "litchron-dev-run-final",
])
def test_validate_accepts_valid_ids(good_id: str) -> None:
    # Must not raise.
    validate_run_id(good_id)


# ---------------------------------------------------------------------------
# generate_run_id
# ---------------------------------------------------------------------------

def test_generated_id_matches_charset() -> None:
    for _ in range(20):
        run_id = generate_run_id()
        assert _RUN_ID_PATTERN.match(run_id), (
            f"Generated run_id {run_id!r} does not match ^[A-Za-z0-9-]+$"
        )


def test_generated_ids_are_unique() -> None:
    ids = {generate_run_id() for _ in range(10)}
    assert len(ids) == 10, "Expected 10 unique run IDs"


def test_generated_id_is_not_empty() -> None:
    assert generate_run_id() != ""


def test_generated_id_passes_validate() -> None:
    run_id = generate_run_id()
    # Must not raise.
    validate_run_id(run_id)
