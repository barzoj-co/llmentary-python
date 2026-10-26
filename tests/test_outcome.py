from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from llmentary.types import (
    OUTCOME_CANCELLED,
    OUTCOME_COMMITTED,
    OUTCOME_FAILED,
    CapturedCall,
)

ROOT = Path(__file__).resolve().parent
OUTCOME_CASES = json.loads((ROOT / "testdata" / "call_outcomes.json").read_text())
ABORT_CASES = [row for row in OUTCOME_CASES["cases"] if "abort_phase" in row["call"]]


@pytest.mark.parametrize("case", OUTCOME_CASES["cases"], ids=lambda row: row["name"])
def test_outcome_matches_shared_vector(case: dict[str, Any]) -> None:
    call = CapturedCall(raw=case["call"], index=0)
    assert call.outcome == case["outcome"]
    assert call.outcome in (OUTCOME_COMMITTED, OUTCOME_CANCELLED, OUTCOME_FAILED)


def test_outcome_constants_are_the_three_vector_values() -> None:
    expected = {str(row["outcome"]) for row in OUTCOME_CASES["cases"]}
    assert {OUTCOME_COMMITTED, OUTCOME_CANCELLED, OUTCOME_FAILED} == expected


@pytest.mark.parametrize("case", ABORT_CASES, ids=lambda row: row["name"])
def test_abort_phase_never_changes_the_outcome(case: dict[str, Any]) -> None:
    recorded = dict(case["call"])
    base = CapturedCall(raw=recorded, index=0).outcome
    assert base == case["outcome"]
    for phase in ("before_headers", "after_headers"):
        flipped = dict(recorded)
        flipped["abort_phase"] = phase
        assert CapturedCall(raw=flipped, index=0).outcome == base
