from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from llmentary.assertions import Expectation
from llmentary.types import (
    OUTCOME_CANCELLED,
    OUTCOME_COMMITTED,
    OUTCOME_FAILED,
    CapturedCall,
    SessionTrace,
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


def _settled_call(name: str, **extra: Any) -> dict[str, Any]:
    call: dict[str, Any] = {
        "id": name,
        "seq": 1,
        "api_type": "chat_completions",
        "started_at": "2026-10-26T10:00:00Z",
        "ended_at": "2026-10-26T10:00:01Z",
        "stream": True,
        "http_status": 200,
        "upstream_called": True,
        "client_aborted": False,
        "normalized": {"finish_reason": "stop"},
    }
    call.update(extra)
    return call


def test_mid_stream_error_without_abort_fails_a_200_call() -> None:
    call = _settled_call("mid_stream", mid_stream_error="upstream reset")
    trace = SessionTrace.from_payload(
        "mid-stream-error",
        {"id": "mid-stream-error", "calls": [call]},
    )
    captured = trace.calls[0]
    assert captured.error is None
    assert captured.http_status == 200
    assert captured.client_aborted is False
    assert captured.outcome == OUTCOME_FAILED
    with pytest.raises(AssertionError, match=r"expected no errors, got 1 failed call\(s\): #1 200"):
        Expectation(trace).no_errors()
    Expectation(trace).mid_stream_error("upstream reset")


def test_mid_stream_error_on_aborted_call_stays_cancelled() -> None:
    trace = SessionTrace.from_payload(
        "aborted-mid-stream",
        {
            "id": "aborted-mid-stream",
            "calls": [
                _settled_call(
                    "aborted_mid_stream",
                    client_aborted=True,
                    abort_phase="after_headers",
                    mid_stream_error="upstream reset",
                    normalized={"finish_reason": "cancelled", "client_aborted": True},
                )
            ],
        },
    )
    assert trace.calls[0].outcome == OUTCOME_CANCELLED
    Expectation(trace).no_errors()
    with pytest.raises(AssertionError, match="expected no cancelled calls, got 1"):
        Expectation(trace).no_cancelled()


def test_recorded_outcome_beats_mid_stream_error() -> None:
    call = _settled_call("recorded_committed", outcome="committed", mid_stream_error="reset")
    trace = SessionTrace.from_payload("recorded", {"id": "recorded", "calls": [call]})
    assert trace.calls[0].outcome == OUTCOME_COMMITTED
    Expectation(trace).no_errors()


def test_unknown_recorded_outcome_still_derives_from_mid_stream_error() -> None:
    call = _settled_call("recorded_unknown", outcome="something_new", mid_stream_error="reset")
    assert CapturedCall(raw=call, index=0).outcome == OUTCOME_FAILED
