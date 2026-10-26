from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

import httpx
import pytest

from llmentary.assertions import Expectation
from llmentary.client import ControlClient, ProxyHandle, Query, Session
from llmentary.types import (
    OUTCOME_CANCELLED,
    OUTCOME_COMMITTED,
    OUTCOME_FAILED,
    SessionTrace,
)

SPECIALIST_TOOLS = [
    {"type": "function", "function": {"name": f"tool_{index}"}}
    for index in range(6)
]

ROOT = Path(__file__).resolve().parent
OUTCOME_CASES = json.loads((ROOT / "testdata" / "call_outcomes.json").read_text())["cases"]


def _call(
    *,
    seq: int,
    tools: Optional[Sequence[Mapping[str, Any]]] = None,
    messages: Optional[list[Mapping[str, Any]]] = None,
    response_format: Optional[Mapping[str, Any]] = None,
    tool_calls: Optional[list[Mapping[str, Any]]] = None,
    content: Optional[str] = None,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": "fake-model",
        "messages": messages
        or [{"role": "user", "content": f"turn-{seq}"}],
    }
    if tools is not None:
        request["tools"] = list(tools)
    if response_format is not None:
        request["response_format"] = dict(response_format)
    message: dict[str, Any] = {"role": "assistant", "content": content or "ok"}
    if tool_calls:
        message["tool_calls"] = tool_calls
        message["content"] = None
    return {
        "seq": seq,
        "api_type": "chat_completions",
        "request": request,
        "response": {"choices": [{"message": message, "finish_reason": "stop"}]},
    }


def _h1_trace() -> SessionTrace:
    return SessionTrace.from_payload(
        "h1",
        {
            "calls": [
                _call(
                    seq=1,
                    response_format={"type": "json_schema"},
                    content='{"route":"specialist"}',
                ),
                _call(
                    seq=2,
                    tools=SPECIALIST_TOOLS,
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "tool_0",
                                "arguments": '{"city":"Paris"}',
                            },
                        }
                    ],
                ),
                _call(
                    seq=3,
                    tools=SPECIALIST_TOOLS,
                    messages=[
                        {"role": "user", "content": "follow-up"},
                        {
                            "role": "tool",
                            "tool_call_id": "call_1",
                            "content": '{"weather":"sunny"}',
                        },
                    ],
                    content="sunny in Paris",
                ),
            ]
        },
    )


def test_offers_tool_and_without_tools_h1_table() -> None:
    trace = _h1_trace()
    routers = trace.without_tools()
    assert [call.sequence_number() for call in routers.calls] == [1]
    assert routers.calls[0].request["response_format"]["type"] == "json_schema"

    specialists = trace.offers_tool("tool_0")
    assert [call.sequence_number() for call in specialists.calls] == [2, 3]

    with_tools = trace.with_tools()
    assert [call.sequence_number() for call in with_tools.calls] == [2, 3]
    assert "tool_5" in with_tools.calls[0].offered_tool_names()

    follow_up = trace.since(2)
    assert [call.sequence_number() for call in follow_up.calls] == [3]
    assert '{"weather":"sunny"}' in follow_up.calls[0].prompt_text


def test_filtered_kwargs_match_chainable_methods() -> None:
    trace = _h1_trace()
    assert [call.sequence_number() for call in trace.filtered(with_tools=False).calls] == [1]
    assert [call.sequence_number() for call in trace.filtered(offers_tool="tool_3").calls] == [
        2,
        3,
    ]
    assert [call.sequence_number() for call in trace.filtered(since=1).calls] == [2, 3]


def test_call_count_exact_and_range() -> None:
    expect = Expectation(_h1_trace())
    expect.call_count(3)
    expect.call_count(min=2, max=4)
    expect.call_count(min=3)
    expect.call_count(max=3)

    with pytest.raises(AssertionError, match="expected 2 captured call\\(s\\), got 3") as caught:
        Expectation(_h1_trace()).call_count(2)
    message = str(caught.value)
    assert "llmentary session h1" in message
    assert "3 captured call" in message

    with pytest.raises(AssertionError, match="expected at least 4 captured call\\(s\\), got 3"):
        Expectation(_h1_trace()).call_count(min=4)
    with pytest.raises(AssertionError, match="expected at most 2 captured call\\(s\\), got 3"):
        Expectation(_h1_trace()).call_count(max=2)
    with pytest.raises(TypeError, match="exact count or min/max"):
        Expectation(_h1_trace()).call_count()
    with pytest.raises(TypeError, match="not both"):
        Expectation(_h1_trace()).call_count(2, min=1)
    Expectation(_h1_trace().since(3)).call_count(0)
    Expectation(_h1_trace().since(3)).call_count(min=0, max=0)
    with pytest.raises(TypeError, match="non-negative integer"):
        Expectation(_h1_trace()).call_count(-1)
    with pytest.raises(TypeError, match="non-negative integer"):
        Expectation(_h1_trace().since(3)).call_count(min=-1)
    with pytest.raises(TypeError, match="non-negative integer"):
        Expectation(_h1_trace()).call_count(True)
    with pytest.raises(TypeError, match="non-negative integer"):
        Expectation(_h1_trace()).call_count(min=True)
    with pytest.raises(TypeError, match="non-negative integer"):
        Expectation(_h1_trace()).call_count("2")  # type: ignore[arg-type]


def test_empty_filter_no_errors_and_not_called_fail() -> None:
    trace = _h1_trace()
    cases = [
        trace.without_tools().since(1),
        trace.offers_tool("missing"),
        trace.since(3),
    ]
    for empty in cases:
        assert empty.calls == []
        with pytest.raises(AssertionError, match="no calls matched query") as no_errors:
            Expectation(empty).no_errors()
        with pytest.raises(AssertionError, match="no calls matched query") as not_called:
            Expectation(empty).not_called("tool_0")
        for caught in (no_errors, not_called):
            message = str(caught.value)
            assert "no calls matched query (" in message
            assert "session has" not in message
            assert "llmentary session h1" in message
        Expectation(empty).allow_empty().no_errors()
        Expectation(empty).allow_empty().not_called("tool_0")

    with pytest.raises(
        AssertionError, match=r"no calls matched query \(without_tools, since=1\)"
    ):
        Expectation(trace.without_tools().since(1)).no_errors()
    with pytest.raises(
        AssertionError, match=r'no calls matched query \(offers_tool\("missing"\)\)'
    ):
        Expectation(trace.offers_tool("missing")).not_called("x")
    with pytest.raises(AssertionError, match=r"no calls matched query \(since=3\)"):
        Expectation(trace.since(3)).no_errors()


def test_non_empty_filter_no_errors_still_passes() -> None:
    Expectation(_h1_trace().without_tools()).call_count(1).no_errors()
    Expectation(_h1_trace().offers_tool("tool_0")).call_count(2).not_called("missing")


def _vector_trace(session_id: str = "dog-58") -> SessionTrace:
    calls = [dict(case["call"]) for case in OUTCOME_CASES]
    return SessionTrace.from_payload(session_id, {"id": session_id, "calls": calls})


def test_outcome_filters_count_shared_vector_calls() -> None:
    trace = _vector_trace()
    assert len(trace.calls) == len(OUTCOME_CASES)

    cancelled = trace.filtered(outcomes=(OUTCOME_CANCELLED,))
    failed = trace.filtered(outcomes=(OUTCOME_FAILED,))
    committed = trace.filtered(outcomes=(OUTCOME_COMMITTED,))

    assert [call.seq for call in cancelled.calls] == [3, 4, 5]
    assert len(cancelled.calls) == 3
    assert len(failed.calls) == 9
    assert len(committed.calls) == 3
    assert all(call.outcome == OUTCOME_CANCELLED for call in cancelled.calls)
    assert all(call.outcome == OUTCOME_FAILED for call in failed.calls)
    assert all(call.outcome == OUTCOME_COMMITTED for call in committed.calls)
    assert len(cancelled.calls) + len(failed.calls) + len(committed.calls) == len(trace.calls)

    both = trace.filtered(outcomes=(OUTCOME_CANCELLED, OUTCOME_FAILED))
    assert len(both.calls) == 12
    assert [call.seq for call in both.calls] == list(range(3, 15))

    Expectation(cancelled).call_count(3, include_cancelled=True)
    Expectation(committed).call_count(3).no_errors()
    Expectation(both).call_count(min=12, max=12, include_cancelled=True)
    with pytest.raises(AssertionError, match=r"got 9 failed call\(s\)"):
        Expectation(failed).call_count(9).no_errors()


def test_committed_and_aborted_counts_are_unchanged() -> None:
    trace = _vector_trace()
    committed = trace.committed()
    aborted = trace.aborted()
    assert len(committed.calls) == 11
    assert len(aborted.calls) == 4
    not_aborted = [call for call in trace.calls if not call.client_aborted]
    assert [call.seq for call in committed.calls] == [call.seq for call in not_aborted]
    Expectation(committed).call_count(11)
    Expectation(aborted).call_count(4, include_cancelled=True)
    assert all(not call.client_aborted for call in committed.calls)
    failed_committed = [call.seq for call in committed.calls if call.outcome == OUTCOME_FAILED]
    assert failed_committed == [6, 8, 9, 10, 11, 12, 13, 14]


def test_outcome_filter_is_labelled_in_empty_match_message() -> None:
    committed_calls = [
        dict(case["call"]) for case in OUTCOME_CASES if case["outcome"] == OUTCOME_COMMITTED
    ]
    committed_only = SessionTrace.from_payload(
        "committed-only",
        {"id": "committed-only", "calls": committed_calls},
    )
    empty = committed_only.filtered(outcomes=(OUTCOME_CANCELLED,))
    assert empty.calls == []
    assert "outcome=cancelled" in empty.empty_match_message()
    with pytest.raises(AssertionError, match=r"no calls matched query \(outcome=cancelled\)"):
        Expectation(empty).no_errors()
    with pytest.raises(AssertionError, match=r"no calls matched query \(outcome=cancelled\)"):
        Expectation(empty).not_called("tool_0")
    Expectation(empty).allow_empty().no_errors()
    Expectation(empty).allow_empty().not_called("tool_0")

    mixed = _vector_trace().filtered(outcomes=(OUTCOME_CANCELLED, OUTCOME_FAILED))
    assert "outcome=cancelled|failed" in mixed.empty_match_message()
    none_left = mixed.filtered(outcomes=(OUTCOME_COMMITTED,))
    with pytest.raises(
        AssertionError,
        match=r"no calls matched query \(outcome=cancelled\|failed, outcome=committed\)",
    ):
        Expectation(none_left).no_errors()


def test_query_outcome_filters_pass_through_to_the_trace() -> None:
    payload = {"id": "s", "calls": [dict(case["call"]) for case in OUTCOME_CASES]}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/wait") or request.url.path.endswith("/sessions/s"):
            return httpx.Response(200, json=payload)
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        session = Session(
            session_id="s",
            proxy=ProxyHandle(
                url="http://127.0.0.1:7777",
                control=ControlClient("http://127.0.0.1:7778", client=http),
            ),
        )
        query = session.query()
        assert query.cancelled().outcomes == (OUTCOME_CANCELLED,)
        assert query.failed().outcomes == (OUTCOME_FAILED,)
        assert query.outcome(OUTCOME_COMMITTED).outcomes == (OUTCOME_COMMITTED,)
        assert query.outcome("committed", "cancelled").outcomes == ("committed", "cancelled")
        assert query.outcomes is None
        assert query.committed().outcomes is None
        assert query.committed().aborted_only is False

        query.cancelled().expect().call_count(3, include_cancelled=True)
        query.failed().expect().call_count(9)
        query.outcome(OUTCOME_COMMITTED).expect().call_count(3)
        query.outcome(OUTCOME_CANCELLED, OUTCOME_FAILED).expect().call_count(
            12, include_cancelled=True
        )
        query.committed().expect().call_count(11)
        query.aborted().expect().call_count(4, include_cancelled=True)
        query.failed().committed().expect().call_count(8)


def _validation_query() -> Query:
    session = Session(
        session_id="outcome-validation",
        proxy=ProxyHandle(
            url="http://127.0.0.1:7777",
            control=ControlClient("http://127.0.0.1:7778"),
        ),
    )
    return session.query()


def test_query_outcome_rejects_no_states() -> None:
    with pytest.raises(ValueError, match="at least one state"):
        _validation_query().outcome()


def test_query_outcome_rejects_unknown_state() -> None:
    with pytest.raises(ValueError, match="bogus"):
        _validation_query().outcome("bogus")
    with pytest.raises(ValueError, match="bogus"):
        _validation_query().outcome(OUTCOME_COMMITTED, "bogus")


def test_in_flight_call_is_excluded_by_outcome_filter() -> None:
    hanging = {
        "id": "call_hanging",
        "seq": 1,
        "api_type": "chat_completions",
        "started_at": "2026-10-26T10:00:00Z",
        "client_aborted": False,
        "request": {"model": "fake-model", "messages": [{"role": "user", "content": "hang"}]},
    }
    trace = SessionTrace.from_payload("in-flight", {"id": "in-flight", "calls": [hanging]})
    assert trace.calls[0].http_status is None

    for state in (OUTCOME_FAILED, OUTCOME_CANCELLED, OUTCOME_COMMITTED):
        filtered = trace.filtered(outcomes=(state,))
        assert filtered.calls == []
        assert f"outcome={state}" in filtered.empty_match_message()
        Expectation(filtered).allow_empty().no_errors()

    Expectation(trace.filtered(outcomes=(OUTCOME_FAILED,))).call_count(0)
