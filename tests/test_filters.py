from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

import pytest

from llmentary.assertions import Expectation
from llmentary.types import SessionTrace

SPECIALIST_TOOLS = [
    {"type": "function", "function": {"name": f"tool_{index}"}}
    for index in range(6)
]


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
