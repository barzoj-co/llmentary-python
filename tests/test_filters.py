from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

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
