from __future__ import annotations

from typing import Any, Optional

import pytest

from llmentary.assertions import SoftExpectation
from llmentary.types import SessionTrace


def _router(seq: int, source: str, content: str) -> dict[str, Any]:
    return {
        "seq": seq,
        "session_source": source,
        "api_family": "chat.completions",
        "http_status": 200,
        "upstream": {"url": "https://api.openai.com/v1/chat/completions"},
        "request": {
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": f"route-{seq}"}],
            "response_format": {"type": "json_schema"},
        },
        "response": {
            "choices": [
                {"message": {"content": content}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
        },
    }


def _tool_turn(seq: int, source: str) -> dict[str, Any]:
    return {
        "seq": seq,
        "session_source": source,
        "api_family": "chat.completions",
        "http_status": 200,
        "upstream": "https://api.openai.com/v1/chat/completions",
        "request": {
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "weather"}],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        },
        "response": {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city":"Paris"}',
                                },
                            }
                        ]
                    },
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
        },
    }


def _text_answer(seq: int, source: str, text: str) -> dict[str, Any]:
    return {
        "seq": seq,
        "session_source": source,
        "api_family": "chat.completions",
        "http_status": 200,
        "upstream": {"url": "https://api.openai.com/v1/chat/completions"},
        "request": {
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": text}],
        },
        "response": {
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11},
        },
    }


def _http_error(
    seq: int,
    source: str,
    status: int,
    code: str,
    message: str,
) -> dict[str, Any]:
    return {
        "seq": seq,
        "session_source": source,
        "api_family": "chat.completions",
        "http_status": status,
        "normalized": {"finish_reason": "unknown"},
        "upstream": {"url": "https://api.openai.com/v1/chat/completions"},
        "request": {
            "model": "missing-model" if status == 404 else "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hi"}],
        },
        "error": {
            "type": "invalid_request_error",
            "code": code,
            "message": message,
        },
    }


def langgraph_trace() -> SessionTrace:
    return SessionTrace.from_payload(
        "graph-8",
        {
            "id": "graph-8",
            "calls": [
                _router(1, "path", '{"route":"specialist"}'),
                _tool_turn(2, "path"),
                {
                    **_text_answer(3, "path", "sunny in Paris"),
                    "request": {
                        "model": "gpt-4.1-mini",
                        "messages": [
                            {"role": "user", "content": "weather"},
                            {
                                "role": "tool",
                                "tool_call_id": "call_1",
                                "content": '{"weather":"sunny"}',
                            },
                        ],
                        "tools": [
                            {"type": "function", "function": {"name": "get_weather"}}
                        ],
                    },
                },
                _router(4, "header", '{"route":"end"}'),
                _text_answer(5, "baggage", "plain answer"),
                _text_answer(6, "default", "follow-up"),
                _http_error(7, "path", 401, "invalid_api_key", "invalid api key"),
                _http_error(8, "path", 404, "model_not_found", "model not found"),
            ],
        },
    )


def test_trace_summary_langgraph_snapshot() -> None:
    summary = langgraph_trace().trace_summary()
    assert "call keys=" not in summary
    assert "finish=unknown" not in summary
    assert "src=path" in summary
    assert "src=header" in summary
    assert "src=baggage" in summary
    assert "src=default" in summary
    assert "tokens=12/4" in summary
    assert "tools=[get_weather]" in summary
    assert "http=401" in summary
    assert "http=404" in summary
    assert "finish=stop" in summary
    assert "unparsed call" not in summary
    lines = summary.splitlines()
    assert lines[0] == "llmentary session graph-8: 8 captured call(s)"
    assert sum(1 for line in lines if line.startswith("  #")) == 8


def test_seq_alias_and_summary_prints_global_id() -> None:
    trace = SessionTrace.from_payload(
        "weather",
        {
            "calls": [
                {
                    "id": "call_3",
                    "seq": 1,
                    "api_family": "chat.completions",
                    "request": {"messages": [{"role": "user", "content": "weather"}]},
                    "response": {"choices": [{"message": {"content": "ok"}}]},
                }
            ]
        },
    )
    call = trace.calls[0]
    assert call.seq == 1
    assert call.sequence_number() == 1
    assert call.id == "call_3"
    assert "#1 (call_3)" in trace.trace_summary()


def test_unparsed_call_prints_keys_once() -> None:
    summary = SessionTrace.from_payload(
        "raw",
        {"calls": [{"id": "c1", "session_source": "header", "latency_ms": 1}]},
    ).trace_summary()
    assert "unparsed call, keys=" in summary
    assert "call keys=" not in summary


def test_soft_expectation_includes_summary_once() -> None:
    expectation = SoftExpectation(langgraph_trace())
    expectation.tool_called("missing_a").tool_called("missing_b").tool_called("missing_c")
    with pytest.raises(AssertionError, match="3 soft assertion") as caught:
        expectation.assert_all()
    message = str(caught.value)
    assert message.count("llmentary session graph-8") == 1
    assert "missing_a" in message
    assert "missing_b" in message
    assert "missing_c" in message


@pytest.mark.parametrize(
    ("source", "upstream", "expected_url"),
    [
        ("header", "https://api.openai.com/v1/chat/completions", "https://api.openai.com/v1/chat/completions"),
        (
            "baggage",
            {"url": "https://openrouter.ai/api/v1/chat/completions"},
            "https://openrouter.ai/api/v1/chat/completions",
        ),
        ("path", {"url": "https://api.openai.com/v1/chat/completions"}, "https://api.openai.com/v1/chat/completions"),
        ("default", None, None),
    ],
)
def test_session_source_and_upstream_url(
    source: str, upstream: Optional[object], expected_url: Optional[str]
) -> None:
    call: dict[str, Any] = {
        "session_source": source,
        "request": {"messages": [{"role": "user", "content": "hi"}]},
        "response": {"choices": [{"message": {"content": "ok"}}]},
    }
    if upstream is not None:
        call["upstream"] = upstream
    captured = SessionTrace.from_payload("attr", {"calls": [call]}).calls[0]
    assert captured.session_source == source
    assert captured.upstream_url == expected_url
