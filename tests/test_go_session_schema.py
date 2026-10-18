from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from typing import Any

from llmtestlab.assertions import Expectation
from llmtestlab.types import SessionTrace, first_present, normalize_api_type

GO_CALL_KEYS = [
    "id",
    "session_id",
    "session_source",
    "seq",
    "started_at",
    "first_byte_at",
    "ended_at",
    "latency_ms",
    "provider",
    "api_family",
    "upstream",
    "request",
    "response",
    "request_ids",
    "raw_request",
    "raw_response",
    "stream",
    "http_status",
    "normalized",
]


def chat_tool_request(model: str = "fake-model") -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        "tools": [
            {
                "type": "function",
                "function": {"name": "get_weather", "parameters": {"type": "object"}},
            }
        ],
    }


def chat_tool_response(model: str = "fake-model") -> dict[str, Any]:
    return {
        "id": "chatcmpl-tool",
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_weather",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": json.dumps({"city": "Paris"}),
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def chat_final_request(model: str = "fake-model") -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "user", "content": "What is the weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_weather",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"city": "Paris"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_weather",
                "content": '{"weather":"sunny"}',
            },
        ],
    }


def chat_final_response(model: str = "fake-model") -> dict[str, Any]:
    return {
        "id": "chatcmpl-final",
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": '{"answer":"sunny"}'},
            }
        ],
        "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
    }


def go_call(
    *,
    seq: int,
    api_family: str,
    request_body: Mapping[str, Any],
    response_body: Mapping[str, Any],
    encode: str = "json",
    request: object = None,
    response: object = None,
    normalized: str = "meta",
) -> dict[str, Any]:
    raw_request: object
    raw_response: object
    if encode == "base64":
        raw_request = base64.b64encode(json.dumps(request_body).encode("utf-8")).decode("ascii")
        raw_response = base64.b64encode(json.dumps(response_body).encode("utf-8")).decode("ascii")
    else:
        raw_request = json.dumps(request_body)
        raw_response = json.dumps(response_body)
    payload: dict[str, Any] = {
        "id": f"call-{seq}",
        "session_id": "sess",
        "session_source": "header",
        "seq": seq,
        "started_at": "2026-01-01T00:00:00Z",
        "first_byte_at": "2026-01-01T00:00:00Z",
        "ended_at": "2026-01-01T00:00:01Z",
        "latency_ms": 12 if seq == 1 else 41,
        "provider": "openai",
        "api_family": api_family,
        "upstream": "http://127.0.0.1:9",
        "request": request,
        "response": response,
        "request_ids": ["req_1"],
        "raw_request": raw_request,
        "raw_response": raw_response,
        "stream": False,
        "http_status": 200,
        "normalized": (
            {
                "model": request_body.get("model"),
                "api_family": api_family,
                "request": request_body,
                "response": response_body,
            }
            if normalized == "full"
            else {"model": request_body.get("model"), "api_family": api_family}
        ),
    }
    assert list(payload) == GO_CALL_KEYS
    return payload


def go_chat_trace(*, encode: str = "json") -> SessionTrace:
    return SessionTrace.from_payload(
        "test-session",
        {
            "id": "test-session",
            "calls": [
                go_call(
                    seq=1,
                    api_family="chat.completions",
                    request_body=chat_tool_request(),
                    response_body=chat_tool_response(),
                    encode=encode,
                ),
                go_call(
                    seq=2,
                    api_family="chat.completions",
                    request_body=chat_final_request(),
                    response_body=chat_final_response(),
                    encode=encode,
                ),
            ],
        },
    )


def test_first_present_skips_null_values() -> None:
    mapping = {"request": None, "raw_request": {"model": "x"}}
    assert first_present(mapping, "request", "raw_request") == {"model": "x"}


def test_normalize_api_family_spellings() -> None:
    assert normalize_api_type("chat.completions") == "chat_completions"
    assert normalize_api_type("openai.chat.completions") == "chat_completions"
    assert normalize_api_type("chat_completions") == "chat_completions"
    assert normalize_api_type("responses") == "responses"
    assert normalize_api_type("openai.responses") == "responses"
    assert normalize_api_type("response") == "responses"


def test_parses_go_call_keys_with_null_request_response() -> None:
    trace = go_chat_trace()
    assert [call.api_type for call in trace.calls] == ["chat_completions", "chat_completions"]
    assert [call.model for call in trace.calls] == ["fake-model", "fake-model"]
    Expectation(trace).tool_called("get_weather").args_subset(
        "get_weather", {"city": "Paris"}
    ).tool_result_fed_back("get_weather").response_json_valid().no_errors()
    filtered = trace.filtered(api_type="chat_completions", model="fake-model")
    assert len(filtered.calls) == 2
    summary = trace.trace_summary()
    assert "api=chat_completions" in summary
    assert "tools=[get_weather]" in summary


def test_parses_go_base64_raw_request_and_response() -> None:
    trace = go_chat_trace(encode="base64")
    Expectation(trace).tool_called("get_weather").args_subset("get_weather", {"city": "Paris"})


def test_parses_http_envelope_with_json_string_body() -> None:
    request_body = chat_tool_request()
    response_body = chat_tool_response()
    call = go_call(
        seq=1,
        api_family="chat.completions",
        request_body=request_body,
        response_body=response_body,
        request={
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": {"content-type": "application/json"},
            "body": json.dumps(request_body),
        },
        response={
            "status": 200,
            "path": "/v1/chat/completions",
            "body": json.dumps(response_body),
        },
    )
    trace = SessionTrace.from_payload("envelope", {"id": "envelope", "calls": [call]})
    captured = trace.calls[0]
    assert captured.api_type == "chat_completions"
    assert captured.path == "/v1/chat/completions"
    assert captured.tool_calls[0].name == "get_weather"


def test_parses_responses_api_family_and_output() -> None:
    request_body = {
        "model": "fake-model",
        "input": "What is the weather in Paris?",
        "tools": [{"type": "function", "name": "get_weather"}],
    }
    response_body = {
        "id": "resp_tool",
        "model": "fake-model",
        "output": [
            {
                "type": "function_call",
                "call_id": "call_weather",
                "name": "get_weather",
                "arguments": json.dumps({"city": "Paris"}),
            }
        ],
    }
    final_request = {
        "model": "fake-model",
        "input": [
            response_body["output"][0],
            {
                "type": "function_call_output",
                "call_id": "call_weather",
                "output": '{"weather":"sunny"}',
            },
        ],
    }
    final_response = {
        "id": "resp_final",
        "model": "fake-model",
        "output_text": '{"answer":"sunny"}',
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": '{"answer":"sunny"}'}],
            }
        ],
    }
    trace = SessionTrace.from_payload(
        "responses",
        {
            "id": "responses",
            "calls": [
                go_call(
                    seq=1,
                    api_family="openai.responses",
                    request_body=request_body,
                    response_body=response_body,
                ),
                go_call(
                    seq=2,
                    api_family="openai.responses",
                    request_body=final_request,
                    response_body=final_response,
                ),
            ],
        },
    )
    assert [call.api_type for call in trace.calls] == ["responses", "responses"]
    filtered = trace.filtered(api_type="responses")
    Expectation(filtered).tool_called("get_weather").tool_result_fed_back(
        "get_weather"
    ).response_json_valid()


def test_query_filter_keeps_go_chat_calls() -> None:
    trace = go_chat_trace()
    kept = trace.filtered(api_type="chat_completions")
    assert [call.api_type for call in kept.calls] == ["chat_completions", "chat_completions"]
    empty = trace.filtered(api_type="responses")
    assert empty.calls == []
    assert list(empty.raw.keys()) == ["id", "calls"]


def test_prefers_request_model_over_response_snapshot() -> None:
    call = go_call(
        seq=1,
        api_family="chat.completions",
        request_body=chat_tool_request("gpt-4.1-mini"),
        response_body=chat_tool_response("gpt-4.1-mini-2025-04-14"),
    )
    trace = SessionTrace.from_payload("snap", {"id": "snap", "calls": [call]})
    assert trace.calls[0].model == "gpt-4.1-mini"
    assert len(trace.filtered(model="gpt-4.1-mini").calls) == 1


def test_model_filter_does_not_match_sibling_models() -> None:
    payload = {
        "id": "siblings",
        "calls": [
            {
                "api_family": "chat.completions",
                "request": {"model": "gpt-4.1-mini"},
                "response": {"choices": []},
            },
            {
                "api_family": "chat.completions",
                "request": {"model": "gpt-4.1-nano"},
                "response": {"choices": []},
            },
        ],
    }
    trace = SessionTrace.from_payload("siblings", payload)
    assert [call.model for call in trace.filtered(model="gpt-4.1").calls] == []
    assert [call.model for call in trace.filtered(model="gpt-4.1-mini").calls] == ["gpt-4.1-mini"]


def test_model_filter_accepts_openai_snapshot_id() -> None:
    payload = {
        "id": "snap",
        "calls": [
            {
                "api_family": "chat.completions",
                "request": None,
                "response": {"model": "gpt-4.1-mini-2025-04-14", "choices": []},
            }
        ],
    }
    trace = SessionTrace.from_payload("snap", payload)
    assert trace.calls[0].model == "gpt-4.1-mini-2025-04-14"
    assert len(trace.filtered(api_type="chat_completions", model="gpt-4.1-mini").calls) == 1


def test_sparse_request_response_do_not_hide_raw_payloads() -> None:
    request_body = chat_tool_request()
    response_body = chat_tool_response()
    call = go_call(
        seq=1,
        api_family="chat.completions",
        request_body=request_body,
        response_body=response_body,
        request={"model": "", "messages": None, "path": "/v1/chat/completions"},
        response={"model": "", "choices": None},
    )
    trace = SessionTrace.from_payload("sparse", {"id": "sparse", "calls": [call]})
    captured = trace.calls[0]
    assert captured.api_type == "chat_completions"
    assert captured.tool_calls[0].name == "get_weather"
    assert captured.request["messages"][0]["content"] == "What is the weather in Paris?"


def test_http_envelope_with_model_key_uses_body() -> None:
    request_body = chat_tool_request()
    response_body = chat_tool_response()
    call = go_call(
        seq=1,
        api_family="chat.completions",
        request_body=request_body,
        response_body=response_body,
        request={
            "method": "POST",
            "path": "/v1/chat/completions",
            "model": "ignored-wrapper",
            "body": json.dumps(request_body),
        },
        response={
            "status": 200,
            "model": "ignored-wrapper",
            "body": json.dumps(response_body),
        },
    )
    captured = SessionTrace.from_payload("env", {"id": "env", "calls": [call]}).calls[0]
    assert captured.model == "fake-model"
    assert captured.tool_calls[0].name == "get_weather"


def test_parses_json_byte_array_raw_response() -> None:
    response_body = chat_tool_response()
    call = go_call(
        seq=1,
        api_family="chat.completions",
        request_body=chat_tool_request(),
        response_body=response_body,
        request=None,
        response=None,
    )
    call["raw_response"] = list(json.dumps(response_body).encode("utf-8"))
    captured = SessionTrace.from_payload("bytes", {"id": "bytes", "calls": [call]}).calls[0]
    assert captured.tool_calls[0].name == "get_weather"


def test_normalized_transcript_messages_are_invocations() -> None:
    payload = {
        "id": "transcript",
        "calls": [
            {
                "api_family": "chat.completions",
                "request": None,
                "response": None,
                "normalized": {
                    "messages": [
                        {"role": "user", "content": "weather"},
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city":"Paris"}',
                                    },
                                }
                            ],
                        },
                    ]
                },
            }
        ],
    }
    Expectation(SessionTrace.from_payload("transcript", payload)).tool_called("get_weather")
