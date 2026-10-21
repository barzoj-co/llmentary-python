from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from typing import Any

from llmentary.assertions import Expectation
from llmentary.types import SessionTrace, first_present, normalize_api_type

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


def test_parses_stream_status_usage_details_and_error_fields() -> None:
    request_body = {"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]}
    response_body = {
        "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "total_tokens": 14,
            "prompt_tokens_details": {"cached_tokens": 3},
            "completion_tokens_details": {"reasoning_tokens": 2},
        },
    }
    call = go_call(
        seq=1,
        api_family="chat.completions",
        request_body=request_body,
        response_body=response_body,
    )
    call["stream"] = True
    call["http_status"] = 400
    call["error"] = {"error": {"type": "invalid_request_error", "code": "model_not_found"}}
    captured = SessionTrace.from_payload("details", {"id": "details", "calls": [call]}).calls[0]
    assert captured.streamed is True
    assert captured.http_status == 400
    assert captured.cached_tokens == 3
    assert captured.reasoning_tokens == 2
    assert captured.error_type == "invalid_request_error"
    assert captured.error_code == "model_not_found"


def test_parses_go_normalized_cached_input_tokens() -> None:
    request_body = {"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]}
    response_body = {"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]}
    call = go_call(
        seq=1,
        api_family="chat.completions",
        request_body=request_body,
        response_body=response_body,
    )
    call["usage"] = {
        "input_tokens": 3215,
        "output_tokens": 1,
        "total_tokens": 3216,
        "cached_input_tokens": 3072,
        "reasoning_tokens": 0,
        "source": "body",
        "raw": {
            "prompt_tokens": 3215,
            "completion_tokens": 1,
            "total_tokens": 3216,
            "prompt_tokens_details": {"cached_tokens": 3072},
        },
    }
    captured = SessionTrace.from_payload("cache", {"id": "cache", "calls": [call]}).calls[0]
    assert captured.cached_tokens == 3072
    assert captured.reasoning_tokens == 0


def test_prefers_nested_usage_details_over_zero_top_level_aliases() -> None:
    request_body = {"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]}
    response_body = {"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]}
    call = go_call(
        seq=1,
        api_family="chat.completions",
        request_body=request_body,
        response_body=response_body,
    )
    call["usage"] = {
        "input_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
        "cached_input_tokens": 0,
        "reasoning_tokens": 0,
        "source": "body",
        "raw": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 12},
            "completion_tokens_details": {"reasoning_tokens": 9},
        },
    }
    captured = SessionTrace.from_payload("nested", {"id": "nested", "calls": [call]}).calls[0]
    assert captured.cached_tokens == 12
    assert captured.reasoning_tokens == 9


def test_parses_normalized_client_aborted_on_disconnect() -> None:
    request_body = {"model": "fake-model", "messages": [{"role": "user", "content": "count"}]}
    response_body = {"choices": [{"message": {"content": "One,"}, "finish_reason": None}]}
    call = go_call(
        seq=1,
        api_family="chat.completions",
        request_body=request_body,
        response_body=response_body,
    )
    call["stream"] = True
    call["http_status"] = 200
    call["normalized"] = {
        "derived_version": 1,
        "model": {"requested": "fake-model", "returned": "fake-model"},
        "finish_reason": "unknown",
        "stream_truncated": True,
        "client_aborted": True,
        "parser_warnings": ["response copy error: context canceled"],
    }
    captured = SessionTrace.from_payload("abort", {"id": "abort", "calls": [call]}).calls[0]
    assert captured.raw.get("ended_at")
    assert captured.client_aborted is True
    assert captured.stream_truncated is True
    summary = SessionTrace.from_payload("abort", {"id": "abort", "calls": [call]}).trace_summary()
    assert "status=aborted" in summary
    assert captured.abort_phase is None


def test_top_level_client_aborted_and_abort_phase() -> None:
    request_body = {"model": "fake-model", "messages": [{"role": "user", "content": "count"}]}
    call = go_call(
        seq=1,
        api_family="chat.completions",
        request_body=request_body,
        response_body={"choices": []},
    )
    call["http_status"] = 0
    call["client_aborted"] = True
    call["abort_phase"] = "before_headers"
    call["normalized"] = {
        "client_aborted": True,
        "finish_reason": "cancelled",
    }
    captured = SessionTrace.from_payload("abort", {"id": "abort", "calls": [call]}).calls[0]
    assert captured.client_aborted is True
    assert captured.abort_phase == "before_headers"
    summary = SessionTrace.from_payload("abort", {"id": "abort", "calls": [call]}).trace_summary()
    assert "status=aborted" in summary
    assert "phase=before_headers" in summary
    Expectation(
        SessionTrace.from_payload("abort", {"id": "abort", "calls": [call]})
    ).aborted(phase="before_headers")


def test_client_aborted_falls_back_to_normalized_when_top_level_absent() -> None:
    request_body = {"model": "fake-model", "messages": [{"role": "user", "content": "count"}]}
    call = go_call(
        seq=1,
        api_family="chat.completions",
        request_body=request_body,
        response_body={"choices": []},
    )
    call["normalized"] = {
        "client_aborted": True,
        "abort_phase": "after_headers",
        "finish_reason": "cancelled",
    }
    captured = SessionTrace.from_payload("abort", {"id": "abort", "calls": [call]}).calls[0]
    assert captured.client_aborted is True
    assert captured.abort_phase == "after_headers"


def test_top_level_after_headers_abort_phase() -> None:
    request_body = {"model": "fake-model", "messages": [{"role": "user", "content": "count"}]}
    call = go_call(
        seq=1,
        api_family="chat.completions",
        request_body=request_body,
        response_body={"choices": [{"message": {"content": "One,"}}]},
    )
    call["http_status"] = 200
    call["client_aborted"] = True
    call["abort_phase"] = "after_headers"
    call["normalized"] = {"finish_reason": "cancelled", "client_aborted": True}
    captured = SessionTrace.from_payload("abort", {"id": "abort", "calls": [call]}).calls[0]
    assert captured.client_aborted is True
    assert captured.abort_phase == "after_headers"
    Expectation(
        SessionTrace.from_payload("abort", {"id": "abort", "calls": [call]})
    ).aborted(phase="after_headers")


def test_done_then_close_is_not_aborted() -> None:
    request_body = {"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]}
    call = go_call(
        seq=1,
        api_family="chat.completions",
        request_body=request_body,
        response_body={"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]},
    )
    call["http_status"] = 200
    call["stream"] = True
    call["normalized"] = {
        "finish_reason": "stop",
        "client_aborted": False,
    }
    captured = SessionTrace.from_payload("done", {"id": "done", "calls": [call]}).calls[0]
    assert captured.client_aborted is False
    assert captured.abort_phase is None
    assert "abort_phase" not in captured.raw
    summary = SessionTrace.from_payload("done", {"id": "done", "calls": [call]}).trace_summary()
    assert "client_aborted" not in summary
    assert "phase=" not in summary


def test_top_level_client_aborted_false_wins() -> None:
    request_body = {"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]}
    call = go_call(
        seq=1,
        api_family="chat.completions",
        request_body=request_body,
        response_body={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
    )
    call["client_aborted"] = False
    call["normalized"] = {
        "client_aborted": True,
        "abort_phase": "after_headers",
    }
    captured = SessionTrace.from_payload("ok", {"id": "ok", "calls": [call]}).calls[0]
    assert captured.client_aborted is False
    assert captured.abort_phase is None


def test_abort_phase_rejects_unknown_value() -> None:
    request_body = {"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]}
    call = go_call(
        seq=1,
        api_family="chat.completions",
        request_body=request_body,
        response_body={"choices": []},
    )
    call["client_aborted"] = True
    call["abort_phase"] = "upstream"
    captured = SessionTrace.from_payload("abort", {"id": "abort", "calls": [call]}).calls[0]
    assert captured.client_aborted is True
    assert captured.abort_phase is None


def test_missing_ended_at_is_not_client_aborted() -> None:
    request_body = {"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]}
    response_body = {"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]}
    call = go_call(
        seq=1,
        api_family="chat.completions",
        request_body=request_body,
        response_body=response_body,
    )
    call["ended_at"] = None
    call["stream"] = True
    captured = SessionTrace.from_payload("open", {"id": "open", "calls": [call]}).calls[0]
    assert captured.raw.get("ended_at") in (None, "")
    assert captured.client_aborted is False
    assert captured.stream_truncated is False


def test_parses_error_from_raw_response_without_top_level_error() -> None:
    request_body = {"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]}
    call = go_call(
        seq=1,
        api_family="chat.completions",
        request_body=request_body,
        response_body={"choices": []},
        response=None,
    )
    call["http_status"] = 401
    call["raw_response"] = json.dumps(
        {
            "error": {
                "message": "invalid api key",
                "type": "invalid_request_error",
                "code": "invalid_api_key",
            }
        }
    )
    captured = SessionTrace.from_payload("auth", {"id": "auth", "calls": [call]}).calls[0]
    assert captured.http_status == 401
    assert captured.error_type == "invalid_request_error"
    assert captured.error_code == "invalid_api_key"
    assert captured.error


def test_null_error_on_successful_response_is_not_an_error() -> None:
    request_body = {"model": "fake-model", "input": "ping"}
    response_body = {
        "id": "resp_ok",
        "status": "completed",
        "error": None,
        "output_text": "ping",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "ping"}]}],
        "usage": {"input_tokens": 4, "output_tokens": 1, "total_tokens": 5},
    }
    call = go_call(
        seq=1,
        api_family="openai.responses",
        request_body=request_body,
        response_body=response_body,
        response=None,
    )
    captured = SessionTrace.from_payload("ok", {"id": "ok", "calls": [call]}).calls[0]
    assert captured.error is None
    assert captured.error_mapping == {}
    Expectation(SessionTrace.from_payload("ok", {"id": "ok", "calls": [call]})).no_errors()


def test_path_from_upstream_url_for_passthrough() -> None:
    call = go_call(
        seq=1,
        api_family="openai.responses",
        request_body={"model": "fake-model"},
        response_body={"object": "list", "data": [{"id": "gpt-4.1-nano"}]},
        request=None,
        response=None,
    )
    call["upstream"] = {"method": "GET", "url": "https://api.openai.com/v1/models"}
    captured = SessionTrace.from_payload("models", {"id": "models", "calls": [call]}).calls[0]
    assert (captured.path or "").rstrip("/").endswith("/models")
