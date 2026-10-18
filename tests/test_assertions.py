from __future__ import annotations

import pytest

from llmtestlab.assertions import Expectation, SoftExpectation
from llmtestlab.types import SessionTrace


def trace() -> SessionTrace:
    return SessionTrace.from_payload(
        "test-session",
        {
            "calls": [
                {
                    "api_type": "chat_completions",
                    "request": {
                        "body": {
                            "model": "gpt-4.1-mini",
                            "messages": [{"role": "user", "content": "weather in Paris"}],
                            "temperature": 0,
                            "parallel_tool_calls": True,
                        }
                    },
                    "response": {
                        "body": {
                            "model": "gpt-4.1-mini",
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
                                                    "arguments": '{"city": "Paris", "unit": "c"}',
                                                },
                                            },
                                            {
                                                "id": "call_2",
                                                "type": "function",
                                                "function": {
                                                    "name": "get_time",
                                                    "arguments": '{"city": "Paris"}',
                                                },
                                            },
                                        ]
                                    },
                                }
                            ],
                            "usage": {
                                "prompt_tokens": 10,
                                "completion_tokens": 5,
                                "total_tokens": 15,
                            },
                        }
                    },
                    "latency_ms": 123,
                    "ttft_ms": 45,
                },
                {
                    "api_type": "chat_completions",
                    "request": {
                        "body": {
                            "model": "gpt-4.1-mini",
                            "messages": [
                                {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}
                            ],
                        }
                    },
                    "response": {
                        "body": {
                            "choices": [
                                {
                                    "finish_reason": "stop",
                                    "message": {"content": '{"answer": "It is sunny."}'},
                                }
                            ]
                        }
                    },
                },
            ]
        },
    )


def test_expectation_tool_and_content_assertions() -> None:
    Expectation(trace()).tool_called("get_weather").called_times("get_weather", 1).args_subset(
        "get_weather", {"city": "Paris"}
    ).order("get_weather", "get_time").parallel_tool_calls().tool_result_fed_back(
        "get_weather"
    ).model(
        "gpt-4.1-mini"
    ).request_param(
        "temperature", 0
    ).finish_reason(
        "stop"
    ).tokens(
        min_total=15, max_total=15
    ).latency(
        max_ms=200
    ).ttft(
        min_ms=10
    ).prompt_contains(
        "weather"
    ).response_contains(
        "sunny"
    ).response_json_valid().response_json_schema(
        {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }
    ).no_errors().no_retries()


def test_failure_includes_trace_summary() -> None:
    with pytest.raises(AssertionError, match="llm-test-lab session test-session"):
        Expectation(trace()).tool_called("missing_tool")


def test_soft_expectation_accumulates_failures() -> None:
    expectation = SoftExpectation(trace())
    expectation.tool_called("missing_tool").not_called("get_weather")
    with pytest.raises(AssertionError, match="2 soft assertion"):
        expectation.assert_all()


def test_json_valid_checks_model_output_not_response_envelope() -> None:
    plain_text_trace = SessionTrace.from_payload(
        "plain-text",
        {
            "calls": [
                {
                    "response": {
                        "body": {
                            "choices": [
                                {
                                    "message": {"content": "this is not json"},
                                    "finish_reason": "stop",
                                }
                            ]
                        }
                    }
                }
            ]
        },
    )

    with pytest.raises(AssertionError, match="valid JSON"):
        Expectation(plain_text_trace).response_json_valid()


def test_tokens_only_require_requested_usage_field() -> None:
    total_only_trace = SessionTrace.from_payload(
        "usage",
        {"calls": [{"response": {"body": {"usage": {"total_tokens": 42}}}}]},
    )

    Expectation(total_only_trace).tokens(min_total=40, max_total=50)
