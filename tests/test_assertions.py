from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from llmentary.assertions import Expectation, SoftExpectation
from llmentary.types import OUTCOME_CANCELLED, OUTCOME_FAILED, SessionTrace

OUTCOME_CASES = json.loads(
    (Path(__file__).resolve().parent / "testdata" / "call_outcomes.json").read_text()
)["cases"]


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


def test_model_assertion_accepts_openai_snapshot_id() -> None:
    payload = {
        "id": "snap",
        "calls": [
            {
                "api_type": "chat_completions",
                "request": {"model": "gpt-4.1-nano-2025-04-14", "messages": []},
                "response": {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
            }
        ],
    }
    Expectation(SessionTrace.from_payload("snap", payload)).model("gpt-4.1-nano")


def test_failure_includes_trace_summary() -> None:
    with pytest.raises(AssertionError, match="llmentary session test-session"):
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


def test_parses_json_string_request_and_response_bodies() -> None:
    encoded = SessionTrace.from_payload(
        "encoded",
        {
            "calls": [
                {
                    "latency_ms": 3,
                    "request": '{"model":"fake-model","messages":[{"role":"user","content":"hi"}]}',
                    "response": (
                        '{"choices":[{"message":{"tool_calls":[{"id":"call_1","function":'
                        '{"name":"get_weather","arguments":"{\\"city\\":\\"Paris\\"}"}}]}}]}'
                    ),
                }
            ]
        },
    )
    Expectation(encoded).tool_called("get_weather").args_subset("get_weather", {"city": "Paris"})


def test_null_request_does_not_block_raw_response_tools() -> None:
    encoded = SessionTrace.from_payload(
        "null-request",
        {
            "calls": [
                {
                    "api_family": "chat.completions",
                    "request": None,
                    "response": None,
                    "raw_response": (
                        '{"choices":[{"message":{"tool_calls":[{"id":"call_1","function":'
                        '{"name":"get_weather","arguments":"{\\"city\\":\\"Paris\\"}"}}]}}]}'
                    ),
                    "normalized": {"model": "fake-model"},
                }
            ]
        },
    )
    Expectation(encoded).tool_called("get_weather").model("fake-model")


def test_tool_result_fed_back_uses_call_id_not_name_substring() -> None:
    mixed = SessionTrace.from_payload(
        "substring",
        {
            "calls": [
                {
                    "request": {"body": {"messages": [{"role": "user", "content": "weather"}]}},
                    "response": {
                        "body": {
                            "choices": [
                                {
                                    "message": {
                                        "tool_calls": [
                                            {
                                                "id": "call_1",
                                                "function": {
                                                    "name": "get_weather",
                                                    "arguments": "{}",
                                                },
                                            }
                                        ]
                                    }
                                }
                            ]
                        }
                    },
                },
                {
                    "request": {
                        "body": {
                            "messages": [
                                {
                                    "role": "tool",
                                    "tool_call_id": "call_other",
                                    "content": "get_weather mentioned in the result text",
                                }
                            ]
                        }
                    },
                    "response": {"body": {"choices": [{"message": {"content": "ok"}}]}},
                },
            ]
        },
    )
    with pytest.raises(AssertionError, match="fed back"):
        Expectation(mixed).tool_result_fed_back("get_weather")


def test_tool_result_fed_back_name_fallback_is_exact() -> None:
    named = SessionTrace.from_payload(
        "name-only",
        {
            "calls": [
                {
                    "request": {"body": {"messages": [{"role": "user", "content": "weather"}]}},
                    "response": {
                        "body": {
                            "choices": [
                                {
                                    "message": {
                                        "tool_calls": [
                                            {
                                                "function": {
                                                    "name": "get_weather",
                                                    "arguments": "{}",
                                                }
                                            }
                                        ]
                                    }
                                }
                            ]
                        }
                    },
                },
                {
                    "request": {
                        "body": {
                            "input": [
                                {
                                    "type": "function_call_output",
                                    "name": "get_weather",
                                    "output": "sunny",
                                }
                            ]
                        }
                    },
                    "response": {"body": {"output_text": "ok"}},
                },
            ]
        },
    )
    Expectation(named).tool_result_fed_back("get_weather")

    sibling = SessionTrace.from_payload(
        "sibling-name",
        {
            "calls": [
                {
                    "request": {"body": {"messages": [{"role": "user", "content": "weather"}]}},
                    "response": {
                        "body": {
                            "choices": [
                                {
                                    "message": {
                                        "tool_calls": [
                                            {
                                                "function": {
                                                    "name": "get_weather",
                                                    "arguments": "{}",
                                                }
                                            }
                                        ]
                                    }
                                }
                            ]
                        }
                    },
                },
                {
                    "request": {
                        "body": {
                            "input": [
                                {
                                    "type": "function_call_output",
                                    "name": "get_weather_v2",
                                    "output": "get_weather is mentioned in the output",
                                }
                            ]
                        }
                    },
                    "response": {"body": {"output_text": "ok"}},
                },
            ]
        },
    )
    with pytest.raises(AssertionError, match="fed back"):
        Expectation(sibling).tool_result_fed_back("get_weather")


def test_request_echoed_tool_calls_are_not_invocations() -> None:
    echoed = SessionTrace.from_payload(
        "echo",
        {
            "calls": [
                {
                    "request": {
                        "body": {
                            "messages": [
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
                                }
                            ]
                        }
                    },
                    "response": {"body": {"choices": [{"message": {"content": "done"}}]}},
                }
            ]
        },
    )
    Expectation(echoed).not_called("get_weather")


def outcome_case(name: str) -> dict[str, Any]:
    return next(case for case in OUTCOME_CASES if case["name"] == name)


def outcome_trace(case: dict[str, Any]) -> SessionTrace:
    call = dict(case["call"])
    return SessionTrace.from_payload("outcome", {"id": "outcome", "calls": [call]})


@pytest.mark.parametrize("case", OUTCOME_CASES, ids=lambda row: row["name"])
def test_no_errors_passes_committed_and_cancelled_fails_failed(case: dict[str, Any]) -> None:
    expectation = Expectation(outcome_trace(case))
    if case["outcome"] != OUTCOME_FAILED:
        expectation.no_errors()
        return
    with pytest.raises(AssertionError) as caught:
        expectation.no_errors()
    message = str(caught.value)
    assert "expected no errors, got 1 failed call(s):" in message
    assert f"#{case['call']['seq']}" in message


@pytest.mark.parametrize("case", OUTCOME_CASES, ids=lambda row: row["name"])
def test_no_cancelled_fails_on_cancelled_calls_only(case: dict[str, Any]) -> None:
    expectation = Expectation(outcome_trace(case))
    if case["outcome"] != OUTCOME_CANCELLED:
        expectation.no_cancelled()
        return
    with pytest.raises(AssertionError) as caught:
        expectation.no_cancelled()
    message = str(caught.value)
    assert "expected no cancelled calls, got 1:" in message
    assert f"#{case['call']['seq']} phase={case['call']['abort_phase']}" in message


def test_no_errors_names_every_failed_call_with_reason() -> None:
    calls = [
        dict(outcome_case("upstream_500")["call"]),
        dict(outcome_case("before_headers_abort_source_client")["call"]),
        dict(outcome_case("http_401_invalid_key")["call"]),
    ]
    trace = SessionTrace.from_payload("two-failures", {"id": "two-failures", "calls": calls})
    with pytest.raises(AssertionError) as caught:
        Expectation(trace).no_errors()
    assert (
        "expected no errors, got 2 failed call(s): #9 server_error, #10 auth"
        in str(caught.value)
    )


def test_no_errors_before_headers_cancel_is_not_an_error() -> None:
    case = outcome_case("before_headers_abort_old_proxy_no_source")
    expectation = Expectation(outcome_trace(case))
    expectation.no_errors()
    expectation.aborted(phase="before_headers")
    expectation.has_error("client_aborted")
    with pytest.raises(
        AssertionError,
        match="expected no cancelled calls, got 1: #4 phase=before_headers",
    ):
        expectation.no_cancelled()


def test_no_errors_after_headers_cancel_is_not_an_error() -> None:
    expectation = Expectation(outcome_trace(outcome_case("after_headers_abort_mid_stream")))
    expectation.no_errors()
    expectation.aborted(phase="after_headers")
    with pytest.raises(
        AssertionError,
        match="expected no cancelled calls, got 1: #5 phase=after_headers",
    ):
        expectation.no_cancelled()


def test_no_errors_ignores_legacy_mid_stream_error_on_client_cancel() -> None:
    call = dict(outcome_case("after_headers_abort_mid_stream")["call"])
    call["mid_stream_error"] = "upstream reset"
    session_id = "legacy-mid-stream"
    trace = SessionTrace.from_payload(session_id, {"id": session_id, "calls": [call]})
    Expectation(trace).no_errors()
    Expectation(trace).mid_stream_error("upstream reset")
    with pytest.raises(AssertionError, match="expected no cancelled calls, got 1: #5"):
        Expectation(trace).no_cancelled()


def test_no_cancelled_and_no_errors_pass_on_committed_trace() -> None:
    Expectation(outcome_trace(outcome_case("complete_stream"))).no_errors().no_cancelled()


def test_no_errors_and_no_cancelled_skip_in_flight_calls() -> None:
    Expectation(trace()).no_errors().no_cancelled()


def outcome_trace_of(*names: str, session_id: str = "outcome-counts") -> SessionTrace:
    calls = [dict(outcome_case(name)["call"]) for name in names]
    return SessionTrace.from_payload(session_id, {"id": session_id, "calls": calls})


def test_call_count_default_excludes_cancelled_counts_failed() -> None:
    counts = outcome_trace_of(
        "complete_stream", "before_headers_abort_source_client", "upstream_500"
    )
    Expectation(counts).call_count(2)
    Expectation(counts).call_count(3, include_cancelled=True)
    Expectation(counts).call_count(2, exclude_aborted=True)
    Expectation(counts).call_count(min=2, max=2)
    Expectation(counts).call_count(min=1, max=3, include_cancelled=True)
    Expectation(counts).call_count(min=1, max=2, exclude_aborted=True)


def test_call_count_default_failure_names_the_cancelled_calls_left_out() -> None:
    counts = outcome_trace_of(
        "complete_stream", "before_headers_abort_source_client", "upstream_500"
    )
    with pytest.raises(AssertionError) as caught:
        Expectation(counts).call_count(3)
    assert (
        "expected 3 captured call(s), got 2 "
        "(1 cancelled not counted; pass include_cancelled=True)" in str(caught.value)
    )
    assert "llmentary session outcome-counts" in str(caught.value)
    with pytest.raises(AssertionError, match="at least 3 captured call"):
        Expectation(counts).call_count(min=3)


def test_call_count_exclude_aborted_still_counts_committed_subset() -> None:
    both = outcome_trace_of(
        "complete_stream",
        "after_headers_abort_mid_stream",
        "aborted_with_upstream_error",
        "upstream_500",
    )
    Expectation(both).call_count(3)
    Expectation(both).call_count(2, exclude_aborted=True)
    Expectation(both).call_count(4, include_cancelled=True)
    with pytest.raises(AssertionError, match="committed call"):
        Expectation(both).call_count(3, exclude_aborted=True)


def test_call_count_include_cancelled_and_exclude_aborted_conflict() -> None:
    counts = outcome_trace_of("complete_stream", "before_headers_abort_source_client")
    with pytest.raises(TypeError, match="include_cancelled=True"):
        Expectation(counts).call_count(2, exclude_aborted=True, include_cancelled=True)
    with pytest.raises(TypeError, match="conflict"):
        Expectation(counts).call_count(min=1, max=2, exclude_aborted=True, include_cancelled=True)
    Expectation(counts).call_count(1, exclude_aborted=True, include_cancelled=False)
    Expectation(counts).call_count(2, include_cancelled=True)


def test_call_count_includes_in_flight_calls_by_default() -> None:
    inflight = SessionTrace.from_payload(
        "inflight-counts",
        {
            "id": "inflight-counts",
            "calls": [
                {
                    "seq": 1,
                    "id": "call_1",
                    "started_at": "2026-10-26T10:00:00Z",
                    "http_status": 200,
                    "client_aborted": False,
                    "normalized": {"finish_reason": "stop"},
                }
            ],
        },
    )
    Expectation(inflight).call_count(1)
    Expectation(inflight).call_count(1, include_cancelled=True)
