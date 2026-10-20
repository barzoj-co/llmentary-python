from __future__ import annotations

import pytest

from llmentary import answer_mentions_any
from llmentary.assertions import Expectation, SoftExpectation
from llmentary.types import SessionTrace


def golden_trace() -> SessionTrace:
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
                            "tools": [
                                {
                                    "type": "function",
                                    "function": {"name": "get_weather"},
                                }
                            ],
                        }
                    },
                    "response": {
                        "body": {
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
                                                    "arguments": '{"city": "Paris"}',
                                                },
                                            }
                                        ]
                                    },
                                }
                            ]
                        }
                    },
                },
                {
                    "api_type": "chat_completions",
                    "request": {
                        "body": {
                            "model": "gpt-4.1-mini",
                            "messages": [
                                {
                                    "role": "tool",
                                    "tool_call_id": "call_1",
                                    "content": "sunny",
                                }
                            ],
                        }
                    },
                    "response": {
                        "body": {
                            "choices": [
                                {
                                    "finish_reason": "stop",
                                    "message": {"content": "It is sunny."},
                                }
                            ]
                        }
                    },
                },
            ]
        },
    )


@pytest.mark.parametrize(
    ("text", "candidates"),
    [
        ("The reading is 20°C today.", ("20 °C",)),
        ("The reading is 20 °C today.", ("20°C",)),
        ("The reading is 20 degrees today.", ("20°C",)),
        ("The reading is 20°C today.", ("20 degrees",)),
        ("Risk looks Moderate this week.", ("moderate",)),
        ("Risk looks moderate this week.", ("Moderate",)),
    ],
)
def test_answer_mentions_any_table(text: str, candidates: tuple[str, ...]) -> None:
    answer_mentions_any(text, *candidates)


def test_answer_mentions_any_requires_token_boundaries() -> None:
    with pytest.raises(AssertionError):
        answer_mentions_any("temperature 120 today", "20")
    with pytest.raises(AssertionError):
        answer_mentions_any("unsunny skies", "sunny")
    answer_mentions_any("It is sunny and 20 degrees.", "20", "sunny")


def test_answer_mentions_any_failure_lists_candidates_and_excerpt() -> None:
    text = "x" * 80 + " no numbers here " + "y" * 80
    with pytest.raises(AssertionError) as caught:
        answer_mentions_any(text, "20°C", "moderate")
    message = str(caught.value)
    assert "20°C" in message
    assert "moderate" in message
    assert "excerpt=" in message
    excerpt_start = message.index("excerpt=") + len("excerpt=")
    excerpt = message[excerpt_start:]
    assert len(excerpt) <= 220


def test_final_response_mentions_any_and_tool_result_in_prompt() -> None:
    Expectation(golden_trace()).tool_result_in_prompt("get_weather").final_response_mentions_any(
        "sunny"
    )


def test_tool_result_in_prompt_and_mentions_fail_on_mutated_trace() -> None:
    payload = {
        "calls": [
            {
                "api_type": "chat_completions",
                "request": {
                    "messages": [{"role": "user", "content": "weather"}],
                    "tools": [{"type": "function", "function": {"name": "get_weather"}}],
                },
                "response": {
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
                },
            },
            {
                "api_type": "chat_completions",
                "request": {
                    "messages": [
                        {"role": "user", "content": "weather"},
                        {"role": "assistant", "content": "ok"},
                    ]
                },
                "response": {"choices": [{"message": {"content": "cloudy"}}]},
            },
        ]
    }
    mutated = SessionTrace.from_payload("mutated", payload)
    with pytest.raises(AssertionError, match="result content"):
        Expectation(mutated).tool_result_in_prompt("get_weather")
    with pytest.raises(AssertionError, match="mention any"):
        Expectation(mutated).final_response_mentions_any("sunny")


def test_tool_result_in_prompt_accepts_result_outside_prompt_text() -> None:
    trace = SessionTrace.from_payload(
        "normalized-result",
        {
            "calls": [
                {
                    "response": {
                        "output": [
                            {
                                "type": "function_call",
                                "id": "fc_item",
                                "call_id": "call_weather",
                                "name": "get_weather",
                                "arguments": "{}",
                            }
                        ]
                    }
                },
                {
                    "request": {"messages": [{"role": "user", "content": "continue"}]},
                    "normalized": {
                        "request": {
                            "input": [
                                {
                                    "type": "function_call_output",
                                    "call_id": "call_weather",
                                    "output": "sunny",
                                }
                            ]
                        }
                    },
                },
            ]
        },
    )
    assert "sunny" not in trace.calls[1].prompt_text
    assert trace.calls[0].tool_calls[0].call_id == "call_weather"
    Expectation(trace).tool_result_in_prompt("get_weather")


def test_final_response_mentions_any_ignores_response_envelope() -> None:
    envelope_only = SessionTrace.from_payload(
        "envelope",
        {
            "calls": [
                {
                    "response": {
                        "id": "chatcmpl-20",
                        "created": 20,
                        "usage": {"total_tokens": 20},
                        "choices": [
                            {
                                "message": {
                                    "content": None,
                                    "tool_calls": [
                                        {
                                            "id": "call_1",
                                            "function": {"name": "get_weather", "arguments": "{}"},
                                        }
                                    ],
                                }
                            }
                        ],
                    }
                }
            ]
        },
    )
    with pytest.raises(AssertionError, match="model output text"):
        Expectation(envelope_only).final_response_mentions_any("20")


def test_soft_expectation_reports_three_failures() -> None:
    expectation = SoftExpectation(golden_trace())
    expectation.tool_called("missing_a").tool_called("missing_b").tool_called("missing_c")
    with pytest.raises(AssertionError, match="3 soft assertion") as caught:
        expectation.assert_all()
    message = str(caught.value)
    assert "missing_a" in message
    assert "missing_b" in message
    assert "missing_c" in message
