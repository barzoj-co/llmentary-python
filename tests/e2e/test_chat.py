from __future__ import annotations

import json

import pytest
from helpers import (
    CHEAP_MODEL,
    INVALID_KEY_HEADERS,
    INVALID_MODEL,
    JSON_SCHEMA,
    PARALLEL_MODEL,
    TIME_CHAT,
    WEATHER_CHAT,
    assemble_chat_text,
    assemble_chat_tool_arguments,
    assert_assembled_matches_recorded,
    assert_plain_text_recorded,
    assert_recorded_error,
    assert_streamed,
    call_openai,
    first_call,
    openai_client,
    openai_client_errors,
    recorded_incomplete_or_errored,
    tool_argument_json,
)

from llmtestlab import LabSession

pytestmark = pytest.mark.e2e


def test_chat_nonstreaming_plain_text(lab: LabSession) -> None:
    client = openai_client(lab)
    call_openai(
        lambda: client.chat.completions.create(
            model=CHEAP_MODEL,
            messages=[{"role": "user", "content": "Reply with the single word ping."}],
            temperature=0,
            max_tokens=8,
        )
    )
    expect = lab.query(api_type="chat_completions", model=CHEAP_MODEL).expect()
    assert_plain_text_recorded(expect)
    expect.finish_reason("stop").prompt_contains("ping")


def test_chat_streaming_plain_text(lab: LabSession) -> None:
    client = openai_client(lab)
    stream = call_openai(
        lambda: client.chat.completions.create(
            model=CHEAP_MODEL,
            messages=[{"role": "user", "content": "Reply with the single word ping."}],
            temperature=0,
            max_tokens=8,
            stream=True,
            stream_options={"include_usage": True},
        )
    )
    assembled = assemble_chat_text(stream)
    trace = lab.wait()
    assert_streamed(trace)
    expect = lab.query(api_type="chat_completions", model=CHEAP_MODEL).expect()
    assert_plain_text_recorded(expect)
    expect.finish_reason("stop")
    call = first_call(trace)
    assert_assembled_matches_recorded(assembled, call.response_text)


def test_chat_forced_tool_then_result(lab: LabSession) -> None:
    client = openai_client(lab)
    first = call_openai(
        lambda: client.chat.completions.create(
            model=CHEAP_MODEL,
            messages=[{"role": "user", "content": "Use the tool for weather in Paris."}],
            tools=[WEATHER_CHAT],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
            temperature=0,
            max_tokens=32,
        )
    )
    tool_call = first.choices[0].message.tool_calls[0]
    call_openai(
        lambda: client.chat.completions.create(
            model=CHEAP_MODEL,
            messages=[
                {"role": "user", "content": "Use the tool for weather in Paris."},
                first.choices[0].message,
                {"role": "tool", "tool_call_id": tool_call.id, "content": '{"weather":"sunny"}'},
            ],
            temperature=0,
            max_tokens=16,
        )
    )
    expect = lab.query(api_type="chat_completions").expect()
    expect.tool_called("get_weather").args_subset("get_weather", {"city": "Paris"})
    expect.tool_result_fed_back("get_weather").no_errors()
    parsed = json.loads(tool_call.function.arguments)
    assert parsed["city"]
    assert tool_argument_json(expect.trace, "get_weather")["city"]


def test_chat_streaming_forced_tool_arguments_json(lab: LabSession) -> None:
    client = openai_client(lab)
    stream = call_openai(
        lambda: client.chat.completions.create(
            model=CHEAP_MODEL,
            messages=[{"role": "user", "content": "Use the tool for weather in Paris."}],
            tools=[WEATHER_CHAT],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
            temperature=0,
            max_tokens=32,
            stream=True,
        )
    )
    assembled = assemble_chat_tool_arguments(stream)
    parsed = json.loads(assembled)
    assert isinstance(parsed, dict)
    trace = lab.wait()
    assert_streamed(trace)
    expect = lab.query(api_type="chat_completions").expect()
    expect.tool_called("get_weather").args_subset("get_weather", {"city": parsed["city"]})
    expect.no_errors()


def test_chat_parallel_tool_calls(lab: LabSession) -> None:
    client = openai_client(lab)
    call_openai(
        lambda: client.chat.completions.create(
            model=PARALLEL_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Call get_weather and get_time in the same turn. Never reply with text."
                    ),
                },
                {"role": "user", "content": "Paris"},
            ],
            tools=[WEATHER_CHAT, TIME_CHAT],
            tool_choice="required",
            parallel_tool_calls=True,
            temperature=0,
            max_tokens=64,
        )
    )
    expect = lab.query(api_type="chat_completions").expect()
    expect.tool_called("get_weather").tool_called("get_time").parallel_tool_calls().no_errors()


def test_chat_structured_json_schema(lab: LabSession) -> None:
    client = openai_client(lab)
    call_openai(
        lambda: client.chat.completions.create(
            model=CHEAP_MODEL,
            messages=[{"role": "user", "content": "Return city Paris and ok true."}],
            temperature=0,
            max_tokens=32,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "city_flag",
                    "strict": True,
                    "schema": JSON_SCHEMA,
                },
            },
        )
    )
    expect = lab.query(api_type="chat_completions").expect()
    expect.response_json_valid().response_json_schema(JSON_SCHEMA).no_errors()


def test_chat_multi_turn_session_helpers(lab: LabSession) -> None:
    client = openai_client(lab)
    call_openai(
        lambda: client.chat.completions.create(
            model=CHEAP_MODEL,
            messages=[{"role": "user", "content": "Say one."}],
            temperature=0,
            max_tokens=8,
        )
    )
    call_openai(
        lambda: client.chat.completions.create(
            model=CHEAP_MODEL,
            messages=[
                {"role": "user", "content": "Say one."},
                {"role": "assistant", "content": "one"},
                {"role": "user", "content": "Say two."},
            ],
            temperature=0,
            max_tokens=8,
        )
    )
    call_openai(
        lambda: client.chat.completions.create(
            model=CHEAP_MODEL,
            messages=[{"role": "user", "content": "Say three."}],
            temperature=0,
            max_tokens=8,
        )
    )
    trace = lab.wait()
    assert len(trace.calls) == 3, trace.trace_summary()
    prompts = [call.prompt_text for call in trace.calls]
    assert "Say one." in prompts[0]
    assert "Say two." in prompts[1]
    assert "Say three." in prompts[2]
    filtered = lab.query(api_type="chat_completions", model=CHEAP_MODEL).expect()
    assert len(filtered.trace.calls) == 3
    filtered.prompt_contains("Say two.").no_errors()


def test_chat_upstream_invalid_model(lab: LabSession) -> None:
    client = openai_client(lab)
    with pytest.raises(openai_client_errors()):
        client.chat.completions.create(
            model=INVALID_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            temperature=0,
            max_tokens=8,
        )
    call = assert_recorded_error(lab.wait(), pattern=r"invalid|not.?found|model")
    assert call.http_status is not None and call.http_status >= 400
    assert call.error_type or call.error_code


def test_chat_upstream_invalid_api_key(lab: LabSession) -> None:
    client = openai_client(lab)
    with pytest.raises(openai_client_errors()):
        client.chat.completions.create(
            model=CHEAP_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            temperature=0,
            max_tokens=8,
            extra_headers=INVALID_KEY_HEADERS,
        )
    call = assert_recorded_error(lab.wait(), pattern=r"auth|api.?key|401|invalid")
    assert call.http_status == 401 or (call.http_status or 0) >= 400


def test_chat_cancel_mid_stream_keeps_proxy_healthy(lab: LabSession) -> None:
    client = openai_client(lab)
    stream = call_openai(
        lambda: client.chat.completions.create(
            model=CHEAP_MODEL,
            messages=[{"role": "user", "content": "Count slowly from 1 to 80 in words."}],
            temperature=0,
            max_tokens=400,
            stream=True,
        )
    )
    try:
        for event in stream:
            delta = event.choices[0].delta if event.choices else None
            if delta is not None and delta.content:
                break
    finally:
        stream.close()

    trace = lab.wait()
    assert trace.calls, trace.trace_summary()
    assert recorded_incomplete_or_errored(trace.calls), trace.trace_summary()
    lab.proxy.control.health()
    call_openai(
        lambda: client.chat.completions.create(
            model=CHEAP_MODEL,
            messages=[{"role": "user", "content": "Reply with the single word ok."}],
            temperature=0,
            max_tokens=8,
        )
    )
    lab.query(api_type="chat_completions").expect().prompt_contains("single word ok").no_errors()
