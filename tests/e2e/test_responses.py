from __future__ import annotations

import json

import pytest
from e2e_helpers import (
    CHEAP_MODEL,
    INVALID_KEY_ERROR_CODE,
    INVALID_KEY_ERROR_TYPE,
    INVALID_KEY_HEADERS,
    INVALID_KEY_STATUS,
    INVALID_MODEL,
    INVALID_MODEL_ERROR_CODE,
    INVALID_MODEL_ERROR_TYPE,
    INVALID_MODEL_STATUS,
    JSON_SCHEMA,
    PARALLEL_MODEL,
    TIME_RESPONSES,
    WEATHER_RESPONSES,
    assemble_responses_text,
    assemble_responses_tool_arguments,
    assert_assembled_matches_recorded,
    assert_plain_text_recorded,
    assert_recorded_error,
    assert_streamed,
    call_openai,
    first_call,
    openai_client,
    openai_invalid_key_errors,
    openai_invalid_model_errors,
    recorded_client_aborted,
    responses_function_call,
    tool_argument_json,
)

from llmentary import Session

pytestmark = pytest.mark.e2e


def test_responses_nonstreaming_plain_text(llmentary: Session) -> None:
    client = openai_client(llmentary)
    call_openai(
        lambda: client.responses.create(
            model=CHEAP_MODEL,
            input="Reply with the single word ping.",
            temperature=0,
            max_output_tokens=16,
        )
    )
    expect = llmentary.query(api_type="responses", model=CHEAP_MODEL).expect()
    assert_plain_text_recorded(expect)
    expect.prompt_contains("ping")


def test_responses_streaming_plain_text(llmentary: Session) -> None:
    client = openai_client(llmentary)
    stream = call_openai(
        lambda: client.responses.create(
            model=CHEAP_MODEL,
            input="Reply with the single word ping.",
            temperature=0,
            max_output_tokens=16,
            stream=True,
        )
    )
    assembled = assemble_responses_text(stream)
    trace = llmentary.wait()
    assert_streamed(trace)
    expect = llmentary.query(api_type="responses", model=CHEAP_MODEL).expect()
    assert_plain_text_recorded(expect)
    call = first_call(trace)
    assert_assembled_matches_recorded(assembled, call.response_text)


def test_responses_forced_tool_then_result(llmentary: Session) -> None:
    client = openai_client(llmentary)
    first = call_openai(
        lambda: client.responses.create(
            model=CHEAP_MODEL,
            input="Use the tool for weather in Paris.",
            tools=[WEATHER_RESPONSES],
            tool_choice={"type": "function", "name": "get_weather"},
            temperature=0,
            max_output_tokens=32,
        )
    )
    function_call = responses_function_call(first)
    call_openai(
        lambda: client.responses.create(
            model=CHEAP_MODEL,
            input=[
                function_call,
                {
                    "type": "function_call_output",
                    "call_id": function_call.call_id,
                    "output": '{"weather":"sunny"}',
                },
            ],
            temperature=0,
            max_output_tokens=16,
        )
    )
    expect = llmentary.query(api_type="responses").expect()
    expect.tool_called("get_weather").args_subset("get_weather", {"city": "Paris"})
    expect.tool_result_fed_back("get_weather").no_errors()
    parsed = json.loads(function_call.arguments)
    assert parsed["city"]
    assert tool_argument_json(expect.trace, "get_weather")["city"]


def test_responses_streaming_forced_tool_arguments_json(llmentary: Session) -> None:
    client = openai_client(llmentary)
    stream = call_openai(
        lambda: client.responses.create(
            model=CHEAP_MODEL,
            input="Use the tool for weather in Paris.",
            tools=[WEATHER_RESPONSES],
            tool_choice={"type": "function", "name": "get_weather"},
            temperature=0,
            max_output_tokens=32,
            stream=True,
        )
    )
    assembled = assemble_responses_tool_arguments(stream)
    assert assembled, "expected streamed function-call argument deltas"
    parsed = json.loads(assembled)
    assert isinstance(parsed, dict)
    trace = llmentary.wait()
    assert_streamed(trace)
    expect = llmentary.query(api_type="responses").expect()
    expect.tool_called("get_weather").args_subset("get_weather", {"city": parsed["city"]})
    expect.no_errors()


def test_responses_parallel_tool_calls(llmentary: Session) -> None:
    client = openai_client(llmentary)
    call_openai(
        lambda: client.responses.create(
            model=PARALLEL_MODEL,
            input=(
                "Call get_weather and get_time for Paris in the same turn. Never reply with text."
            ),
            tools=[WEATHER_RESPONSES, TIME_RESPONSES],
            tool_choice="required",
            parallel_tool_calls=True,
            temperature=0,
            max_output_tokens=64,
        )
    )
    expect = llmentary.query(api_type="responses").expect()
    expect.tool_called("get_weather").tool_called("get_time").parallel_tool_calls().no_errors()


def test_responses_structured_json_schema(llmentary: Session) -> None:
    client = openai_client(llmentary)
    call_openai(
        lambda: client.responses.create(
            model=CHEAP_MODEL,
            input="Return city Paris and ok true.",
            temperature=0,
            max_output_tokens=32,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "city_flag",
                    "strict": True,
                    "schema": JSON_SCHEMA,
                }
            },
        )
    )
    expect = llmentary.query(api_type="responses").expect()
    expect.response_json_valid().response_json_schema(JSON_SCHEMA).no_errors()


def test_responses_multi_turn_session_helpers(llmentary: Session) -> None:
    client = openai_client(llmentary)
    call_openai(
        lambda: client.responses.create(
            model=CHEAP_MODEL,
            input="Say one.",
            temperature=0,
            max_output_tokens=16,
        )
    )
    call_openai(
        lambda: client.responses.create(
            model=CHEAP_MODEL,
            input="Say two.",
            temperature=0,
            max_output_tokens=16,
        )
    )
    call_openai(
        lambda: client.responses.create(
            model=CHEAP_MODEL,
            input="Say three.",
            temperature=0,
            max_output_tokens=16,
        )
    )
    call_openai(
        lambda: client.chat.completions.create(
            model=CHEAP_MODEL,
            messages=[{"role": "user", "content": "Say other API."}],
            temperature=0,
            max_tokens=8,
        )
    )
    unfiltered = llmentary.wait()
    assert len(unfiltered.calls) == 4, unfiltered.trace_summary()
    prompts = [call.prompt_text for call in unfiltered.calls]
    assert "Say one." in prompts[0]
    assert "Say two." in prompts[1]
    assert "Say three." in prompts[2]
    assert any("Say other API." in text for text in prompts)
    filtered = llmentary.query(api_type="responses", model=CHEAP_MODEL).expect()
    assert len(filtered.trace.calls) == 3
    assert len(unfiltered.calls) > len(filtered.trace.calls)
    assert all(call.api_type == "responses" for call in filtered.trace.calls)
    assert all("Say other API." not in call.prompt_text for call in filtered.trace.calls)
    filtered.prompt_contains("Say two.").no_errors()


def test_responses_upstream_invalid_model(llmentary: Session) -> None:
    client = openai_client(llmentary)
    with pytest.raises(openai_invalid_model_errors()):
        client.responses.create(
            model=INVALID_MODEL,
            input="ping",
            temperature=0,
            max_output_tokens=16,
        )
    assert_recorded_error(
        llmentary.wait(),
        pattern=r"invalid|not.?found|model",
        http_status=INVALID_MODEL_STATUS,
        error_type=INVALID_MODEL_ERROR_TYPE,
        error_code=INVALID_MODEL_ERROR_CODE,
    )


def test_responses_upstream_invalid_api_key(llmentary: Session) -> None:
    client = openai_client(llmentary)
    with pytest.raises(openai_invalid_key_errors()):
        client.responses.create(
            model=CHEAP_MODEL,
            input="ping",
            temperature=0,
            max_output_tokens=16,
            extra_headers=INVALID_KEY_HEADERS,
        )
    assert_recorded_error(
        llmentary.wait(),
        pattern=r"auth|api.?key|401|invalid",
        http_status=INVALID_KEY_STATUS,
        error_type=INVALID_KEY_ERROR_TYPE,
        error_code=INVALID_KEY_ERROR_CODE,
    )


def test_responses_cancel_mid_stream_keeps_proxy_healthy(llmentary: Session) -> None:
    client = openai_client(llmentary)
    stream = call_openai(
        lambda: client.responses.create(
            model=CHEAP_MODEL,
            input="Count slowly from 1 to 80 in words.",
            temperature=0,
            max_output_tokens=400,
            stream=True,
        )
    )
    try:
        for event in stream:
            if getattr(event, "type", "") == "response.output_text.delta":
                delta = getattr(event, "delta", "")
                if delta:
                    break
    finally:
        stream.close()

    trace = llmentary.wait()
    assert trace.calls, trace.trace_summary()
    assert recorded_client_aborted(trace.calls), (
        "expected normalized.client_aborted after client disconnect\n"
        f"{trace.trace_summary()}"
    )
    assert any(call.client_aborted for call in trace.calls), trace.trace_summary()
    llmentary.proxy.control.health()
    call_openai(
        lambda: client.responses.create(
            model=CHEAP_MODEL,
            input="Reply with the single word ok.",
            temperature=0,
            max_output_tokens=16,
        )
    )
    llmentary.query(api_type="responses").expect().prompt_contains("single word ok").no_errors()
