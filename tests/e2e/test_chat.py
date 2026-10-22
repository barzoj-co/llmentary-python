from __future__ import annotations

import json
import time

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
    openai_invalid_key_errors,
    openai_invalid_model_errors,
    recorded_client_aborted,
    tool_argument_json,
)

from llmentary import Session

pytestmark = pytest.mark.e2e


def test_chat_nonstreaming_plain_text(llmentary: Session) -> None:
    client = openai_client(llmentary)
    call_openai(
        lambda: client.chat.completions.create(
            model=CHEAP_MODEL,
            messages=[{"role": "user", "content": "Reply with the single word ping."}],
            temperature=0,
            max_tokens=8,
        )
    )
    expect = llmentary.query(api_type="chat_completions", model=CHEAP_MODEL).expect()
    assert_plain_text_recorded(expect)
    expect.finish_reason("stop").prompt_contains("ping")


def test_chat_streaming_plain_text(llmentary: Session) -> None:
    client = openai_client(llmentary)
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
    trace = llmentary.wait()
    assert_streamed(trace)
    expect = llmentary.query(api_type="chat_completions", model=CHEAP_MODEL).expect()
    assert_plain_text_recorded(expect)
    expect.finish_reason("stop")
    call = first_call(trace)
    assert_assembled_matches_recorded(assembled, call.response_text)


def test_wait_defaults_after_openai_streaming(llmentary: Session) -> None:
    client = openai_client(llmentary)
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
    trace = llmentary.wait()
    assert_streamed(trace)
    assert trace.calls, trace.trace_summary()
    expect = llmentary.query(api_type="chat_completions", model=CHEAP_MODEL).expect()
    assert_plain_text_recorded(expect)
    expect.finish_reason("stop")
    call = first_call(trace)
    assert_assembled_matches_recorded(assembled, call.response_text)


def test_chat_forced_tool_then_result(llmentary: Session) -> None:
    client = openai_client(llmentary)
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
    expect = llmentary.query(api_type="chat_completions").expect()
    expect.tool_called("get_weather").args_subset("get_weather", {"city": "Paris"})
    expect.tool_result_fed_back("get_weather").no_errors()
    parsed = json.loads(tool_call.function.arguments)
    assert parsed["city"]
    assert tool_argument_json(expect.trace, "get_weather")["city"]


def test_chat_streaming_forced_tool_arguments_json(llmentary: Session) -> None:
    client = openai_client(llmentary)
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
    trace = llmentary.wait()
    assert_streamed(trace)
    expect = llmentary.query(api_type="chat_completions").expect()
    expect.tool_called("get_weather").args_subset("get_weather", {"city": parsed["city"]})
    expect.no_errors()


def test_chat_parallel_tool_calls(llmentary: Session) -> None:
    client = openai_client(llmentary)
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
    expect = llmentary.query(api_type="chat_completions").expect()
    expect.tool_called("get_weather").tool_called("get_time").parallel_tool_calls().no_errors()


def test_chat_structured_json_schema(llmentary: Session) -> None:
    client = openai_client(llmentary)
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
    expect = llmentary.query(api_type="chat_completions").expect()
    expect.response_json_valid().response_json_schema(JSON_SCHEMA).no_errors()


def test_chat_multi_turn_session_helpers(llmentary: Session) -> None:
    client = openai_client(llmentary)
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
    call_openai(
        lambda: client.responses.create(
            model=CHEAP_MODEL,
            input="Say other API.",
            temperature=0,
            max_output_tokens=16,
        )
    )
    unfiltered = llmentary.wait()
    assert len(unfiltered.calls) == 4, unfiltered.trace_summary()
    prompts = [call.prompt_text for call in unfiltered.calls]
    assert "Say one." in prompts[0]
    assert "Say two." in prompts[1]
    assert "Say three." in prompts[2]
    assert any("Say other API." in text for text in prompts)
    filtered = llmentary.query(api_type="chat_completions", model=CHEAP_MODEL).expect()
    assert len(filtered.trace.calls) == 3
    assert len(unfiltered.calls) > len(filtered.trace.calls)
    assert all(call.api_type == "chat_completions" for call in filtered.trace.calls)
    assert all("Say other API." not in call.prompt_text for call in filtered.trace.calls)
    filtered.prompt_contains("Say two.").no_errors()


def test_chat_upstream_invalid_model(llmentary: Session) -> None:
    client = openai_client(llmentary)
    with pytest.raises(openai_invalid_model_errors()):
        client.chat.completions.create(
            model=INVALID_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            temperature=0,
            max_tokens=8,
        )
    assert_recorded_error(
        llmentary.wait(),
        pattern=r"invalid|not.?found|model",
        http_status=INVALID_MODEL_STATUS,
        error_type=INVALID_MODEL_ERROR_TYPE,
        error_code=INVALID_MODEL_ERROR_CODE,
    )


def test_chat_upstream_invalid_api_key(llmentary: Session) -> None:
    client = openai_client(llmentary)
    with pytest.raises(openai_invalid_key_errors()):
        client.chat.completions.create(
            model=CHEAP_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            temperature=0,
            max_tokens=8,
            extra_headers=INVALID_KEY_HEADERS,
        )
    assert_recorded_error(
        llmentary.wait(),
        pattern=r"auth|api.?key|401|invalid",
        http_status=INVALID_KEY_STATUS,
        error_type=INVALID_KEY_ERROR_TYPE,
        error_code=INVALID_KEY_ERROR_CODE,
    )


def test_chat_cancel_mid_stream_keeps_proxy_healthy(llmentary: Session) -> None:
    client = openai_client(llmentary)
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

    trace = llmentary.wait()
    assert trace.calls, trace.trace_summary()
    assert recorded_client_aborted(trace.calls), (
        "expected normalized.client_aborted after client disconnect\n"
        f"{trace.trace_summary()}"
    )
    assert any(call.client_aborted for call in trace.calls), trace.trace_summary()
    llmentary.proxy.control.health()
    call_openai(
        lambda: client.chat.completions.create(
            model=CHEAP_MODEL,
            messages=[{"role": "user", "content": "Reply with the single word ok."}],
            temperature=0,
            max_tokens=8,
        )
    )
    llmentary.query(api_type="chat_completions").expect().prompt_contains(
        "single word ok"
    ).no_errors()


def test_summary_request_only_for_auth_failure_and_abort(llmentary: Session) -> None:
    client = openai_client(llmentary)
    with pytest.raises(openai_invalid_key_errors()):
        client.chat.completions.create(
            model=CHEAP_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            temperature=0,
            max_tokens=8,
            extra_headers=INVALID_KEY_HEADERS,
        )
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
    trace = llmentary.wait()
    summary = trace.trace_summary()
    assert "http=0" not in summary, summary
    assert "spec: request_only" in summary, summary
    auth = [call for call in trace.calls if call.http_status == INVALID_KEY_STATUS]
    assert auth, summary
    assert auth[0].spec_status_label() == "request_only"
    aborted = [call for call in trace.calls if call.client_aborted]
    assert aborted, summary
    abort = aborted[0]
    assert abort.http_status != 502
    # Live OpenAI headers arrive before 50ms, so this is after_headers:
    # the request was validated (clean). before_headers abort is request_only
    # with http_status None; unit fixture A2 covers that shape.
    if abort.http_status is None:
        assert abort.spec_status_label() == "request_only", summary
    else:
        assert abort.spec_status_label() in {
            "clean",
            "request_only",
            "not checked",
        } or abort.spec_status_label().startswith("skipped:"), summary
    assert "http=0" not in summary


def test_client_aborted_typed_accessors(llmentary: Session) -> None:
    client = openai_client(llmentary)
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
        time.sleep(0.05)
        stream.close()
    except Exception:
        pass
    trace = llmentary.wait()
    aborted = [call for call in trace.calls if call.client_aborted]
    assert aborted, trace.trace_summary()
    call = aborted[0]
    assert call.client_aborted is True
    # Proxy 0.3.0 still records finish_reason=unknown on abort; DOG-27
    # proxy PR #39 sets cancelled. Accept both until that ships.
    assert call.finish_reason in {"cancelled", "unknown"}, call.finish_reason
    assert call.http_status != 502
    assert call.abort_phase in (None, "before_headers", "after_headers")
    llmentary.expect().aborted()


def test_before_headers_timeout_is_status_aborted(llmentary: Session) -> None:
    from httpx import Timeout
    from openai import APIConnectionError, APITimeoutError

    # Connect/write stay generous so the proxy can forward the request.
    # A short read timeout aborts before response headers (DOG-35 A4).
    client = openai_client(
        llmentary, timeout=Timeout(5.0, connect=5.0, write=5.0, read=0.05, pool=5.0)
    )
    with pytest.raises((APITimeoutError, APIConnectionError, TimeoutError)):
        client.chat.completions.create(
            model=CHEAP_MODEL,
            messages=[{"role": "user", "content": "Count slowly from 1 to 80 in words."}],
            temperature=0,
            max_tokens=400,
            stream=True,
        )
    trace = llmentary.wait()
    aborted = [call for call in trace.calls if call.client_aborted]
    assert aborted, trace.trace_summary()
    call = aborted[0]
    assert call.upstream_called is True, trace.trace_summary()
    assert call.abort_phase == "before_headers", trace.trace_summary()
    assert "status=aborted" in trace.trace_summary()
    assert "http=0" not in trace.trace_summary()


def test_after_first_chunk_close_is_status_aborted(llmentary: Session) -> None:
    client = openai_client(llmentary)
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
    trace = llmentary.wait()
    aborted = [call for call in trace.calls if call.client_aborted]
    assert aborted, trace.trace_summary()
    call = aborted[0]
    assert call.upstream_called is True, trace.trace_summary()
    assert call.abort_phase == "after_headers", trace.trace_summary()
    assert "status=aborted" in trace.trace_summary()
    assert "http=200" in trace.trace_summary()


def test_after_headers_abort_response_text_is_partial_not_json(llmentary: Session) -> None:
    client = openai_client(llmentary)
    assembled = []
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
                assembled.append(delta.content)
                break
    finally:
        stream.close()
    trace = llmentary.wait()
    aborted = [call for call in trace.calls if call.client_aborted]
    assert aborted, trace.trace_summary()
    call = aborted[0]
    text = call.response_text
    assert text != "{}"
    assert not text.lstrip().startswith("{")
    assert '"choices"' not in text
    assert '"delta"' not in text
    assert text
    if assembled:
        assert text.startswith(assembled[0]) or assembled[0].startswith(text)
    llmentary.expect().call_count(1, exclude_aborted=False)
    llmentary.query().aborted().expect().call_count(1)
    listed = llmentary.list_sessions()
    mine = next(row for row in listed if row.id == llmentary.session_id)
    if mine.spec is not None:
        assert mine.spec.checked + mine.spec.request_only + mine.spec.not_checked == mine.calls
