from __future__ import annotations

import os
import threading
import time

import pytest
from e2e_helpers import (
    CHEAP_MODEL,
    WEATHER_CHAT,
    assemble_chat_text,
    assert_plain_text_recorded,
    assert_streamed,
    call_openai,
    openai_client,
    require_e2e,
)

from llmentary import Session
from llmentary.client import new_session_id, start_proxy

pytestmark = pytest.mark.e2e


def test_wait_min_calls_async(llmentary: Session) -> None:
    client = openai_client(llmentary)
    thread_error: list[BaseException] = []

    def later_calls() -> None:
        try:
            time.sleep(1.5)
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
            list(stream)
            call_openai(
                lambda: client.responses.create(
                    model=CHEAP_MODEL,
                    input="Reply with the single word ping.",
                    temperature=0,
                    max_output_tokens=16,
                )
            )
        except Exception as exc:
            thread_error.append(exc)

    worker = threading.Thread(target=later_calls)
    worker.start()
    try:
        trace = llmentary.wait(timeout=30, quiet=0.5, min_calls=2)
    finally:
        worker.join(timeout=40)
    assert not thread_error, thread_error[0]
    assert len(trace.calls) == 2, trace.trace_summary()
    assert all(call.usage for call in trace.calls), trace.trace_summary()
    assert all((call.http_status or 0) == 200 for call in trace.calls), trace.trace_summary()


def test_default_upstream_no_explicit_setting() -> None:
    require_e2e()
    api_key = os.environ["OPENAI_API_KEY"]
    with start_proxy(
        upstream_url=None,
        upstream_api_key=api_key,
        env={
            "LLMENTARY_UPSTREAM_BASE_URL": "",
            "LLMENTARY_PROXY_URL": "",
            "LLMENTARY_START_COMMAND": "",
        },
    ) as handle:
        session = Session(session_id=new_session_id("default-up"), proxy=handle)
        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=api_key,
                base_url=session.openai_base_url,
                default_headers=session.headers,
                max_retries=0,
            )
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
            trace = session.wait(timeout=30, quiet=0.5)
            assert_streamed(trace)
            assert assembled
            assert trace.calls[0].usage, trace.trace_summary()
            assert (trace.calls[0].http_status or 0) == 200, trace.trace_summary()
        finally:
            session.cleanup()


def test_no_spec_violations_on_streaming_tool(llmentary: Session) -> None:
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
            stream_options={"include_usage": True},
        )
    )
    list(stream)
    expect = llmentary.query(api_type="chat_completions").expect()
    expect.tool_called("get_weather").no_spec_violations()


def test_unknown_nested_field_fails_no_spec_violations(llmentary: Session) -> None:
    import httpx

    # Send the extra field with raw JSON so the typed OpenAI SDK cannot strip it.
    with httpx.Client(
        base_url=llmentary.openai_base_url,
        headers={**llmentary.headers, "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        timeout=30,
    ) as client:
        response = client.post(
            "/chat/completions",
            json={
                "model": CHEAP_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Reply with the single word ping.",
                                "ltl_unknown_nested": "dog3-l2",
                            }
                        ],
                    }
                ],
                "temperature": 0,
                "max_tokens": 8,
            },
        )
    assert response.status_code == 200, f"expected 200 from OpenAI, got {response.status_code}"
    expect = llmentary.query(api_type="chat_completions").expect()
    assert_plain_text_recorded(expect)
    with pytest.raises(AssertionError) as caught:
        expect.no_spec_violations()
    message = str(caught.value)
    assert "validation was not run" not in message
    assert " — " in message
    assert "ltl_unknown_nested" in message or "/ltl_unknown" in message
