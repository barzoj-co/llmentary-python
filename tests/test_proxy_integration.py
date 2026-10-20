from __future__ import annotations

import time
from urllib.parse import quote

import httpx
import pytest

from llmentary import Session
from llmentary.client import ProxyHandle


@pytest.mark.integration
def test_chat_completions_tool_flow_through_real_proxy(llmentary: Session) -> None:
    with httpx.Client(base_url=llmentary.openai_base_url, headers=llmentary.headers) as client:
        first = client.post(
            "/chat/completions",
            json={
                "model": "fake-model",
                "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                        },
                    }
                ],
            },
        )
        first.raise_for_status()
        tool_call = first.json()["choices"][0]["message"]["tool_calls"][0]
        second = client.post(
            "/chat/completions",
            json={
                "model": "fake-model",
                "messages": [
                    {"role": "user", "content": "What is the weather in Paris?"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": '{"weather":"sunny"}',
                    },
                ],
            },
        )
        second.raise_for_status()

    llmentary.expect().tool_called("get_weather").args_subset(
        "get_weather", {"city": "Paris"}
    ).tool_result_fed_back("get_weather").response_json_valid().response_json_schema(
        {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }
    ).no_errors()


@pytest.mark.integration
def test_responses_tool_flow_through_real_proxy(llmentary: Session) -> None:
    with httpx.Client(base_url=llmentary.openai_base_url, headers=llmentary.headers) as client:
        first = client.post(
            "/responses",
            json={
                "model": "fake-model",
                "input": "What is the weather in Paris?",
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    }
                ],
            },
        )
        first.raise_for_status()
        tool_call = first.json()["output"][0]
        second = client.post(
            "/responses",
            json={
                "model": "fake-model",
                "input": [
                    tool_call,
                    {
                        "type": "function_call_output",
                        "call_id": tool_call["call_id"],
                        "output": '{"weather":"sunny"}',
                    },
                ],
            },
        )
        second.raise_for_status()

    llmentary.query(api_type="responses").expect().tool_called("get_weather").args_subset(
        "get_weather", {"city": "Paris"}
    ).tool_result_fed_back("get_weather").response_json_valid().no_errors()


@pytest.mark.integration
def test_wait_honours_quiet_after_finished_call(llmentary: Session) -> None:
    with httpx.Client(base_url=llmentary.openai_base_url, headers=llmentary.headers) as client:
        response = client.post(
            "/chat/completions",
            json={
                "model": "fake-model",
                "messages": [{"role": "user", "content": "ping"}],
            },
        )
        response.raise_for_status()

    started = time.monotonic()
    trace = llmentary.wait(timeout=5, quiet=0.1)
    elapsed = time.monotonic() - started
    assert trace.calls, trace.trace_summary()
    assert elapsed < 0.4, (
        f"wait with quiet=100ms returned in {elapsed:.3f}s; "
        "the 500ms proxy default would have failed this"
    )


def _chat_ping(client: httpx.Client, text: str) -> None:
    response = client.post(
        "/chat/completions",
        json={"model": "fake-model", "messages": [{"role": "user", "content": text}]},
    )
    response.raise_for_status()


@pytest.mark.integration
def test_mark_since_sees_only_second_turn(llmentary: Session) -> None:
    with httpx.Client(base_url=llmentary.openai_base_url, headers=llmentary.headers) as client:
        _chat_ping(client, "turn-1-a")
        _chat_ping(client, "turn-1-b")
        llmentary.wait()
        mark = llmentary.mark()
        _chat_ping(client, "turn-2-a")
        _chat_ping(client, "turn-2-b")

    scoped = llmentary.expect(since=mark)
    assert len(scoped.trace.calls) == 2, scoped.trace.trace_summary()
    prompts = [call.prompt_text for call in scoped.trace.calls]
    assert all("turn-2" in prompt for prompt in prompts), scoped.trace.trace_summary()
    assert all("turn-1" not in prompt for prompt in prompts), scoped.trace.trace_summary()
    with pytest.raises(AssertionError, match="turn-1"):
        scoped.prompt_contains("turn-1")


@pytest.mark.integration
def test_openai_env_base_url_lands_on_path_session(
    llmentary: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    openai = pytest.importorskip("openai")
    monkeypatch.setenv("OPENAI_BASE_URL", llmentary.path_base_url())
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = openai.OpenAI(max_retries=0)
    client.chat.completions.create(
        model="fake-model",
        messages=[{"role": "user", "content": "env-base-url ping"}],
    )
    llmentary.expect().prompt_contains("env-base-url ping").no_errors()


@pytest.mark.integration
def test_openai_client_built_before_session_uses_path(
    llmentary_proxy: ProxyHandle, monkeypatch: pytest.MonkeyPatch
) -> None:
    openai = pytest.importorskip("openai")
    session_id = "pre-session-path"
    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        f"{llmentary_proxy.url.rstrip('/')}/s/{quote(session_id, safe='')}/v1",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = openai.OpenAI(max_retries=0)
    session = Session(session_id=session_id, proxy=llmentary_proxy)
    try:
        client.chat.completions.create(
            model="fake-model",
            messages=[{"role": "user", "content": "before-session ping"}],
        )
        session.expect().prompt_contains("before-session ping").no_errors()
    finally:
        session.cleanup()
