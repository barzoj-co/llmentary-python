from __future__ import annotations

import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any, Optional

import httpx
import pytest
from fake_upstream import FakeOpenAIServer

from llmentary import Session, SessionNotFound
from llmentary.client import new_session_id, start_proxy
from llmentary.types import zero_calls_message

CHAT_BODY: dict[str, Any] = {
    "model": "fake-model",
    "messages": [{"role": "user", "content": "ping"}],
}


@contextmanager
def isolated_proxy(
    *,
    upstream_url: Optional[str] = None,
    upstream_name: str = "openai",
    extra_env: Optional[Mapping[str, str]] = None,
) -> Iterator[Any]:
    env = {
        "LLMENTARY_UPSTREAM_BASE_URL": "",
        "LLMENTARY_UPSTREAM_NAME": upstream_name,
        "LLMENTARY_PROXY_URL": "",
        "LLMENTARY_PORT": "",
        "LLMENTARY_CONTROL_PORT": "",
        "LLMENTARY_START_COMMAND": "",
    }
    if extra_env:
        env.update(extra_env)
    with start_proxy(upstream_url=upstream_url, env=env) as handle:
        yield handle


def _post_chat(url: str, path: str, headers: Optional[Mapping[str, str]] = None) -> httpx.Response:
    with httpx.Client(base_url=url, headers=dict(headers or {}), timeout=10) as client:
        return client.post(path, json=CHAT_BODY)


@pytest.mark.integration
def test_list_sessions_and_session_not_found_against_real_proxy(llmentary: Session) -> None:
    response = _post_chat(llmentary.openai_base_url, "/chat/completions", llmentary.headers)
    response.raise_for_status()
    trace = llmentary.wait(timeout=5, quiet=0.1)
    assert trace.calls, trace.trace_summary()

    listed = llmentary.list_sessions()
    ids = {item.id for item in listed}
    assert llmentary.session_id in ids, listed
    mine = next(item for item in listed if item.id == llmentary.session_id)
    if mine.spec is not None:
        assert (
            mine.spec.checked + mine.spec.request_only + mine.spec.not_checked
            == mine.calls
        )

    with pytest.raises(SessionNotFound, match="definitely-missing-dog2") as caught:
        llmentary.proxy.control.get_session("definitely-missing-dog2")
    assert caught.value.session_id == "definitely-missing-dog2"
    assert llmentary.session_id in caught.value.known_ids or llmentary.session_id in str(
        caught.value
    )


@pytest.mark.integration
def test_zero_calls_expect_includes_session_and_known_ids(llmentary: Session) -> None:
    response = _post_chat(llmentary.openai_base_url, "/chat/completions", llmentary.headers)
    response.raise_for_status()
    llmentary.wait(timeout=5, quiet=0.1)

    empty = Session(
        session_id="empty-dog2-zero",
        proxy=llmentary.proxy,
        wait_timeout=2,
        quiet_for=0.1,
    )
    with pytest.raises(
        AssertionError, match="no calls recorded for session empty-dog2-zero"
    ) as caught:
        empty.expect()
    message = str(caught.value)
    assert llmentary.session_id in message
    assert "known sessions" in message
    assert zero_calls_message("empty-dog2-zero", [llmentary.session_id]).split(";")[0] in message


@pytest.mark.integration
def test_wait_never_created_min_calls_zero_is_session_not_found(
    llmentary: Session,
) -> None:
    session_id = new_session_id("ghost-dog17-zero")
    ghost = Session(
        session_id=session_id,
        proxy=llmentary.proxy,
        wait_timeout=3,
        quiet_for=0.2,
    )
    with pytest.raises(SessionNotFound, match=session_id) as caught:
        ghost.wait(timeout=3, quiet=0.2, min_calls=0)
    assert caught.value.session_id == session_id


@pytest.mark.integration
def test_wait_min_calls_against_real_proxy(llmentary: Session) -> None:
    def later_calls() -> None:
        time.sleep(1.0)
        first = _post_chat(llmentary.openai_base_url, "/chat/completions", llmentary.headers)
        first.raise_for_status()
        second = _post_chat(llmentary.openai_base_url, "/chat/completions", llmentary.headers)
        second.raise_for_status()

    worker = threading.Thread(target=later_calls)
    worker.start()
    try:
        trace = llmentary.wait(timeout=8, quiet=0.1, min_calls=2)
    finally:
        worker.join(timeout=10)
    assert len(trace.calls) == 2, trace.trace_summary()


@pytest.mark.integration
def test_proxy_config_includes_version() -> None:
    with isolated_proxy() as handle:
        config = handle.control.config()
        typed = handle.control.proxy_config()
        version = handle.control.version()
        assert version
        blob = str(config)
        assert "version" in config or "openai" in blob
        assert "api.openai.com" in blob
        assert "/v1" in blob
        assert typed.version == version
        assert typed.raw == config


# Shared table used by the Go and JS clients as well (DOG-5 C1).
UPSTREAM_JOIN_CASES = (
    ("{echo}/api/v1", "openrouter", "/v1/chat/completions", "/api/v1/chat/completions"),
    ("{echo}/api/v1", "openrouter", "/chat/completions", "/api/v1/chat/completions"),
    ("{echo}/api/v1", "openrouter", "/u/openrouter/chat/completions", "/api/v1/chat/completions"),
    ("{echo}/v1", "openai", "/v1/chat/completions", "/v1/chat/completions"),
    ("{echo}", "openai", "/v1/chat/completions", "/v1/chat/completions"),
    ("{echo}/openai/v1", "openai", "/v1/chat/completions", "/openai/v1/chat/completions"),
)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("upstream_suffix", "upstream_name", "client_path", "expected_path"),
    UPSTREAM_JOIN_CASES,
)
def test_upstream_join_table_against_real_proxy(
    upstream_suffix: str,
    upstream_name: str,
    client_path: str,
    expected_path: str,
) -> None:
    server = FakeOpenAIServer().start()
    try:
        upstream = upstream_suffix.format(echo=server.url)
        with isolated_proxy(upstream_url=upstream, upstream_name=upstream_name) as handle:
            response = _post_chat(handle.url, client_path)
            response.raise_for_status()
            assert server.received_paths, "upstream received no request"
            assert server.received_paths[-1] == expected_path, server.received_paths
    finally:
        server.close()


@pytest.mark.integration
def test_default_upstream_starts_without_setting() -> None:
    with isolated_proxy() as handle:
        config = handle.control.config()
        blob = str(config).lower()
        assert "api.openai.com" in blob
        assert "v1" in blob
