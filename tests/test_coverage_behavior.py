from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

from llmentary import SessionNotFound
from llmentary.__main__ import main
from llmentary.binary import (
    BinaryResolutionError,
    bundled_proxy_version,
    contract_path,
    ensure_executable,
    find_binary,
    install_kind,
    package_bin_dir,
    sdist_has_bundled_bin,
)
from llmentary.client import (
    ControlClient,
    ProxyHandle,
    Session,
    allocate_listen_and_control_ports,
    coerce_session_row,
    control_error_payload,
    default_control_url,
    free_port,
    new_session_id,
    redact_command,
    redact_secrets,
    start_proxy,
)
from llmentary.doctor import (
    binary_version_output,
    doctor,
    proxy_binary_path,
    proxy_version,
    proxy_version_line,
)


def test_cli_doctor_and_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    binary = tmp_path / "llmentary"
    binary.write_text("#!/bin/sh\necho 'llmentary 0.4.0'\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setenv("LLMENTARY_BIN", str(binary))
    monkeypatch.setenv("PATH", "")
    assert main(["doctor"]) == 0
    assert "binary --version: llmentary 0.4.0" in capsys.readouterr().out
    assert main([]) == 2
    assert main(["nope"]) == 2
    err = capsys.readouterr().err
    assert "usage: python -m llmentary doctor | --version | version" in err


def test_doctor_in_process_and_version_corners(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "llmentary"
    binary.write_text("#!/bin/sh\necho 'llmentary 0.4.0'\n", encoding="utf-8")
    binary.chmod(0o755)
    out = io.StringIO()
    assert doctor(env={"LLMENTARY_BIN": str(binary), "PATH": ""}, out=out) == 0
    assert "binary --version: llmentary 0.4.0" in out.getvalue()
    env = {"LLMENTARY_BIN": str(binary), "PATH": ""}
    assert proxy_version(env=env) == "0.4.0"
    assert proxy_version_line(env=env) == "llmentary 0.4.0"

    assert doctor() in {0, 1}

    monkeypatch.setattr(
        "llmentary.doctor.find_binary",
        lambda env=None: (_ for _ in ()).throw(BinaryResolutionError("no binary")),
    )
    failed = io.StringIO()
    assert doctor(env={"PATH": ""}, out=failed) == 1
    assert "no binary" in failed.getvalue()
    assert proxy_binary_path(env={"PATH": ""}) is None
    assert proxy_version(env={"PATH": ""}) is None
    assert proxy_version_line(env={"PATH": ""}) is None

    assert binary_version_output(None) == "unknown"

    def boom(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("nope")

    monkeypatch.setattr("llmentary.doctor.subprocess.run", boom)
    assert binary_version_output(tmp_path / "missing") == "unknown"

    def timed_out(*_args: Any, **_kwargs: Any) -> None:
        raise subprocess.TimeoutExpired(cmd="llmentary", timeout=5)

    monkeypatch.setattr("llmentary.doctor.subprocess.run", timed_out)
    assert binary_version_output(tmp_path / "slow") == "unknown"

    empty = tmp_path / "empty"
    empty.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    empty.chmod(0o755)
    monkeypatch.undo()
    assert binary_version_output(empty) == "unknown"


def test_binary_missing_file_install_kind_and_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(BinaryResolutionError, match="missing file"):
        find_binary(env={"LLMENTARY_BIN": str(tmp_path / "gone"), "PATH": ""})

    assert package_bin_dir().name == "bin"
    assert install_kind() in {"wheel", "sdist or source tree"}
    assert isinstance(bundled_proxy_version(), str)

    version_dir = tmp_path / "plat"
    version_dir.mkdir()
    bundled = version_dir / "llmentary"
    bundled.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr("llmentary.binary.bundled_binary", lambda: bundled)
    assert bundled_proxy_version() == "unknown"
    (version_dir / "VERSION").write_text("\n", encoding="utf-8")
    assert bundled_proxy_version() == "unknown"
    (version_dir / "VERSION").write_text("0.4.0\n", encoding="utf-8")
    assert bundled_proxy_version() == "0.4.0"

    monkeypatch.setattr(
        "llmentary.binary.find_binary",
        lambda env=None: (_ for _ in ()).throw(BinaryResolutionError("none")),
    )
    assert contract_path(env={"PATH": ""}) is None

    locked = tmp_path / "noexec"
    locked.write_text("#!/bin/sh\n", encoding="utf-8")
    locked.chmod(0o644)
    ensure_executable(locked)
    assert os.access(locked, os.X_OK)

    assert sdist_has_bundled_bin(["src/llmentary/types.py"]) is False
    assert sdist_has_bundled_bin(["src/llmentary/bin/manylinux2014_x86_64/llmentary"]) is True
    assert sdist_has_bundled_bin(["pkg/llmentary/bin"]) is True


def _client(handler: Any) -> ControlClient:
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return ControlClient("http://127.0.0.1:7778", token="tok", client=http)


def test_control_client_health_config_and_list_corners() -> None:
    state = {"health": "ok", "config": "ok", "sessions": "ok"}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/healthz":
            kind = state["health"]
            if kind == "empty":
                return httpx.Response(200, content=b"")
            if kind == "text":
                return httpx.Response(200, text="not-json")
            if kind == "list":
                return httpx.Response(200, json=["ok"])
            return httpx.Response(200, json={"ok": True})
        if path == "/config":
            kind = state["config"]
            if kind == "bad-config":
                return httpx.Response(200, json=["nope"])
            if kind == "no-version":
                return httpx.Response(200, json={"upstreams": {}})
            return httpx.Response(200, json={"version": "0.4.0"})
        if path == "/openapi.yaml":
            return httpx.Response(200, text="- just a list\n")
        if path == "/sessions":
            kind = state["sessions"]
            if kind == "none":
                return httpx.Response(200, json={})
            if kind == "items":
                return httpx.Response(200, json={"items": [{"id": "a"}]})
            if kind == "bad-list":
                return httpx.Response(200, json={"sessions": {"id": "a"}})
            if kind == "not-map":
                return httpx.Response(200, json=["sessions"])
            return httpx.Response(200, json={"sessions": [{"id": "a", "calls": 1}]})
        return httpx.Response(404)

    client = _client(handler)
    assert client.headers == {"Authorization": "Bearer tok"}
    assert client.health() == {"ok": True}
    state["health"] = "empty"
    assert client.health() == {"ok": True}
    state["health"] = "text"
    assert client.health() == {"ok": "not-json"}
    state["health"] = "list"
    assert client.health() == {"ok": ["ok"]}

    state["config"] = "bad-config"
    with pytest.raises(TypeError, match="config payload"):
        client.config()
    state["config"] = "no-version"
    with pytest.raises(TypeError, match="version string"):
        client.version()

    with pytest.raises(TypeError, match=r"openapi\.yaml"):
        client.openapi()

    state["sessions"] = "none"
    assert client.list_sessions() == []
    state["sessions"] = "items"
    assert client.list_sessions()[0].id == "a"
    state["sessions"] = "bad-list"
    with pytest.raises(TypeError, match="sessions list"):
        client.list_sessions()
    state["sessions"] = "not-map"
    with pytest.raises(TypeError, match="sessions payload"):
        client.list_sessions()
    client.close()


def test_control_client_wait_get_delete_otlp_and_session_helpers() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/wait"):
            if request.method == "GET":
                return httpx.Response(405)
            return httpx.Response(
                200,
                json={
                    "id": "s",
                    "calls": [
                        {
                            "id": "call-1",
                            "seq": 1,
                            "request": {"messages": [{"role": "user", "content": "hi"}]},
                            "response": {"choices": [{"message": {"content": "ok"}}]},
                        }
                    ],
                },
            )
        if path == "/sessions/bad":
            return httpx.Response(200, json="not-a-map")
        if path == "/sessions/s":
            if request.method == "DELETE":
                return httpx.Response(204)
            return httpx.Response(
                200,
                json={
                    "id": "s",
                    "calls": [
                        {
                            "id": "call-1",
                            "seq": 1,
                            "request": {"messages": [{"role": "user", "content": "hi"}]},
                            "response": {"choices": [{"message": {"content": "ok"}}]},
                        }
                    ],
                },
            )
        if path == "/sessions/missing":
            if request.method == "DELETE":
                return httpx.Response(404)
            return httpx.Response(404, content=b"")
        if path.endswith("/otlp"):
            if path.endswith("/sessions/missing/otlp"):
                return httpx.Response(404, json={"error": "nope"})
            return httpx.Response(200, content=b"otlp")
        if path == "/sessions":
            return httpx.Response(500, json={"error": "boom"})
        return httpx.Response(404)

    client = _client(handler)
    trace = client.wait_session("s", timeout=1, quiet=0.01, min_calls=0)
    assert len(trace.calls) == 1
    with pytest.raises(TypeError, match="session payload"):
        client.get_session("bad")
    client.delete_session("s")
    client.delete_session("missing")
    assert client.otlp("s") == b"otlp"
    with pytest.raises(SessionNotFound):
        client.otlp("missing")
    assert client._known_session_ids() == []

    session = Session(
        session_id="s",
        proxy=ProxyHandle(url="http://127.0.0.1:7777", control=client),
    )
    assert session.base_url == "http://127.0.0.1:7777"
    assert session.openai_base_url.endswith("/v1")
    assert "/s/s/v1" in session.path_base_url()
    assert "/s/s/u/openai" in session.path_base_url("openai")
    assert session.headers["X-Test-Session"] == "s"
    assert "llmentary.session.id=" in session.baggage
    assert session.client.base_url is not None
    session.cleanup()
    session.otlp()
    session.mark()
    session.query(api_type="chat_completions", model="gpt").offers_tool("t")
    session.query().with_tools()
    session.query().without_tools()
    session.query().since(1)
    client.close()


def test_wait_for_health_times_out() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    client = _client(handler)
    with pytest.raises(TimeoutError, match="did not become healthy"):
        client.wait_for_health(timeout=0.05, interval=0.01)
    client.close()


def test_start_proxy_attaches_to_existing_url(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {"health": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/healthz":
            seen["health"] += 1
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(404)

    real_client = httpx.Client

    def factory(*_args: Any, **_kwargs: Any) -> httpx.Client:
        return real_client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr("llmentary.client.httpx.Client", factory)
    with start_proxy(proxy_url="http://127.0.0.1:7777", startup_timeout=1) as handle:
        assert handle.url == "http://127.0.0.1:7777"
        assert handle.process is None
    assert seen["health"] >= 1


def test_start_proxy_failed_start_redacts_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "llmentary"
    binary.write_text("#!/bin/sh\necho 'bearer sk-secret-token'\nexit 1\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setattr(
        "llmentary.client.ControlClient.wait_for_health",
        lambda self, timeout=10.0, interval=0.05: (_ for _ in ()).throw(
            TimeoutError("down")
        ),
    )
    with pytest.raises(RuntimeError, match="failed to start") as caught:
        with start_proxy(binary=binary, startup_timeout=0.2, env={"PATH": ""}):
            pass
    text = str(caught.value)
    assert "sk-secret-token" not in text
    assert "***" in text


def test_redact_ports_and_payload_helpers() -> None:
    assert redact_command(["llmentary", "-control-token", "secret"]) == [
        "llmentary",
        "-control-token",
        "***",
    ]
    assert redact_command(["llmentary", "-control-token=secret"]) == [
        "llmentary",
        "-control-token=***",
    ]
    assert "sk-abc" not in redact_secrets("key sk-abc and Bearer tok")
    assert "Bearer ***" in redact_secrets("Bearer tok")

    host = "127.0.0.1"
    listen, control = allocate_listen_and_control_ports(
        host, {"LLMENTARY_PORT": "18001", "LLMENTARY_CONTROL_PORT": "18002"}
    )
    assert (listen, control) == (18001, 18002)
    listen, control = allocate_listen_and_control_ports(host, {"LLMENTARY_PORT": "18003"})
    assert listen == 18003
    assert control != 18003
    listen, control = allocate_listen_and_control_ports(
        host, {"LLMENTARY_CONTROL_PORT": "18004"}
    )
    assert control == 18004
    assert listen != 18004
    assert free_port(host) > 0
    assert new_session_id("t").startswith("t-")

    response = httpx.Response(404, content=b"")
    assert control_error_payload(response) is None
    response = httpx.Response(404, text="not-json")
    assert control_error_payload(response) is None
    response = httpx.Response(404, json=["x"])
    assert control_error_payload(response) is None
    response = httpx.Response(404, json={"error": {"type": "session_not_found"}})
    assert control_error_payload(response) == {"type": "session_not_found"}
    response = httpx.Response(404, json={"type": "other"})
    assert control_error_payload(response) == {"type": "other"}
    assert coerce_session_row({"id": "a"}) == {"id": "a"}
    assert coerce_session_row("nope") == {}
    assert default_control_url("http://127.0.0.1:9", {"LLMENTARY_CONTROL_URL": "http://c"}) == (
        "http://c"
    )
    assert default_control_url("http://127.0.0.1:9", {}) == "http://127.0.0.1:7778"


def test_proxy_handle_close_terminates_process(tmp_path: Path) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    control = ControlClient("http://127.0.0.1:7778", client=http)
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    log_path = tmp_path / "proxy.log"
    log_path.write_text("log\n", encoding="utf-8")
    handle = ProxyHandle(
        url="http://127.0.0.1:7777",
        control=control,
        process=process,
        log_file=log_path.open("w", encoding="utf-8"),
        log_path=log_path,
    )
    handle.close()
    assert process.poll() is not None
    assert not log_path.exists()


def test_session_expect_and_query_on_recorded_calls() -> None:
    payload = {
        "id": "s",
        "calls": [
            {
                "id": "call-1",
                "seq": 1,
                "session_id": "s",
                "model": "fake-model",
                "api_family": "chat.completions",
                "request": {
                    "model": "fake-model",
                    "messages": [{"role": "user", "content": "hi"}],
                    "tools": [{"function": {"name": "lookup"}}],
                },
                "response": {
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
                },
                "http_status": 200,
            }
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/wait") or request.url.path.endswith("/sessions/s"):
            return httpx.Response(200, json=payload)
        if request.url.path == "/sessions":
            return httpx.Response(200, json={"sessions": [{"id": "s", "calls": 1}]})
        return httpx.Response(404)

    client = _client(handler)
    session = Session(
        session_id="s",
        proxy=ProxyHandle(url="http://127.0.0.1:7777", control=client),
    )
    expect = session.expect()
    expect.call_count(1)
    session.query(api_type="chat", model="fake-model").expect().call_count(1)
    session.query().offers_tool("lookup").expect().call_count(1)
    with session.soft_expect() as soft:
        soft.call_count(1)
    session.list_sessions()
    session.trace()
    client.close()


def test_session_expect_no_calls_and_allow_empty() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/wait"):
            return httpx.Response(
                200, json={"id": "empty", "calls": []}
            )
        if request.url.path == "/sessions":
            return httpx.Response(200, json={"sessions": []})
        return httpx.Response(404, json={"error": {"type": "session_not_found"}})

    client = _client(handler)
    session = Session(
        session_id="empty",
        proxy=ProxyHandle(url="http://127.0.0.1:7777", control=client),
    )
    session.expect_no_calls(timeout=1, quiet=0.01)
    session.expect(allow_empty=True).call_count(0)
    client.close()


def test_session_expect_no_calls_fails_when_calls_exist() -> None:
    payload = {
        "id": "s",
        "calls": [
            {
                "id": "call-1",
                "seq": 1,
                "request": {"messages": [{"role": "user", "content": "hi"}]},
                "response": {"choices": [{"message": {"content": "ok"}}]},
            }
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/wait") or request.url.path.endswith("/sessions/s"):
            return httpx.Response(200, json=payload)
        return httpx.Response(200, json={"sessions": []})

    client = _client(handler)
    session = Session(
        session_id="s",
        proxy=ProxyHandle(url="http://127.0.0.1:7777", control=client),
    )
    with pytest.raises(AssertionError, match="expected 0 LLM calls"):
        session.expect_no_calls(timeout=1, quiet=0.01)
    client.close()


def test_mark_returns_zero_on_missing_session() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/sessions":
            return httpx.Response(200, json={"sessions": []})
        return httpx.Response(404, json={"error": {"type": "session_not_found"}})

    client = _client(handler)
    session = Session(
        session_id="ghost",
        proxy=ProxyHandle(url="http://127.0.0.1:7777", control=client),
    )
    assert session.mark() == 0
    client.close()
