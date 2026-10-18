from __future__ import annotations

import contextlib
import os
import shlex
import socket
import subprocess
import time
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, cast

import httpx

from .assertions import Expectation, SoftExpectation
from .binary import BinaryResolutionError, find_binary
from .types import SessionTrace, normalize_api_type

SESSION_HEADER = "X-Test-Session"
BAGGAGE_SESSION_KEY = "llmtestlab.session.id"


class ControlClient:
    """Client for the llm-test-lab control API."""

    def __init__(
        self,
        base_url: str,
        *,
        token: Optional[str] = None,
        timeout: float = 10.0,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self._own_client = client is None
        self._client = client or httpx.Client(timeout=timeout)

    @property
    def headers(self) -> dict[str, str]:
        if not self.token:
            return {}
        return {"Authorization": f"Bearer {self.token}"}

    def close(self) -> None:
        if self._own_client:
            self._client.close()

    def health(self) -> Mapping[str, Any]:
        response = self._client.get(f"{self.base_url}/healthz", headers=self.headers)
        response.raise_for_status()
        if not response.content:
            return {"ok": True}
        try:
            payload = response.json()
        except ValueError:
            return {"ok": response.text}
        return payload if isinstance(payload, Mapping) else {"ok": payload}

    def wait_for_health(self, *, timeout: float = 10.0, interval: float = 0.05) -> None:
        deadline = time.monotonic() + timeout
        last_error: Optional[BaseException] = None
        while time.monotonic() < deadline:
            try:
                self.health()
                return
            except (httpx.HTTPError, OSError) as exc:
                last_error = exc
                time.sleep(interval)
        raise TimeoutError(
            f"llm-test-lab did not become healthy at {self.base_url}"
        ) from last_error

    def get_session(self, session_id: str) -> SessionTrace:
        response = self._client.get(
            f"{self.base_url}/sessions/{session_id}",
            headers=self.headers,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise TypeError(f"Unexpected session payload for {session_id}: {payload!r}")
        return SessionTrace.from_payload(session_id, payload)

    def wait_session(
        self,
        session_id: str,
        *,
        timeout: float = 5.0,
        quiet_for: float = 0.2,
    ) -> SessionTrace:
        params = {"timeout": f"{timeout}s", "quiet_for": f"{quiet_for}s"}
        url = f"{self.base_url}/sessions/{session_id}/wait"
        response = self._client.get(url, params=params, headers=self.headers)
        if response.status_code == 405:
            response = self._client.post(url, json=params, headers=self.headers)
        response.raise_for_status()
        if response.content:
            try:
                payload = response.json()
            except ValueError:
                payload = None
            if isinstance(payload, Mapping) and any(
                key in payload for key in ("calls", "requests", "records", "session")
            ):
                return SessionTrace.from_payload(session_id, payload)
        return self.get_session(session_id)

    def delete_session(self, session_id: str) -> None:
        response = self._client.delete(
            f"{self.base_url}/sessions/{session_id}",
            headers=self.headers,
        )
        if response.status_code == 404:
            return
        response.raise_for_status()

    def otlp(self, session_id: str) -> bytes:
        response = self._client.get(
            f"{self.base_url}/sessions/{session_id}/otlp",
            headers=self.headers,
        )
        response.raise_for_status()
        return cast(bytes, response.content)


@dataclass
class ProxyHandle:
    """Running or attached llm-test-lab proxy."""

    url: str
    control: ControlClient
    process: Optional[subprocess.Popen[str]] = None

    def close(self) -> None:
        self.control.close()
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)


@dataclass
class LabSession:
    """Function-scoped test session helper returned by the ``lab`` fixture."""

    session_id: str
    proxy: ProxyHandle
    wait_timeout: float = 5.0
    quiet_for: float = 0.2

    @property
    def base_url(self) -> str:
        return self.proxy.url

    @property
    def openai_base_url(self) -> str:
        return f"{self.proxy.url.rstrip('/')}/v1"

    @property
    def headers(self) -> dict[str, str]:
        return {SESSION_HEADER: self.session_id}

    @property
    def baggage(self) -> str:
        return f"{BAGGAGE_SESSION_KEY}={self.session_id}"

    @property
    def client(self) -> httpx.Client:
        return httpx.Client(base_url=self.openai_base_url, headers=self.headers)

    def wait(
        self,
        *,
        timeout: Optional[float] = None,
        quiet_for: Optional[float] = None,
    ) -> SessionTrace:
        return self.proxy.control.wait_session(
            self.session_id,
            timeout=timeout if timeout is not None else self.wait_timeout,
            quiet_for=quiet_for if quiet_for is not None else self.quiet_for,
        )

    def trace(self) -> SessionTrace:
        return self.proxy.control.get_session(self.session_id)

    def cleanup(self) -> None:
        self.proxy.control.delete_session(self.session_id)

    def otlp(self) -> bytes:
        return self.proxy.control.otlp(self.session_id)

    def query(
        self,
        *,
        api_type: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Query:
        return Query(self, api_type=api_type, model=model)

    def expect(self, *, soft: bool = False) -> Expectation:
        trace = self.wait()
        return SoftExpectation(trace) if soft else Expectation(trace)

    @contextlib.contextmanager
    def soft_expect(self) -> Iterator[SoftExpectation]:
        expectation = SoftExpectation(self.wait())
        yield expectation
        expectation.assert_all()


@dataclass
class Query:
    lab: LabSession
    api_type: Optional[str] = None
    model: Optional[str] = None

    def expect(self, *, soft: bool = False) -> Expectation:
        trace = self.lab.wait().filtered(
            api_type=normalize_api_type(self.api_type) if self.api_type else None,
            model=self.model,
        )
        return SoftExpectation(trace) if soft else Expectation(trace)


@contextlib.contextmanager
def start_proxy(
    *,
    upstream_url: Optional[str] = None,
    upstream_api_key: Optional[str] = None,
    proxy_url: Optional[str] = None,
    token: Optional[str] = None,
    binary: Optional[Path] = None,
    download: bool = False,
    startup_timeout: float = 10.0,
    env: Optional[Mapping[str, str]] = None,
) -> Iterator[ProxyHandle]:
    """Start or attach to a llm-test-lab proxy."""

    merged_env = dict(os.environ)
    if env:
        merged_env.update(env)

    attached_url = proxy_url or merged_env.get("LLMTESTLAB_PROXY_URL")
    token = token or merged_env.get("LLMTESTLAB_CONTROL_TOKEN")
    if attached_url:
        control = ControlClient(attached_url, token=token)
        control.wait_for_health(timeout=startup_timeout)
        handle = ProxyHandle(url=attached_url.rstrip("/"), control=control)
        try:
            yield handle
        finally:
            handle.close()
        return

    bin_path = binary or find_binary(
        env=merged_env,
        download=download or merged_env.get("LLMTESTLAB_DOWNLOAD") == "1",
    )
    host = merged_env.get("LLMTESTLAB_HOST", "127.0.0.1")
    port = int(merged_env.get("LLMTESTLAB_PORT") or free_port(host))
    url = f"http://{host}:{port}"
    command = build_start_command(
        bin_path=bin_path,
        host=host,
        port=port,
        upstream_url=upstream_url or merged_env.get("LLMTESTLAB_UPSTREAM_BASE_URL"),
        upstream_api_key=upstream_api_key or merged_env.get("LLMTESTLAB_UPSTREAM_API_KEY"),
        token=token,
        env=merged_env,
    )
    process = subprocess.Popen(
        command,
        env=merged_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    control = ControlClient(url, token=token)
    handle = ProxyHandle(url=url, control=control, process=process)
    try:
        try:
            control.wait_for_health(timeout=startup_timeout)
        except BaseException as exc:
            output = read_process_output(process)
            raise RuntimeError(
                f"llm-test-lab failed to start with command {command!r}.\n{output}"
            ) from exc
        yield handle
    finally:
        handle.close()


def build_start_command(
    *,
    bin_path: Path,
    host: str,
    port: int,
    upstream_url: Optional[str],
    upstream_api_key: Optional[str],
    token: Optional[str],
    env: Mapping[str, str],
) -> list[str]:
    listen = f"{host}:{port}"
    template = env.get("LLMTESTLAB_START_COMMAND")
    if template:
        rendered = template.format(
            bin=str(bin_path),
            host=host,
            port=port,
            listen=listen,
            upstream_url=upstream_url or "",
            upstream_api_key=upstream_api_key or "",
            token=token or "",
        )
        return shlex.split(rendered)

    command = [str(bin_path), "--listen", listen]
    if upstream_url:
        command.extend(["--upstream-url", upstream_url])
    if upstream_api_key:
        command.extend(["--upstream-api-key", upstream_api_key])
    if token:
        command.extend(["--control-token", token])
    return command


def free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def read_process_output(process: subprocess.Popen[str]) -> str:
    if process.poll() is None:
        process.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=2)
    if process.stdout is None:
        return "<no process output>"
    with contextlib.suppress(Exception):
        return process.stdout.read()
    return "<could not read process output>"


def new_session_id(prefix: str = "pytest") -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def require_binary_for_tests() -> Path:
    try:
        return find_binary()
    except BinaryResolutionError as exc:
        raise RuntimeError(str(exc)) from exc
