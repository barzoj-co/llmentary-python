from __future__ import annotations

import contextlib
import os
import re
import shlex
import socket
import subprocess
import tempfile
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TextIO, cast
from urllib.parse import quote, urlparse

import httpx

from .assertions import Expectation, SoftExpectation
from .binary import find_binary
from .types import (
    SessionInfo,
    SessionTrace,
    normalize_api_type,
    session_info_from_mapping,
    zero_calls_message,
)
from .wait import (
    WaitTimeoutError,
    default_wait_quiet,
    default_wait_timeout,
    resolve_wait_quiet,
    resolve_wait_timeout,
    wait_query_string,
    wait_timeout_error,
)

SESSION_HEADER = "X-Test-Session"
BAGGAGE_SESSION_KEY = "llmentary.session.id"


class SessionNotFound(LookupError):
    """Raised when a control-API GET returns 404 ``session_not_found``."""

    def __init__(
        self,
        session_id: str,
        message: str = "",
        *,
        known_ids: Sequence[str] = (),
    ) -> None:
        text = message or zero_calls_message(session_id, known_ids)
        super().__init__(text)
        self.session_id = session_id
        self.known_ids = tuple(known_ids)


class ControlClient:
    """Client for the llmentary control API."""

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
            f"llmentary did not become healthy at {self.base_url}"
        ) from last_error

    def config(self) -> Mapping[str, Any]:
        response = self._client.get(f"{self.base_url}/config", headers=self.headers)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise TypeError(f"Unexpected config payload: {payload!r}")
        return payload

    def version(self) -> str:
        payload = self.config()
        value = payload.get("version")
        if isinstance(value, str) and value:
            return value
        raise TypeError(f"GET /config did not include a version string: {payload!r}")

    def list_sessions(self, *, source: Optional[str] = None) -> list[SessionInfo]:
        params = {"source": source} if source else None
        response = self._client.get(
            f"{self.base_url}/sessions",
            headers=self.headers,
            params=params,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise TypeError(f"Unexpected sessions payload: {payload!r}")
        raw_sessions = payload.get("sessions")
        if raw_sessions is None:
            raw_sessions = payload.get("items")
        if raw_sessions is None:
            return []
        if not isinstance(raw_sessions, list):
            raise TypeError(f"Unexpected sessions list: {raw_sessions!r}")
        return [session_info_from_mapping(coerce_session_row(item)) for item in raw_sessions]

    def get_session(self, session_id: str) -> SessionTrace:
        response = self._client.get(
            f"{self.base_url}/sessions/{session_id}",
            headers=self.headers,
        )
        if response.status_code == 404:
            raise self._session_not_found(session_id, response)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise TypeError(f"Unexpected session payload for {session_id}: {payload!r}")
        return SessionTrace.from_payload(session_id, payload)

    def wait_session(
        self,
        session_id: str,
        *,
        timeout: Optional[float] = None,
        quiet: Optional[float] = None,
        quiet_for: Optional[float] = None,
        min_calls: Optional[int] = None,
    ) -> SessionTrace:
        resolved_timeout = resolve_wait_timeout(timeout)
        resolved_quiet = resolve_wait_quiet(quiet, quiet_for)
        query = wait_query_string(resolved_timeout, resolved_quiet, min_calls=min_calls)
        url = f"{self.base_url}/sessions/{session_id}/wait?{query}"
        request_timeout = resolved_timeout + resolved_quiet + 5
        try:
            response = self._wait_request("GET", url, request_timeout)
            if response.status_code == 405:
                response = self._wait_request("POST", url, request_timeout)
        except httpx.TimeoutException as exc:
            raise self._wait_timeout(session_id, resolved_timeout) from exc
        if response.status_code == 504:
            raise self._wait_timeout(session_id, resolved_timeout, response)
        if response.status_code == 404:
            raise self._session_not_found(session_id, response)
        response.raise_for_status()
        payload: Optional[Mapping[str, Any]] = None
        if response.content:
            try:
                parsed = response.json()
            except ValueError:
                parsed = None
            if isinstance(parsed, Mapping):
                payload = parsed
                trace = SessionTrace.from_payload(session_id, parsed)
                if trace.calls:
                    return trace
        try:
            return self.get_session(session_id)
        except SessionNotFound:
            if payload is not None:
                return SessionTrace.from_payload(session_id, payload)
            return SessionTrace.from_payload(session_id, {"id": session_id, "calls": []})

    def _wait_request(self, method: str, url: str, request_timeout: float) -> httpx.Response:
        return self._client.request(
            method,
            url,
            headers=self.headers,
            timeout=request_timeout,
        )

    def _wait_timeout(
        self,
        session_id: str,
        timeout: float,
        response: Optional[httpx.Response] = None,
    ) -> BaseException:
        payload: Optional[Mapping[str, Any]] = None
        if response is not None and response.content:
            try:
                parsed = response.json()
            except ValueError:
                parsed = None
            if isinstance(parsed, Mapping):
                payload = parsed
        if payload is None:
            try:
                payload = self.get_session(session_id).raw
            except SessionNotFound:
                payload = {}
            except (httpx.HTTPError, OSError, TypeError):
                payload = {}
        return wait_timeout_error(session_id, timeout, payload)

    def _session_not_found(
        self,
        session_id: str,
        response: httpx.Response,
    ) -> SessionNotFound:
        error = control_error_payload(response)
        message = ""
        if error:
            raw_message = error.get("message")
            if isinstance(raw_message, str):
                message = raw_message
        known_ids = self._known_session_ids()
        if known_ids and "known sessions:" not in message.lower():
            suffix = f"known sessions: {', '.join(known_ids)}"
            message = (
                f"{message}; {suffix}"
                if message
                else zero_calls_message(session_id, known_ids)
            )
        elif not message:
            message = zero_calls_message(session_id, known_ids)
        return SessionNotFound(session_id, message, known_ids=known_ids)

    def _known_session_ids(self) -> list[str]:
        try:
            return [item.id for item in self.list_sessions() if item.id]
        except (httpx.HTTPError, OSError, TypeError, ValueError):
            return []

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
        if response.status_code == 404:
            raise self._session_not_found(session_id, response)
        response.raise_for_status()
        return cast(bytes, response.content)


@dataclass
class ProxyHandle:
    """Running or attached llmentary proxy."""

    url: str
    control: ControlClient
    process: Optional[subprocess.Popen[str]] = None
    log_file: Optional[TextIO] = None
    log_path: Optional[Path] = None

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
        if self.log_file is not None:
            self.log_file.close()
        if self.log_path is not None:
            with contextlib.suppress(OSError):
                self.log_path.unlink(missing_ok=True)


@dataclass
class Session:
    """Function-scoped test session helper returned by the ``llmentary`` fixture."""

    session_id: str
    proxy: ProxyHandle
    wait_timeout: float = field(default_factory=default_wait_timeout)
    quiet_for: float = field(default_factory=default_wait_quiet)

    @property
    def base_url(self) -> str:
        return self.proxy.url

    @property
    def openai_base_url(self) -> str:
        return f"{self.proxy.url.rstrip('/')}/v1"

    @property
    def headers(self) -> dict[str, str]:
        return {SESSION_HEADER: self.session_id, "baggage": self.baggage}

    @property
    def baggage(self) -> str:
        return f"{BAGGAGE_SESSION_KEY}={encode_baggage_value(self.session_id)}"

    @property
    def client(self) -> httpx.Client:
        return httpx.Client(base_url=self.openai_base_url, headers=self.headers)

    def wait(
        self,
        *,
        timeout: Optional[float] = None,
        quiet: Optional[float] = None,
        quiet_for: Optional[float] = None,
        min_calls: Optional[int] = None,
    ) -> SessionTrace:
        return self.proxy.control.wait_session(
            self.session_id,
            timeout=timeout if timeout is not None else self.wait_timeout,
            quiet=resolve_wait_quiet(quiet, quiet_for)
            if quiet is not None or quiet_for is not None
            else self.quiet_for,
            min_calls=min_calls,
        )

    def trace(self) -> SessionTrace:
        return self.proxy.control.get_session(self.session_id)

    def list_sessions(self, *, source: Optional[str] = None) -> list[SessionInfo]:
        return self.proxy.control.list_sessions(source=source)

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
        trace = self._trace_or_zero_calls()
        return SoftExpectation(trace) if soft else Expectation(trace)

    @contextlib.contextmanager
    def soft_expect(self) -> Iterator[SoftExpectation]:
        expectation = SoftExpectation(self._trace_or_zero_calls())
        yield expectation
        expectation.assert_all()

    def _trace_or_zero_calls(self) -> SessionTrace:
        try:
            trace = self.wait()
        except WaitTimeoutError as exc:
            if exc.calls == 0:
                raise AssertionError(self._zero_calls_error()) from exc
            raise
        except SessionNotFound as exc:
            raise AssertionError(self._zero_calls_error()) from exc
        if not trace.calls:
            raise AssertionError(self._zero_calls_error(trace))
        return trace

    def _zero_calls_error(self, trace: Optional[SessionTrace] = None) -> str:
        known = self._known_session_ids()
        message = zero_calls_message(self.session_id, known)
        if trace is None:
            return message
        return f"{message}\n\n{trace.trace_summary()}"

    def _known_session_ids(self) -> list[str]:
        try:
            return [item.id for item in self.list_sessions() if item.id]
        except (httpx.HTTPError, OSError, TypeError, ValueError):
            return []


@dataclass
class Query:
    session: Session
    api_type: Optional[str] = None
    model: Optional[str] = None

    def expect(self, *, soft: bool = False) -> Expectation:
        trace = self.session._trace_or_zero_calls().filtered(
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
    startup_timeout: float = 10.0,
    env: Optional[Mapping[str, str]] = None,
) -> Iterator[ProxyHandle]:
    """Start or attach to an llmentary proxy."""

    merged_env = dict(os.environ)
    if env:
        merged_env.update(env)

    attached_url = proxy_url or merged_env.get("LLMENTARY_PROXY_URL")
    token = token or merged_env.get("LLMENTARY_CONTROL_TOKEN")
    if attached_url:
        control_url = default_control_url(attached_url, merged_env)
        control = ControlClient(control_url, token=token)
        control.wait_for_health(timeout=startup_timeout)
        handle = ProxyHandle(url=attached_url.rstrip("/"), control=control)
        try:
            yield handle
        finally:
            handle.close()
        return

    bin_path = binary or find_binary(env=merged_env)
    host = merged_env.get("LLMENTARY_HOST", "127.0.0.1")
    port = int(merged_env.get("LLMENTARY_PORT") or free_port(host))
    control_port = int(merged_env.get("LLMENTARY_CONTROL_PORT") or free_port(host))
    url = f"http://{host}:{port}"
    control_url = f"http://{host}:{control_port}"
    resolved_upstream = upstream_url or merged_env.get("LLMENTARY_UPSTREAM_BASE_URL")
    resolved_api_key = upstream_api_key or merged_env.get("LLMENTARY_UPSTREAM_API_KEY")
    apply_upstream_api_key(merged_env, resolved_api_key)
    command = build_start_command(
        bin_path=bin_path,
        host=host,
        port=port,
        control_port=control_port,
        upstream_url=resolved_upstream,
        upstream_api_key=resolved_api_key,
        token=token,
        env=merged_env,
    )
    log_file = cast(
        TextIO,
        tempfile.NamedTemporaryFile(
            mode="w+",
            prefix="llmentary-proxy-",
            suffix=".log",
            encoding="utf-8",
            delete=False,
        ),
    )
    process = subprocess.Popen(
        command,
        env=merged_env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    control = ControlClient(control_url, token=token)
    handle = ProxyHandle(
        url=url,
        control=control,
        process=process,
        log_file=log_file,
        log_path=Path(log_file.name),
    )
    try:
        try:
            control.wait_for_health(timeout=startup_timeout)
        except BaseException as exc:
            output = read_process_output(process, log_file=log_file)
            raise RuntimeError(
                "llmentary failed to start with command "
                f"{redact_command(command)!r}.\n{redact_secrets(output)}"
            ) from exc
        yield handle
    finally:
        handle.close()


def build_start_command(
    *,
    bin_path: Path,
    host: str,
    port: int,
    control_port: int,
    upstream_url: Optional[str],
    upstream_api_key: Optional[str],
    token: Optional[str],
    env: Mapping[str, str],
) -> list[str]:
    listen = f"{host}:{port}"
    control_listen = f"{host}:{control_port}"
    upstream_name = env.get("LLMENTARY_UPSTREAM_NAME", "openai")
    normalized_upstream = normalize_upstream_url(upstream_url) if upstream_url else ""
    upstream_mapping = f"{upstream_name}={normalized_upstream}" if normalized_upstream else ""
    template = env.get("LLMENTARY_START_COMMAND")
    if template:
        rendered = template.format(
            bin=str(bin_path),
            host=host,
            port=port,
            listen=listen,
            control=control_listen,
            control_listen=control_listen,
            upstream=upstream_mapping,
            upstream_url=normalized_upstream,
            upstream_api_key=upstream_api_key or "",
            token=token or "",
        )
        return shlex.split(rendered)

    command = [str(bin_path), "-listen", listen, "-control", control_listen]
    if upstream_mapping:
        command.extend(["-upstream", upstream_mapping])
    if token:
        command.extend(["-control-token", token])
    return command


def normalize_upstream_url(url: str) -> str:
    """Pass the upstream URL through without rewriting its path.

    The proxy owns the overlap-join rule that prevents ``/v1/v1``. Both
    ``https://api.openai.com`` and ``https://api.openai.com/v1`` are valid and
    must be forwarded unchanged so the three clients share one join table.
    """

    return url.strip().rstrip("/")


def apply_upstream_api_key(env: dict[str, str], api_key: Optional[str]) -> None:
    """Force the proxy child to use the configured upstream key.

    ``setdefault`` would leave a pre-existing ``OPENAI_API_KEY`` in place and the
    Go proxy would authenticate with the wrong credential.
    """

    if not api_key:
        return
    env["OPENAI_API_KEY"] = api_key
    env["LLMENTARY_UPSTREAM_API_KEY"] = api_key


def encode_baggage_value(value: str) -> str:
    """Percent-encode a W3C baggage value.

    Session ids are derived from pytest node names and may contain comma, space,
    or equals, which split baggage list members if left raw.
    """

    return quote(value, safe="-._~")


def default_control_url(proxy_url: str, env: Mapping[str, str]) -> str:
    configured = env.get("LLMENTARY_CONTROL_URL")
    if configured:
        return configured.rstrip("/")
    parsed = urlparse(proxy_url)
    host = parsed.hostname or "127.0.0.1"
    scheme = parsed.scheme or "http"
    return f"{scheme}://{host}:7778"


def redact_command(command: Sequence[str]) -> list[str]:
    redacted = list(command)
    secret_flags = {"-control-token", "--control-token", "-upstream-api-key", "--upstream-api-key"}
    for index, part in enumerate(redacted):
        name, sep, _value = part.partition("=")
        if sep and name in secret_flags:
            redacted[index] = f"{name}=***"
            continue
        if part in secret_flags and index + 1 < len(redacted):
            redacted[index + 1] = "***"
    return redacted


def redact_secrets(text: str) -> str:
    text = re.sub(r"(sk-[A-Za-z0-9_-]+)", "***", text)
    return re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._\-]+", r"\1***", text)


def free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def read_process_output(process: subprocess.Popen[str], *, log_file: TextIO) -> str:
    if process.poll() is None:
        process.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=2)
    with contextlib.suppress(Exception):
        log_file.flush()
        log_file.seek(0)
        return log_file.read()
    return "<could not read process output>"


def new_session_id(prefix: str = "pytest") -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def control_error_payload(response: httpx.Response) -> Optional[Mapping[str, Any]]:
    if not response.content:
        return None
    try:
        parsed = response.json()
    except ValueError:
        return None
    if not isinstance(parsed, Mapping):
        return None
    error = parsed.get("error")
    return error if isinstance(error, Mapping) else parsed


def coerce_session_row(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def require_binary_for_tests() -> Path:
    return find_binary()
