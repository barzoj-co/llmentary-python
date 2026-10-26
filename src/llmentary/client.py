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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional, TextIO, cast
from urllib.parse import quote, urlparse

import httpx

from .assertions import Expectation, SoftExpectation
from .binary import find_binary
from .types import (
    OUTCOME_CANCELLED,
    OUTCOME_COMMITTED,
    OUTCOME_FAILED,
    ProxyConfig,
    SessionInfo,
    SessionTrace,
    normalize_api_type,
    proxy_config_from_mapping,
    session_info_from_mapping,
    zero_calls_message,
)
from .wait import (
    WaitTimeoutError,
    default_wait_quiet,
    default_wait_timeout,
    explicit_active_count,
    resolve_wait_quiet,
    resolve_wait_timeout,
    wait_query_string,
    wait_timeout_error,
)

SESSION_HEADER = "X-Test-Session"
BAGGAGE_SESSION_KEY = "llmentary.session.id"
EXCLUDE_CANCELLED_UNSUPPORTED_MESSAGE = (
    "proxy does not support exclude_cancelled; upgrade the proxy to >= 0.5.3"
)


class SessionNotFound(LookupError):
    """Raised when the control API returns 404 ``session_not_found``.

    That is ``GET /sessions/{id}``, ``GET /sessions/{id}/otlp``, and
    ``GET /sessions/{id}/wait`` with ``min_calls=0`` on a session that was
    never created (or was ``DELETE``d). ``wait`` with ``min_calls>=1`` still
    raises ``WaitTimeoutError`` (504) if no call arrives before ``timeout``.
    """

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

    def proxy_config(self) -> ProxyConfig:
        """Typed ``GET /config`` (version, upstreams, spec targets, timeouts)."""

        return proxy_config_from_mapping(self.config())

    def version(self) -> str:
        payload = self.config()
        value = payload.get("version")
        if isinstance(value, str) and value:
            return value
        raise TypeError(f"GET /config did not include a version string: {payload!r}")

    def openapi(self) -> dict[str, Any]:
        """Served control contract from ``GET /openapi.yaml`` (with the token)."""

        import yaml  # type: ignore[import-untyped]

        response = self._client.get(f"{self.base_url}/openapi.yaml", headers=self.headers)
        response.raise_for_status()
        loaded = yaml.safe_load(response.text)
        if not isinstance(loaded, dict):
            raise TypeError(f"GET /openapi.yaml did not parse as a mapping: {type(loaded)!r}")
        return loaded

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
        exclude_cancelled: bool = False,
    ) -> SessionTrace:
        resolved_timeout = resolve_wait_timeout(timeout)
        resolved_quiet = resolve_wait_quiet(quiet, quiet_for)
        query = wait_query_string(
            resolved_timeout,
            resolved_quiet,
            min_calls=min_calls,
            exclude_cancelled=exclude_cancelled,
        )
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
        unsupported = self._unsupported_exclude_cancelled(response)
        if unsupported is not None:
            raise unsupported
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
        list_active = None
        error_payload = payload.get("error")
        error_mapping: Mapping[str, Any] = (
            error_payload if isinstance(error_payload, Mapping) else {}
        )
        nested_session = payload.get("session")
        session_mapping: Mapping[str, Any] = (
            nested_session if isinstance(nested_session, Mapping) else {}
        )
        if explicit_active_count(error_mapping, session_mapping, payload) is None:
            list_active = self._list_session_active(session_id)
        return wait_timeout_error(
            session_id, timeout, payload, list_active=list_active
        )

    def _unsupported_exclude_cancelled(
        self,
        response: httpx.Response,
    ) -> Optional[ValueError]:
        """400 ``invalid_param`` for ``exclude_cancelled`` means an older proxy."""

        if response.status_code != 400:
            return None
        error = control_error_payload(response)
        if not error:
            return None
        if error.get("type") != "invalid_param":
            return None
        if error.get("param") != "exclude_cancelled":
            return None
        return ValueError(EXCLUDE_CANCELLED_UNSUPPORTED_MESSAGE)

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
        return SessionNotFound(
            session_id,
            with_known_sessions(message, session_id, known_ids),
            known_ids=known_ids,
        )

    def _known_session_ids(self) -> list[str]:
        try:
            return [item.id for item in self.list_sessions() if item.id]
        except (httpx.HTTPError, OSError, TypeError, ValueError):
            return []

    def _list_session_active(self, session_id: str) -> Optional[int]:
        try:
            for item in self.list_sessions():
                if item.id != session_id:
                    continue
                value = item.raw.get("active")
                if isinstance(value, int) and not isinstance(value, bool):
                    return value
        except (httpx.HTTPError, OSError, TypeError, ValueError):
            return None
        return None

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

    def path_base_url(self, upstream: Optional[str] = None) -> str:
        """OpenAI-compatible base URL that attributes calls via ``/s/{session}/``.

        Use this when the service under test cannot send ``X-Test-Session``.
        ``upstream=None`` yields ``{proxy}/s/{id}/v1``. A named upstream yields
        ``{proxy}/s/{id}/u/{name}``.
        """

        root = self.proxy.url.rstrip("/")
        session = quote(self.session_id, safe="")
        if upstream:
            return f"{root}/s/{session}/u/{upstream}"
        return f"{root}/s/{session}/v1"

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
        exclude_cancelled: bool = False,
    ) -> SessionTrace:
        return self.proxy.control.wait_session(
            self.session_id,
            timeout=timeout if timeout is not None else self.wait_timeout,
            quiet=resolve_wait_quiet(quiet, quiet_for)
            if quiet is not None or quiet_for is not None
            else self.quiet_for,
            min_calls=min_calls,
            exclude_cancelled=exclude_cancelled,
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

    def mark(self) -> int:
        """Record the current max call sequence number for ``expect(since=...)``."""

        try:
            trace = self.trace()
        except SessionNotFound:
            return 0
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return 0
            raise
        if not trace.calls:
            return 0
        return max(call.sequence_number() for call in trace.calls)

    def expect(
        self,
        *,
        soft: bool = False,
        since: Optional[int] = None,
        allow_empty: bool = False,
    ) -> Expectation:
        """Wait, then return an expectation.

        The default is strict (DOG-2): a never-created session raises
        ``AssertionError('no calls recorded …')``. Pass ``allow_empty=True``
        (or chain ``allow_empty()`` after that) to get an empty expectation
        so ``call_count(0)`` / ``isolated_from`` against an empty other
        can run.
        """

        trace = self._load_trace(allow_empty=allow_empty)
        if since is not None:
            trace = trace.since(since)
        expectation: Expectation = (
            SoftExpectation(trace, session=self)
            if soft
            else Expectation(trace, session=self)
        )
        if allow_empty:
            expectation.allow_empty()
        return expectation

    def expect_no_calls(
        self,
        *,
        timeout: Optional[float] = None,
        quiet: Optional[float] = None,
    ) -> None:
        """Pass when this session recorded zero calls (including 404).

        Calls ``wait(min_calls=0)``. ``SessionNotFound`` and an empty trace
        pass. A session with calls fails with ``expected 0 LLM calls, got N``
        plus the trace summary.
        """

        try:
            trace = self.wait(timeout=timeout, quiet=quiet, min_calls=0)
        except SessionNotFound:
            return
        count = len(trace.calls)
        if count == 0:
            return
        raise AssertionError(
            f"expected 0 LLM calls, got {count}\n\n{trace.trace_summary()}"
        )

    @contextlib.contextmanager
    def soft_expect(
        self,
        *,
        since: Optional[int] = None,
        allow_empty: bool = False,
    ) -> Iterator[SoftExpectation]:
        trace = self._load_trace(allow_empty=allow_empty)
        if since is not None:
            trace = trace.since(since)
        expectation = SoftExpectation(trace, session=self)
        if allow_empty:
            expectation.allow_empty()
        yield expectation
        expectation.assert_all()

    def _load_trace(self, *, allow_empty: bool) -> SessionTrace:
        if allow_empty:
            return self._trace_allowing_empty()
        return self._trace_or_zero_calls()

    def _empty_trace(self) -> SessionTrace:
        return SessionTrace.from_payload(self.session_id, {"id": self.session_id, "calls": []})

    def _trace_allowing_empty(self) -> SessionTrace:
        try:
            return self.wait()
        except WaitTimeoutError as exc:
            if exc.calls == 0:
                return self._empty_trace()
            raise
        except SessionNotFound:
            return self._empty_trace()

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
    offers_tool_name: Optional[str] = None
    include_tools: Optional[bool] = None
    since_seq: Optional[int] = None
    aborted_only: Optional[bool] = None
    outcomes: Optional[tuple[str, ...]] = None

    def offers_tool(self, name: str) -> Query:
        return replace(self, offers_tool_name=name)

    def with_tools(self) -> Query:
        return replace(self, include_tools=True)

    def without_tools(self) -> Query:
        return replace(self, include_tools=False)

    def since(self, mark: int) -> Query:
        return replace(self, since_seq=mark)

    def committed(self) -> Query:
        """Keep calls that were not client-aborted (DOG-50)."""

        return replace(self, aborted_only=False)

    def aborted(self) -> Query:
        """Keep client-aborted calls only (DOG-50)."""

        return replace(self, aborted_only=True)

    def outcome(self, *states: str) -> Query:
        """Keep calls whose terminal outcome is one of ``states`` (DOG-58)."""

        if not states:
            raise ValueError("outcome() requires at least one state")
        allowed = (OUTCOME_COMMITTED, OUTCOME_CANCELLED, OUTCOME_FAILED)
        for state in states:
            if state not in allowed:
                raise ValueError(
                    f"unknown outcome state {state!r}: expected one of "
                    f"{', '.join(allowed)}"
                )
        return replace(self, outcomes=tuple(states))

    def cancelled(self) -> Query:
        """Keep calls with terminal outcome ``cancelled`` (DOG-58)."""

        return replace(self, outcomes=(OUTCOME_CANCELLED,))

    def failed(self) -> Query:
        """Keep calls with terminal outcome ``failed`` (DOG-58)."""

        return replace(self, outcomes=(OUTCOME_FAILED,))

    def expect(
        self,
        *,
        soft: bool = False,
        since: Optional[int] = None,
        allow_empty: bool = False,
    ) -> Expectation:
        mark = self.since_seq if since is None else since
        trace = self.session._load_trace(allow_empty=allow_empty).filtered(
            api_type=normalize_api_type(self.api_type) if self.api_type else None,
            model=self.model,
            offers_tool=self.offers_tool_name,
            with_tools=self.include_tools,
            since=mark,
            aborted=self.aborted_only,
            outcomes=self.outcomes,
        )
        expectation: Expectation = (
            SoftExpectation(trace, session=self.session)
            if soft
            else Expectation(trace, session=self.session)
        )
        if allow_empty:
            expectation.allow_empty()
        return expectation


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
    port, control_port = allocate_listen_and_control_ports(host, merged_env)
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


def allocate_listen_and_control_ports(
    host: str, env: Mapping[str, str]
) -> tuple[int, int]:
    listen = env.get("LLMENTARY_PORT")
    control = env.get("LLMENTARY_CONTROL_PORT")
    if listen and control:
        return int(listen), int(control)
    if listen:
        port = int(listen)
        return port, free_ports(host, 1, exclude={port})[0]
    if control:
        control_port = int(control)
        return free_ports(host, 1, exclude={control_port})[0], control_port
    allocated = free_ports(host, 2)
    return allocated[0], allocated[1]


def free_port(host: str = "127.0.0.1") -> int:
    return free_ports(host, 1)[0]


def free_ports(
    host: str = "127.0.0.1",
    count: int = 1,
    *,
    exclude: Optional[set[int]] = None,
) -> list[int]:
    """Reserve ``count`` distinct ephemeral ports, held until all are chosen."""

    blocked = set(exclude or ())
    sockets: list[socket.socket] = []
    try:
        while len(sockets) < count:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind((host, 0))
            port = int(sock.getsockname()[1])
            if port in blocked:
                sock.close()
                continue
            blocked.add(port)
            sockets.append(sock)
        return [int(sock.getsockname()[1]) for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


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


KNOWN_SESSIONS_HINT_RE = re.compile(
    r"(?:;\s*)?known sessions:\s*GET\s+/sessions\b",
    re.IGNORECASE,
)
KNOWN_SESSIONS_TAIL_RE = re.compile(
    r"(?:;\s*)?known sessions:.*$",
    re.IGNORECASE | re.DOTALL,
)


def strip_known_sessions_hint(message: str) -> str:
    """Drop the proxy 404 hint ``known sessions: GET /sessions`` (not a list)."""

    cleaned = KNOWN_SESSIONS_HINT_RE.sub("", message)
    return re.sub(r"\s*;\s*;", ";", cleaned).strip(" ;")


def with_known_sessions(
    message: str,
    session_id: str,
    known_ids: Sequence[str],
) -> str:
    """One ``known sessions:`` suffix. Replaces the proxy GET /sessions hint."""

    cleaned = strip_known_sessions_hint(message)
    cleaned = KNOWN_SESSIONS_TAIL_RE.sub("", cleaned).strip(" ;")
    if not cleaned:
        return zero_calls_message(session_id, known_ids)
    listed = ", ".join(known_ids) if known_ids else "(none)"
    return f"{cleaned}; known sessions: {listed}"


def message_lists_known_ids(message: str, known_ids: Sequence[str]) -> bool:
    """True when each known id is listed after ``known sessions:``, not merely mentioned."""

    if not known_ids:
        return True
    marker = "known sessions:"
    position = message.lower().rfind(marker)
    if position < 0:
        return False
    tail = message[position + len(marker) :]
    tokens = {
        token.strip().strip('"')
        for token in tail.replace(";", ",").split(",")
        if token.strip()
    }
    return all(known_id in tokens for known_id in known_ids)


def require_binary_for_tests() -> Path:
    return find_binary()
