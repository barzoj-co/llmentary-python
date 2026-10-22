from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlencode

from .types import SessionTrace

DEFAULT_WAIT_TIMEOUT = 30.0
DEFAULT_WAIT_QUIET = 0.5
TIMEOUT_ENV = "LLMENTARY_WAIT_TIMEOUT"
QUIET_ENV = "LLMENTARY_WAIT_QUIET"
QUIET_ENV_LEGACY = "LLMENTARY_QUIET_FOR"

_DURATION_RE = re.compile(
    r"^(?P<value>(?:\d+(?:\.\d*)?|\.\d+))(?P<unit>ns|us|µs|μs|ms|s|m|h)?$"
)
_UNIT_SECONDS = {
    None: 1.0,
    "ns": 1e-9,
    "us": 1e-6,
    "µs": 1e-6,
    "μs": 1e-6,
    "ms": 1e-3,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
}


@dataclass(frozen=True)
class InFlightCall:
    """Settled-safe snapshot of a call that has not finished (504 ``in_flight``)."""

    call_id: str = ""
    seq: Optional[int] = None
    method: str = ""
    path: str = ""
    phase: Optional[str] = None
    elapsed_ms: Optional[int] = None
    bytes_received: Optional[int] = None
    http_status: Optional[int] = None
    api_family: Optional[str] = None
    upstream: Optional[str] = None
    started_at: Optional[str] = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        parts: list[str] = []
        if self.seq is not None:
            parts.append(f"#{self.seq}")
        elif self.call_id:
            parts.append(self.call_id)
        if self.method:
            parts.append(self.method)
        if self.path:
            parts.append(self.path)
        if self.phase:
            parts.append(self.phase)
        if self.http_status is not None:
            parts.append(f"http {self.http_status}")
        if self.elapsed_ms is not None:
            parts.append(f"{self.elapsed_ms}ms")
        return " ".join(parts) if parts else "in-flight call"


class WaitTimeoutError(TimeoutError):
    """Raised when the proxy wait endpoint times out before the session is quiet."""

    def __init__(
        self,
        message: str,
        *,
        session_id: str,
        timeout: float,
        calls: int,
        active: Optional[int] = None,
        min_calls: Optional[int] = None,
        in_flight: Optional[Sequence[InFlightCall]] = None,
    ) -> None:
        super().__init__(message)
        self.session_id = session_id
        self.timeout = timeout
        self.calls = calls
        self.active = active
        self.min_calls = min_calls
        self.in_flight = None if in_flight is None else tuple(in_flight)


def format_go_duration(seconds: float) -> str:
    """Format seconds as a Go duration with millisecond precision.

    Whole seconds become ``30s`` / ``90s``. Fractional values become ``500ms`` /
    ``200ms`` so the query string is never ``0.2s`` or scientific notation.
    """

    millis = round(float(seconds) * 1000)
    if millis % 1000 == 0:
        return f"{millis // 1000}s"
    return f"{millis}ms"


def parse_duration(value: str, *, name: str) -> float:
    """Parse a Go duration or a bare number of seconds.

    Accepts ``45``, ``45s``, ``0.2``, and ``200ms``. Invalid values raise
    ``ValueError`` naming ``name`` (typically an env var or CLI option).
    """

    text = str(value).strip()
    match = _DURATION_RE.fullmatch(text)
    if match is None:
        raise ValueError(
            f"invalid {name} value {value!r}: expected a number of seconds or a "
            "Go duration such as 45s or 200ms"
        )
    seconds = float(match.group("value")) * _UNIT_SECONDS[match.group("unit")]
    if seconds < 0:
        raise ValueError(
            f"invalid {name} value {value!r}: duration must be non-negative"
        )
    return seconds


def duration_from_env(
    name: str,
    default: float,
    *,
    env: Optional[Mapping[str, str]] = None,
) -> float:
    env_map = os.environ if env is None else env
    raw = env_map.get(name)
    if raw is None or raw == "":
        return default
    return parse_duration(raw, name=name)


def default_wait_timeout(*, env: Optional[Mapping[str, str]] = None) -> float:
    return duration_from_env(TIMEOUT_ENV, DEFAULT_WAIT_TIMEOUT, env=env)


def default_wait_quiet(*, env: Optional[Mapping[str, str]] = None) -> float:
    env_map = os.environ if env is None else env
    configured = env_map.get(QUIET_ENV)
    if configured is not None and configured != "":
        return parse_duration(configured, name=QUIET_ENV)
    legacy = env_map.get(QUIET_ENV_LEGACY)
    if legacy is not None and legacy != "":
        return parse_duration(legacy, name=QUIET_ENV_LEGACY)
    return DEFAULT_WAIT_QUIET


def resolve_wait_timeout(timeout: Optional[float]) -> float:
    if timeout is not None:
        return float(timeout)
    return default_wait_timeout()


def resolve_wait_quiet(
    quiet: Optional[float],
    quiet_for: Optional[float] = None,
) -> float:
    if quiet is not None:
        return float(quiet)
    if quiet_for is not None:
        return float(quiet_for)
    return default_wait_quiet()


def wait_query_string(
    timeout: float,
    quiet: float,
    min_calls: Optional[int] = None,
) -> str:
    params: dict[str, str] = {
        "timeout": format_go_duration(timeout),
        "quiet": format_go_duration(quiet),
    }
    if min_calls is not None:
        params["min_calls"] = str(int(min_calls))
    return urlencode(params)


def explicit_active_count(*payloads: Mapping[str, Any]) -> Optional[int]:
    """Return an explicit ``active`` integer, or ``None`` when the body omits it.

    After proxy 0.4.1 (DOG-36) in-flight calls are not in ``calls``, so counting
    missing ``ended_at`` is always 0 and must not be used.
    """

    for payload in payloads:
        for key in ("active", "active_calls", "still_active"):
            value = payload.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
    return None


def _optional_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    return None


def _optional_str(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value != "" else None


def in_flight_call_from_raw(value: Any) -> Optional[InFlightCall]:
    if not isinstance(value, Mapping):
        return None
    return InFlightCall(
        call_id=_optional_str(value.get("call_id")) or "",
        seq=_optional_int(value.get("seq")),
        method=_optional_str(value.get("method")) or "",
        path=_optional_str(value.get("path")) or "",
        phase=_optional_str(value.get("phase")),
        elapsed_ms=_optional_int(value.get("elapsed_ms")),
        bytes_received=_optional_int(value.get("bytes_received")),
        http_status=_optional_int(value.get("http_status")),
        api_family=_optional_str(value.get("api_family")),
        upstream=_optional_str(value.get("upstream")),
        started_at=_optional_str(value.get("started_at")),
        raw=value,
    )


def parse_in_flight_calls(*payloads: Mapping[str, Any]) -> Optional[tuple[InFlightCall, ...]]:
    """Return ``in_flight`` when the field is present; ``None`` for older proxies.

    An empty array is a present field (no running calls), not a missing one.
    JSON ``null`` is treated as empty. A non-sequence value is unknown
    (``None``) so the count is not invented as 0. Non-object items are skipped.
    """

    for payload in payloads:
        if "in_flight" not in payload:
            continue
        raw = payload.get("in_flight")
        if raw is None:
            return ()
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            return None
        items: list[InFlightCall] = []
        for item in raw:
            parsed = in_flight_call_from_raw(item)
            if parsed is not None:
                items.append(parsed)
        return tuple(items)
    return None


def _payload_layers(
    payload: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    error = payload.get("error")
    error_mapping = error if isinstance(error, Mapping) else {}
    nested_session = payload.get("session")
    session_payload = nested_session if isinstance(nested_session, Mapping) else payload
    return error_mapping, session_payload, payload


def count_active_calls(
    payload: Mapping[str, Any],
    trace: SessionTrace,
    *,
    fallback: Optional[int] = None,
) -> Optional[int]:
    _ = trace
    error_mapping, session_payload, top = _payload_layers(payload)
    active = explicit_active_count(error_mapping, session_payload, top)
    if active is not None:
        return active
    in_flight = parse_in_flight_calls(error_mapping, session_payload, top)
    if in_flight is not None:
        return len(in_flight)
    return fallback


def wait_timeout_error(
    session_id: str,
    timeout: float,
    payload: Optional[Mapping[str, Any]] = None,
    *,
    list_active: Optional[int] = None,
) -> WaitTimeoutError:
    raw = payload or {}
    error_mapping, session_payload, _top = _payload_layers(raw)
    trace = SessionTrace.from_payload(session_id, session_payload)
    calls_seen = error_mapping.get("calls_seen")
    calls = (
        int(calls_seen)
        if isinstance(calls_seen, int) and not isinstance(calls_seen, bool)
        else len(trace.calls)
    )
    min_calls_value = error_mapping.get("min_calls")
    min_calls = (
        int(min_calls_value)
        if isinstance(min_calls_value, int) and not isinstance(min_calls_value, bool)
        else None
    )
    in_flight = parse_in_flight_calls(error_mapping, session_payload, raw)
    active = count_active_calls(raw, trace, fallback=list_active)
    so_far = f"{calls} calls so far"
    if in_flight:
        listed = "; ".join(item.describe() for item in in_flight)
        count = active if active is not None else len(in_flight)
        detail = f"{so_far}, still active: {count}: {listed}"
    elif active is not None:
        detail = f"{so_far}, still active: {active}"
    else:
        detail = so_far
    message = (
        f"wait timed out after {format_go_duration(timeout)} on session {session_id} "
        f"({detail}). "
        f"Raise {TIMEOUT_ENV} if the graph needs more time."
    )
    return WaitTimeoutError(
        message,
        session_id=session_id,
        timeout=timeout,
        calls=calls,
        active=active,
        min_calls=min_calls,
        in_flight=in_flight,
    )
