from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Any, Optional
from urllib.parse import urlencode

from .types import SessionTrace, first_present

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


class WaitTimeoutError(TimeoutError):
    """Raised when the proxy wait endpoint times out before the session is quiet."""

    def __init__(
        self,
        message: str,
        *,
        session_id: str,
        timeout: float,
        calls: int,
        active: int,
        min_calls: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.session_id = session_id
        self.timeout = timeout
        self.calls = calls
        self.active = active
        self.min_calls = min_calls


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


def count_active_calls(payload: Mapping[str, Any], trace: SessionTrace) -> int:
    for key in ("active", "in_flight", "inflight", "active_calls", "still_active"):
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    active = 0
    for call in trace.calls:
        ended = first_present(call.raw, "ended_at", "endedAt")
        if ended in (None, ""):
            active += 1
    return active


def wait_timeout_error(
    session_id: str,
    timeout: float,
    payload: Optional[Mapping[str, Any]] = None,
) -> WaitTimeoutError:
    raw = payload or {}
    error = raw.get("error") if isinstance(raw.get("error"), Mapping) else {}
    error_mapping = error if isinstance(error, Mapping) else {}
    nested_session = raw.get("session")
    session_payload = nested_session if isinstance(nested_session, Mapping) else raw
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
    active = count_active_calls(session_payload, trace)
    message = (
        f"wait timed out after {format_go_duration(timeout)} on session {session_id} "
        f"({calls} calls so far, still active: {active}). "
        f"Raise {TIMEOUT_ENV} if the graph needs more time."
    )
    return WaitTimeoutError(
        message,
        session_id=session_id,
        timeout=timeout,
        calls=calls,
        active=active,
        min_calls=min_calls,
    )
