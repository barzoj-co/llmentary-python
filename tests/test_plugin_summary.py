from __future__ import annotations

from unittest.mock import Mock

import httpx

from llmentary.client import Session, SessionNotFound
from llmentary.pytest_plugin import green_run_call_summary
from llmentary.types import SessionTrace


def test_green_run_call_summary_prints_seq() -> None:
    session = Mock(spec=Session)
    session.trace.return_value = SessionTrace.from_payload(
        "green",
        {"calls": [{"seq": 3, "request": {"messages": [{"role": "user", "content": "hi"}]}}]},
    )
    summary = green_run_call_summary(session)
    assert summary is not None
    assert "#3" in summary
    assert "llmentary session green" in summary


def test_green_run_call_summary_skips_empty_or_missing() -> None:
    empty = Mock(spec=Session)
    empty.trace.return_value = SessionTrace.from_payload("empty", {"calls": []})
    assert green_run_call_summary(empty) is None

    missing = Mock(spec=Session)
    missing.trace.side_effect = httpx.HTTPStatusError(
        "missing",
        request=httpx.Request("GET", "http://127.0.0.1/sessions/x"),
        response=httpx.Response(404),
    )
    assert green_run_call_summary(missing) is None

    gone = Mock(spec=Session)
    gone.trace.side_effect = SessionNotFound("gone")
    assert green_run_call_summary(gone) is None
