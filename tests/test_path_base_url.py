from __future__ import annotations

from typing import Optional
from unittest.mock import Mock

import pytest

from llmentary.client import ProxyHandle, Session
from llmentary.types import SessionTrace


@pytest.mark.parametrize(
    ("session_id", "upstream", "suffix"),
    [
        ("t1", None, "/s/t1/v1"),
        ("t 1", "openrouter", "/s/t%201/u/openrouter"),
        ("id/with/slash", None, "/s/id%2Fwith%2Fslash/v1"),
    ],
)
def test_path_base_url_shared_table(
    session_id: str, upstream: Optional[str], suffix: str
) -> None:
    proxy = ProxyHandle(url="http://127.0.0.1:7777", control=Mock())
    session = Session(session_id=session_id, proxy=proxy)
    assert session.path_base_url(upstream) == f"http://127.0.0.1:7777{suffix}"


def test_trace_summary_uses_seq_not_zero_index() -> None:
    trace = SessionTrace.from_payload(
        "sess",
        {
            "calls": [
                {"api_type": "chat_completions", "request": {"messages": []}},
                {"seq": 4, "api_type": "responses", "request": {"input": "hi"}},
            ]
        },
    )
    summary = trace.trace_summary()
    assert "[1]" in summary
    assert "[4]" in summary
    assert "[0]" not in summary
    assert trace.calls[0].sequence_number() == 1
    assert trace.calls[1].sequence_number() == 4
    assert trace.calls[0].call_number() == "1"
    assert trace.calls[1].call_number() == "4"
