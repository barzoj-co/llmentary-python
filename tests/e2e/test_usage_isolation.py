from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
from e2e_helpers import (
    CHEAP_MODEL,
    REASONING_MODEL,
    REASONING_PROMPT,
    call_openai,
    openai_client,
)

from llmentary import LlmentarySession
from llmentary.client import new_session_id

pytestmark = pytest.mark.e2e

CACHE_PREFIX = ("cache-token LTL-PY-2 " * 400).strip()


def test_cached_tokens_field_is_parsed(llmentary: LlmentarySession) -> None:
    client = openai_client(llmentary)
    prompt = f"{CACHE_PREFIX}\nReply with the single word ping."
    for _ in range(2):
        call_openai(
            lambda: client.chat.completions.create(
                model=CHEAP_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=8,
            )
        )
    trace = llmentary.wait()
    assert len(trace.calls) == 2, trace.trace_summary()
    parsed = [call.cached_tokens for call in trace.calls]
    usage = [dict(call.usage) for call in trace.calls]
    summary = f"cached_tokens={parsed}\nusage={usage}\n{trace.trace_summary()}"
    assert all(value is not None for value in parsed), (
        f"expected cached_tokens to be parsed from usage details\n{summary}"
    )
    assert parsed[0] is not None and parsed[0] >= 0, summary
    assert parsed[1] is not None and parsed[1] > 0, (
        "expected a cache hit (cached_tokens > 0) on the second identical prompt\n"
        f"{summary}"
    )


def test_reasoning_tokens_from_reasoning_model(llmentary: LlmentarySession) -> None:
    client = openai_client(llmentary)
    call_openai(
        lambda: client.responses.create(
            model=REASONING_MODEL,
            input=REASONING_PROMPT,
            reasoning={"effort": "medium"},
            max_output_tokens=400,
        )
    )
    trace = llmentary.query(api_type="responses").expect().trace
    assert trace.calls, trace.trace_summary()
    values = [call.reasoning_tokens for call in trace.calls]
    assert any(value is not None and value > 0 for value in values), (
        f"expected reasoning_tokens > 0 for o4-mini on a multi-step product\n"
        f"reasoning_tokens={values}\nusage={[dict(call.usage) for call in trace.calls]}\n"
        f"{trace.trace_summary()}"
    )


def test_concurrent_sessions_are_isolated(llmentary: LlmentarySession) -> None:
    session_a = LlmentarySession(
        session_id=new_session_id("iso-a"),
        proxy=llmentary.proxy,
        wait_timeout=llmentary.wait_timeout,
        quiet_for=llmentary.quiet_for,
    )
    session_b = LlmentarySession(
        session_id=new_session_id("iso-b"),
        proxy=llmentary.proxy,
        wait_timeout=llmentary.wait_timeout,
        quiet_for=llmentary.quiet_for,
    )
    token_a = "UNIQUE_TOKEN_ALPHA_LTL_PY_2"
    token_b = "UNIQUE_TOKEN_BETA_LTL_PY_2"

    def run(session: LlmentarySession, token: str) -> None:
        client = openai_client(session)
        call_openai(
            lambda: client.chat.completions.create(
                model=CHEAP_MODEL,
                messages=[{"role": "user", "content": f"Echo {token} once."}],
                temperature=0,
                max_tokens=8,
            )
        )

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            future_a = pool.submit(run, session_a, token_a)
            future_b = pool.submit(run, session_b, token_b)
            future_a.result()
            future_b.result()
        trace_a = session_a.wait()
        trace_b = session_b.wait()
        prompt_a = "\n".join(call.prompt_text for call in trace_a.calls)
        prompt_b = "\n".join(call.prompt_text for call in trace_b.calls)
        assert token_a in prompt_a, trace_a.trace_summary()
        assert token_b in prompt_b, trace_b.trace_summary()
        assert token_b not in prompt_a, trace_a.trace_summary()
        assert token_a not in prompt_b, trace_b.trace_summary()
    finally:
        session_a.cleanup()
        session_b.cleanup()
