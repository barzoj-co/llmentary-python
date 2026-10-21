from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import pytest
from e2e_helpers import (
    CHEAP_MODEL,
    REASONING_MODEL,
    REASONING_PROMPT,
    call_openai,
    openai_client,
)

from llmentary import Session
from llmentary.client import new_session_id

pytestmark = pytest.mark.e2e

CACHE_PREFIX = ("cache-token LTL-PY-2 " * 400).strip()


def test_cached_tokens_field_is_parsed(llmentary: Session) -> None:
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


def test_reasoning_tokens_from_reasoning_model(llmentary: Session) -> None:
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


def test_concurrent_sessions_are_isolated(llmentary: Session) -> None:
    session_a = Session(
        session_id=new_session_id("iso-a"),
        proxy=llmentary.proxy,
        wait_timeout=llmentary.wait_timeout,
        quiet_for=llmentary.quiet_for,
    )
    session_b = Session(
        session_id=new_session_id("iso-b"),
        proxy=llmentary.proxy,
        wait_timeout=llmentary.wait_timeout,
        quiet_for=llmentary.quiet_for,
    )
    token_a = "UNIQUE_TOKEN_ALPHA_LTL_PY_2"
    token_b = "UNIQUE_TOKEN_BETA_LTL_PY_2"

    def run(session: Session, token: str) -> None:
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


def test_isolated_from_path_clients(llmentary: Session) -> None:
    from openai import OpenAI

    other = Session(
        session_id=new_session_id("iso-path-b"),
        proxy=llmentary.proxy,
        wait_timeout=llmentary.wait_timeout,
        quiet_for=llmentary.quiet_for,
    )
    shared = (
        "You may mention get_current_weather and AMSTERDAM_CANAL in instructions. "
        "Reply with one word."
    )

    def path_client(session: Session) -> OpenAI:
        return OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=session.path_base_url(),
            max_retries=0,
        )

    try:
        call_openai(
            lambda: path_client(llmentary).chat.completions.create(
                model=CHEAP_MODEL,
                messages=[
                    {"role": "system", "content": shared},
                    {"role": "user", "content": "Reply with the single word PARIS_BRIDGE."},
                ],
                temperature=0,
                max_tokens=16,
            )
        )
        call_openai(
            lambda: path_client(other).chat.completions.create(
                model=CHEAP_MODEL,
                messages=[
                    {"role": "system", "content": shared},
                    {
                        "role": "user",
                        "content": "Reply with the single word AMSTERDAM_CANAL.",
                    },
                ],
                temperature=0,
                max_tokens=16,
            )
        )
        llmentary.expect().isolated_from(other)
        other.expect().isolated_from(llmentary)
        call = llmentary.expect().trace.calls[0]
        assert call.user_texts == ["Reply with the single word PARIS_BRIDGE."]
        assert call.session_id == llmentary.session_id
        assert "get_current_weather" not in call.user_texts
    finally:
        other.cleanup()


def test_isolated_from_same_prompt_and_history_replay(llmentary: Session) -> None:
    other = Session(
        session_id=new_session_id("iso-same-b"),
        proxy=llmentary.proxy,
        wait_timeout=llmentary.wait_timeout,
        quiet_for=llmentary.quiet_for,
    )
    replay = Session(
        session_id=new_session_id("iso-replay"),
        proxy=llmentary.proxy,
        wait_timeout=llmentary.wait_timeout,
        quiet_for=llmentary.quiet_for,
    )
    prompt = "What is my biological age? Reply with one integer."
    try:
        for session in (llmentary, other):
            client = openai_client(session)
            call_openai(
                lambda client=client: client.chat.completions.create(
                    model=CHEAP_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=16,
                )
            )
        llmentary.wait(timeout=30, quiet=0.5, min_calls=1)
        other.wait(timeout=30, quiet=0.5, min_calls=1)
        llmentary.expect().isolated_from(other)
        other.expect().isolated_from(llmentary)

        first = llmentary.trace().calls[0].user_texts[0]
        client = openai_client(replay)
        call_openai(
            lambda: client.chat.completions.create(
                model=CHEAP_MODEL,
                messages=[
                    {"role": "user", "content": first},
                    {"role": "assistant", "content": "42"},
                    {"role": "user", "content": "Explain that number in one word."},
                ],
                temperature=0,
                max_tokens=16,
            )
        )
        replay.wait(timeout=30, quiet=0.5, min_calls=1)
        with pytest.raises(AssertionError, match="user text"):
            replay.expect().isolated_from(llmentary)
    finally:
        other.cleanup()
        replay.cleanup()


def test_isolated_from_never_created_other_fails_without_allow_empty(
    llmentary: Session,
) -> None:
    client = openai_client(llmentary)
    call_openai(
        lambda: client.chat.completions.create(
            model=CHEAP_MODEL,
            messages=[{"role": "user", "content": "Reply with the single word ping."}],
            temperature=0,
            max_tokens=8,
        )
    )
    ghost = Session(
        session_id=new_session_id("ghost-other"),
        proxy=llmentary.proxy,
        wait_timeout=llmentary.wait_timeout,
        quiet_for=llmentary.quiet_for,
    )
    try:
        with pytest.raises(
            AssertionError,
            match=r"no calls recorded for session .+ \(isolated_from needs calls on both sides\)",
        ):
            llmentary.expect().isolated_from(ghost)
        llmentary.expect().allow_empty().isolated_from(ghost)
    finally:
        ghost.cleanup()


def test_later_call_two_new_user_messages_is_not_a_leak(llmentary: Session) -> None:
    peer = Session(
        session_id=new_session_id("iso-q"),
        proxy=llmentary.proxy,
        wait_timeout=llmentary.wait_timeout,
        quiet_for=llmentary.quiet_for,
    )
    opening = "Opening question number one LTL-PY-DOG-33"
    second = "Second message sent by P itself LTL-PY-DOG-33"
    third = "Third message sent by P itself LTL-PY-DOG-33"
    try:
        client_p = openai_client(llmentary)
        call_openai(
            lambda: client_p.chat.completions.create(
                model=CHEAP_MODEL,
                messages=[{"role": "user", "content": opening}],
                temperature=0,
                max_tokens=8,
            )
        )
        call_openai(
            lambda: client_p.chat.completions.create(
                model=CHEAP_MODEL,
                messages=[
                    {"role": "user", "content": opening},
                    {"role": "user", "content": second},
                    {"role": "user", "content": third},
                ],
                temperature=0,
                max_tokens=8,
            )
        )
        client_q = openai_client(peer)
        call_openai(
            lambda: client_q.chat.completions.create(
                model=CHEAP_MODEL,
                messages=[{"role": "user", "content": second}],
                temperature=0,
                max_tokens=8,
            )
        )
        llmentary.wait(timeout=30, quiet=0.5, min_calls=2)
        peer.wait(timeout=30, quiet=0.5, min_calls=1)
        llmentary.expect().isolated_from(peer)
        peer.expect().isolated_from(llmentary)
    finally:
        peer.cleanup()
