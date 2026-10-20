from __future__ import annotations

import argparse
import os
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Optional

import pytest

from .client import ProxyHandle, Session, new_session_id, start_proxy
from .wait import default_wait_quiet, default_wait_timeout, parse_duration


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("llmentary")
    group.addoption(
        "--llmentary-bin",
        action="store",
        default=None,
        help="Path to the llmentary proxy binary (also LLMENTARY_BIN).",
    )
    group.addoption(
        "--llmentary-proxy-url",
        action="store",
        default=None,
        help="Attach to an existing llmentary proxy URL (also LLMENTARY_PROXY_URL).",
    )
    group.addoption(
        "--llmentary-upstream-url",
        action="store",
        default=None,
        help="OpenAI-compatible upstream base URL for a started proxy.",
    )
    group.addoption(
        "--llmentary-upstream-api-key",
        action="store",
        default=None,
        help="OpenAI-compatible upstream API key for a started proxy.",
    )
    group.addoption(
        "--llmentary-control-token",
        action="store",
        default=None,
        help="Bearer token for non-loopback control API access.",
    )
    group.addoption(
        "--llmentary-control-url",
        action="store",
        default=None,
        help="Control API URL when attaching to an existing proxy (also LLMENTARY_CONTROL_URL).",
    )
    group.addoption(
        "--llmentary-wait-timeout",
        action="store",
        type=_cli_duration("LLMENTARY_WAIT_TIMEOUT"),
        default=default_wait_timeout(),
        help="Wait timeout (Go duration or seconds; default 30s, also LLMENTARY_WAIT_TIMEOUT).",
    )
    group.addoption(
        "--llmentary-wait-quiet",
        "--llmentary-quiet-for",
        action="store",
        dest="llmentary_wait_quiet",
        type=_cli_duration("LLMENTARY_WAIT_QUIET"),
        default=default_wait_quiet(),
        help=(
            "Quiet period before wait returns (Go duration or seconds; default 500ms, "
            "also LLMENTARY_WAIT_QUIET). --llmentary-quiet-for is a deprecated alias."
        ),
    )


@pytest.fixture(scope="session")
def llmentary_proxy(request: pytest.FixtureRequest) -> Iterator[ProxyHandle]:
    """Start the llmentary proxy or attach to ``LLMENTARY_PROXY_URL``."""

    option_bin = request.config.getoption("--llmentary-bin")
    if option_bin:
        os.environ["LLMENTARY_BIN"] = str(option_bin)

    upstream_url = optional_fixture_value(request, "llmentary_upstream_url")
    upstream_api_key = optional_fixture_value(request, "llmentary_upstream_api_key")
    token = optional_fixture_value(request, "llmentary_control_token")

    configured_upstream_url = upstream_url or request.config.getoption(
        "--llmentary-upstream-url"
    )
    configured_upstream_api_key = upstream_api_key or request.config.getoption(
        "--llmentary-upstream-api-key"
    )
    configured_token = token or request.config.getoption("--llmentary-control-token")
    control_url = request.config.getoption("--llmentary-control-url")
    if control_url:
        os.environ["LLMENTARY_CONTROL_URL"] = str(control_url)

    with start_proxy(
        upstream_url=str(configured_upstream_url or "") or None,
        upstream_api_key=str(configured_upstream_api_key or "") or None,
        proxy_url=request.config.getoption("--llmentary-proxy-url"),
        token=str(configured_token or "") or None,
        binary=Path(option_bin) if option_bin else None,
    ) as proxy:
        yield proxy


@pytest.fixture
def llmentary(
    request: pytest.FixtureRequest, llmentary_proxy: ProxyHandle
) -> Iterator[Session]:
    """Unique llmentary recording session for one pytest test function."""

    test_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", request.node.name).strip("-.")
    session = Session(
        session_id=new_session_id(test_name),
        proxy=llmentary_proxy,
        wait_timeout=float(request.config.getoption("--llmentary-wait-timeout")),
        quiet_for=float(request.config.getoption("llmentary_wait_quiet")),
    )
    try:
        yield session
    finally:
        session.cleanup()


def _cli_duration(name: str) -> Callable[[str], float]:
    def parse(value: str) -> float:
        try:
            return parse_duration(value, name=name)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc

    return parse


def optional_fixture_value(request: pytest.FixtureRequest, name: str) -> Optional[object]:
    try:
        value: object = request.getfixturevalue(name)
        return value
    except pytest.FixtureLookupError:
        return None
