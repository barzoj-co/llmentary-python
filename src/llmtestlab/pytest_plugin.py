from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Optional

import pytest

from .client import LabSession, ProxyHandle, new_session_id, start_proxy


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("llmtestlab")
    group.addoption(
        "--llmtestlab-bin",
        action="store",
        default=None,
        help="Path to the llm-test-lab proxy binary (also LLMTESTLAB_BIN).",
    )
    group.addoption(
        "--llmtestlab-proxy-url",
        action="store",
        default=None,
        help="Attach to an existing llm-test-lab proxy URL (also LLMTESTLAB_PROXY_URL).",
    )
    group.addoption(
        "--llmtestlab-upstream-url",
        action="store",
        default=None,
        help="OpenAI-compatible upstream base URL for a started proxy.",
    )
    group.addoption(
        "--llmtestlab-upstream-api-key",
        action="store",
        default=None,
        help="OpenAI-compatible upstream API key for a started proxy.",
    )
    group.addoption(
        "--llmtestlab-control-token",
        action="store",
        default=None,
        help="Bearer token for non-loopback control API access.",
    )
    group.addoption(
        "--llmtestlab-download",
        action="store_true",
        default=False,
        help="Opt in to downloading the proxy from GitHub Releases if not found.",
    )
    group.addoption(
        "--llmtestlab-wait-timeout",
        action="store",
        type=float,
        default=float(os.environ.get("LLMTESTLAB_WAIT_TIMEOUT", "5")),
        help="Seconds to wait for session quiescence before assertions.",
    )
    group.addoption(
        "--llmtestlab-quiet-for",
        action="store",
        type=float,
        default=float(os.environ.get("LLMTESTLAB_QUIET_FOR", "0.2")),
        help="Seconds the session must be quiet before wait returns.",
    )


@pytest.fixture(scope="session")
def llmtestlab_proxy(request: pytest.FixtureRequest) -> Iterator[ProxyHandle]:
    """Start the llm-test-lab proxy or attach to ``LLMTESTLAB_PROXY_URL``."""

    option_bin = request.config.getoption("--llmtestlab-bin")
    if option_bin:
        os.environ["LLMTESTLAB_BIN"] = str(option_bin)

    upstream_url = optional_fixture_value(request, "llmtestlab_upstream_url")
    upstream_api_key = optional_fixture_value(request, "llmtestlab_upstream_api_key")
    token = optional_fixture_value(request, "llmtestlab_control_token")

    configured_upstream_url = upstream_url or request.config.getoption(
        "--llmtestlab-upstream-url"
    )
    configured_upstream_api_key = upstream_api_key or request.config.getoption(
        "--llmtestlab-upstream-api-key"
    )
    configured_token = token or request.config.getoption("--llmtestlab-control-token")

    with start_proxy(
        upstream_url=str(configured_upstream_url or "") or None,
        upstream_api_key=str(configured_upstream_api_key or "") or None,
        proxy_url=request.config.getoption("--llmtestlab-proxy-url"),
        token=str(configured_token or "") or None,
        binary=Path(option_bin) if option_bin else None,
        download=bool(request.config.getoption("--llmtestlab-download")),
    ) as proxy:
        yield proxy


@pytest.fixture
def lab(request: pytest.FixtureRequest, llmtestlab_proxy: ProxyHandle) -> Iterator[LabSession]:
    """Unique llm-test-lab recording session for one pytest test function."""

    test_name = request.node.name.replace("[", "-").replace("]", "")
    session = LabSession(
        session_id=new_session_id(test_name),
        proxy=llmtestlab_proxy,
        wait_timeout=float(request.config.getoption("--llmtestlab-wait-timeout")),
        quiet_for=float(request.config.getoption("--llmtestlab-quiet-for")),
    )
    try:
        yield session
    finally:
        session.cleanup()


def optional_fixture_value(request: pytest.FixtureRequest, name: str) -> Optional[object]:
    try:
        value: object = request.getfixturevalue(name)
        return value
    except pytest.FixtureLookupError:
        return None
