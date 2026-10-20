from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from fake_upstream import FakeOpenAIServer

pytest_plugins = ["pytester"]


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    has_proxy = bool(os.environ.get("LLMENTARY_BIN") or os.environ.get("LLMENTARY_PROXY_URL"))
    skip_integration = pytest.mark.skip(
        reason="requires real llmentary Go binary via LLMENTARY_BIN"
    )
    skip_e2e = pytest.mark.skip(reason="set LLMENTARY_E2E=1 to run real OpenAI E2E tests")
    for item in items:
        if "integration" in item.keywords and not has_proxy:
            item.add_marker(skip_integration)
        if "e2e" in item.keywords and os.environ.get("LLMENTARY_E2E") != "1":
            item.add_marker(skip_e2e)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if os.environ.get("LLMENTARY_E2E") != "1":
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:
        return
    skipped = [
        report
        for report in reporter.stats.get("skipped", [])
        if _report_is_e2e(report)
    ]
    if skipped:
        names = [getattr(report, "nodeid", str(report)) for report in skipped]
        reporter.write_line(f"E2E forbids skipped tests when LLMENTARY_E2E=1: {names}")
        if exitstatus == pytest.ExitCode.OK:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED


def _report_is_e2e(report: object) -> bool:
    nodeid = str(getattr(report, "nodeid", ""))
    if "tests/e2e/" in nodeid or nodeid.startswith("tests/e2e"):
        return True
    keywords = getattr(report, "keywords", {})
    return "e2e" in keywords


@pytest.fixture(scope="session")
def fake_openai_server() -> Iterator[FakeOpenAIServer]:
    server = FakeOpenAIServer().start()
    try:
        yield server
    finally:
        server.close()


@pytest.fixture(scope="session")
def llmentary_upstream_url(request: pytest.FixtureRequest) -> str:
    if os.environ.get("LLMENTARY_E2E") == "1":
        upstream_url = os.environ.get("LLMENTARY_UPSTREAM_BASE_URL")
        if not upstream_url:
            pytest.fail("LLMENTARY_UPSTREAM_BASE_URL is required when LLMENTARY_E2E=1")
        assert upstream_url is not None
        return upstream_url
    server: FakeOpenAIServer = request.getfixturevalue("fake_openai_server")
    return server.url


@pytest.fixture(scope="session")
def llmentary_upstream_api_key() -> str:
    if os.environ.get("LLMENTARY_E2E") == "1":
        api_key = os.environ.get("LLMENTARY_UPSTREAM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            pytest.fail("OPENAI_API_KEY is required when LLMENTARY_E2E=1")
        assert api_key is not None
        return api_key
    return "test-api-key"
