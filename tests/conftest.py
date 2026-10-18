from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from fake_upstream import FakeOpenAIServer


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("LLMTESTLAB_BIN") or os.environ.get("LLMTESTLAB_PROXY_URL"):
        return
    skip_integration = pytest.mark.skip(
        reason="requires real llm-test-lab Go binary via LLMTESTLAB_BIN"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


@pytest.fixture(scope="session")
def fake_openai_server() -> Iterator[FakeOpenAIServer]:
    server = FakeOpenAIServer().start()
    try:
        yield server
    finally:
        server.close()


@pytest.fixture(scope="session")
def llmtestlab_upstream_url(request: pytest.FixtureRequest) -> str:
    if os.environ.get("LLMTESTLAB_E2E") == "1":
        upstream_url = os.environ.get("LLMTESTLAB_UPSTREAM_BASE_URL")
        if not upstream_url:
            pytest.fail("LLMTESTLAB_UPSTREAM_BASE_URL is required when LLMTESTLAB_E2E=1")
        assert upstream_url is not None
        return upstream_url
    server: FakeOpenAIServer = request.getfixturevalue("fake_openai_server")
    return server.url


@pytest.fixture(scope="session")
def llmtestlab_upstream_api_key() -> str:
    if os.environ.get("LLMTESTLAB_E2E") == "1":
        api_key = os.environ.get("LLMTESTLAB_UPSTREAM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            pytest.fail("OPENAI_API_KEY is required when LLMTESTLAB_E2E=1")
        assert api_key is not None
        return api_key
    return "test-api-key"
