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
def llmtestlab_upstream_url(fake_openai_server: FakeOpenAIServer) -> str:
    return fake_openai_server.url


@pytest.fixture(scope="session")
def llmtestlab_upstream_api_key() -> str:
    return "test-api-key"
