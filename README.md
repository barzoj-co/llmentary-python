# llmentary for Python

[![CI](https://github.com/barzoj-co/llmentary-python/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/barzoj-co/llmentary-python/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/barzoj-co/llmentary-python/graph/badge.svg)](https://codecov.io/gh/barzoj-co/llmentary-python)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/barzoj-co/llmentary-python/badge)](https://scorecard.dev/viewer/?uri=github.com/barzoj-co/llmentary-python)
[![License: MIT](https://img.shields.io/github/license/barzoj-co/llmentary-python)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/llmentary)](https://pypi.org/project/llmentary/)
[![Python versions](https://img.shields.io/pypi/pyversions/llmentary)](https://pypi.org/project/llmentary/)

Python client and pytest plugin for [llmentary](https://github.com/barzoj-co/llmentary),
a recording proxy for testing LLM applications. Your test sends OpenAI SDK
requests through the proxy to the real provider, then asserts on what was
recorded: tools called and their arguments, call count, tokens, latency, errors.

Documentation: **https://barzoj-co.github.io/llmentary/**

## Install

```bash
pip install "llmentary[pytest,openai]"
```

Requires Python 3.9 or newer. Published platform wheels bundle proxy v0.5.3
for Linux, macOS and Windows; `python -m llmentary doctor` shows which binary
is used.

## Quickstart

```python
import os
from openai import OpenAI

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def test_model_calls_weather_tool(llmentary):
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=llmentary.openai_base_url,
        default_headers=llmentary.headers,
        max_retries=0,
    )
    client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "What's the weather in Paris?"}],
        tools=[WEATHER_TOOL],
        tool_choice="required",
    )

    (
        llmentary.expect()
        .call_count(1)
        .tool_called("get_weather")
        .args_subset("get_weather", {"city": "Paris"})
        .no_errors()
        .tokens(max_total=1000)
    )
```

```console
$ OPENAI_API_KEY=sk-... pytest -q
1 passed in 2.49s
```

The `llmentary` fixture starts the proxy once per pytest run and gives each
test its own session. The SDK's API key is forwarded to the provider unchanged,
so it must be a real key; a placeholder such as `test-key` is rejected by the
provider.

## What you can assert

- Tools: `tool_called`, `not_called`, `args_subset`, `order`, `tool_result_in_prompt`.
- Requests: `model`, `request_param`, `prompt_contains`.
- Reliability: `call_count`, `no_retries`, `no_errors`, `aborted`.
- Budgets: `tokens`, `latency`, `ttft`.
- Isolation between sessions: `isolated_from`
  ([rules](https://github.com/barzoj-co/llmentary/blob/main/docs/isolation-spec.md),
  [test vectors](tests/testdata/isolation/)).

All assertions: [reference](https://barzoj-co.github.io/llmentary/reference/assertions/).
Python API and pytest options: [docs/api.md](docs/api.md), [docs/pytest.md](docs/pytest.md).

<!-- llmentary-repos:start -->
## llmentary repositories

| Repository | What |
|---|---|
| [llmentary](https://github.com/barzoj-co/llmentary) | Recording proxy binary (Go) |
| [llmentary-go](https://github.com/barzoj-co/llmentary-go) | Go client for `go test` |
| [llmentary-python](https://github.com/barzoj-co/llmentary-python) | Python client and pytest plugin |
| [llmentary-js](https://github.com/barzoj-co/llmentary-js) | TypeScript/JavaScript client for Vitest and Jest |
<!-- llmentary-repos:end -->

## License

Licensed under MIT, see LICENSE
