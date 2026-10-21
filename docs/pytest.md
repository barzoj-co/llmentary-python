# Pytest plugin

The package registers a `pytest11` entry point named `llmentary`.

## Fixtures

### `llmentary_proxy`

Session-scoped fixture that starts a local Go proxy or attaches to an existing
one.

Attach mode:

```bash
LLMENTARY_PROXY_URL=http://127.0.0.1:7777 \
LLMENTARY_CONTROL_URL=http://127.0.0.1:7778 \
pytest
```

Start mode:

```bash
LLMENTARY_BIN=/path/to/llmentary \
LLMENTARY_UPSTREAM_BASE_URL=https://api.openai.com/v1 \
LLMENTARY_UPSTREAM_API_KEY="$OPENAI_API_KEY" \
pytest
```

The fixture health-checks `GET /healthz` before yielding and terminates the
started process at the end of the test session. Attached proxies are not
terminated. If no upstream URL is set, the proxy defaults to
`openai=https://api.openai.com/v1`. The client does not strip a trailing
`/v1` from `LLMENTARY_UPSTREAM_BASE_URL`.

### `llmentary`

Function-scoped fixture that returns a `Session`: it creates a unique test
session id, exposes OpenAI client helpers, waits for quiescence before
assertions, and deletes the session after the test.

```python
def test_something(llmentary):
    assert llmentary.headers["X-Test-Session"] == llmentary.session_id
```

## Options and environment

| Pytest option | Environment | Purpose |
| --- | --- | --- |
| `--llmentary-bin` | `LLMENTARY_BIN` | Path to proxy binary |
| `--llmentary-proxy-url` | `LLMENTARY_PROXY_URL` | Attach to existing proxy |
| `--llmentary-upstream-url` | `LLMENTARY_UPSTREAM_BASE_URL` | Upstream OpenAI-compatible base URL |
| `--llmentary-upstream-api-key` | `LLMENTARY_UPSTREAM_API_KEY` | Upstream API key |
| `--llmentary-control-token` | `LLMENTARY_CONTROL_TOKEN` | Bearer token for control API |
| `--llmentary-control-url` | `LLMENTARY_CONTROL_URL` | Control API URL when attaching |
| `--llmentary-wait-timeout` | `LLMENTARY_WAIT_TIMEOUT` | Wait timeout (Go duration or seconds; default `30s`) |
| `--llmentary-wait-quiet` | `LLMENTARY_WAIT_QUIET` | Quiet period (Go duration or seconds; default `500ms`) |
| `--llmentary-quiet-for` | `LLMENTARY_QUIET_FOR` | Deprecated alias of `--llmentary-wait-quiet` |

## Custom binary command

If the Go proxy binary needs a different CLI shape, set
`LLMENTARY_START_COMMAND`.

```bash
export LLMENTARY_START_COMMAND='{bin} -listen {listen} -control {control} -upstream {upstream}'
```

Template variables:

- `{bin}`
- `{host}`
- `{port}`
- `{listen}`
- `{control}` / `{control_listen}`
- `{upstream}` (`name=url`)
- `{upstream_url}`
- `{upstream_api_key}`
- `{token}`

## Providing a fake upstream in tests

Projects can provide session-scoped fixtures named `llmentary_upstream_url`,
`llmentary_upstream_api_key`, and `llmentary_control_token`. The plugin uses
them in preference to CLI options.

```python
@pytest.fixture(scope="session")
def llmentary_upstream_url(fake_openai_server):
    return fake_openai_server.url
```

## OpenAI clients

### openai-python

```python
from openai import OpenAI


def test_openai(llmentary):
    client = OpenAI(
        api_key="test-key",
        base_url=llmentary.openai_base_url,
        default_headers=llmentary.headers,
    )
```

When the client is constructed at import time and cannot take headers, set
`OPENAI_BASE_URL` to `llmentary.path_base_url()` inside the test with
`monkeypatch.setenv`, then import the app. A module-level import happens
before the fixture creates the session id. See the proxy
[black-box guide](https://github.com/barzoj-co/llmentary/blob/main/docs/black-box.md).

```python
import importlib


def test_path_session(llmentary, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", llmentary.path_base_url())
    app = importlib.import_module("myapp")
    client = app.client
```

### httpx

```python
import httpx


def test_httpx(llmentary):
    with httpx.Client(base_url=llmentary.openai_base_url, headers=llmentary.headers) as client:
        client.post("/chat/completions", json={"model": "fake", "messages": []})
```

Use the `baggage` property if your app propagates session ids via W3C baggage:

```python
headers = {"baggage": llmentary.baggage}
```

## Troubleshooting

### `BinaryResolutionError` / no bundled proxy

This install has no bundled proxy (sdist or a source tree) or no wheel for
this platform. The pytest plugin re-raises `BinaryResolutionError` so the
message shows up as one block in `pytest -q`.

Fixes, in order:

1. `pip install --only-binary=:all: llmentary`
2. `LLMENTARY_BIN=/path/to/llmentary`
3. Get the binary from npm `@llmentary/<os>-<cpu>` or the GitHub Release
   (proxy v0.2.0); see the proxy
   [docs/binaries.md](https://github.com/barzoj-co/llmentary/blob/main/docs/binaries.md)

Run `python -m llmentary doctor` to print the package version, platform tag,
resolved binary path, and whether this is a wheel or a source install.

The plugin also adds a `pytest_report_header` line:

`llmentary <client>; proxy: <binary --version first line>; binary: <path>`

When no binary resolves, that stays one line: `proxy: not found`. The same
values are available as `llmentary.proxy_version()` and
`llmentary.proxy_binary_path()`.
