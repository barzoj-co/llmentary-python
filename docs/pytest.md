# Pytest plugin

The package registers a `pytest11` entry point named `llmtestlab`.

## Fixtures

### `llmtestlab_proxy`

Session-scoped fixture that starts a local Go proxy or attaches to an existing
one.

Attach mode:

```bash
LLMTESTLAB_PROXY_URL=http://127.0.0.1:8080 pytest
```

Start mode:

```bash
LLMTESTLAB_BIN=/path/to/llmtestlab \
LLMTESTLAB_UPSTREAM_BASE_URL=https://api.openai.com/v1 \
LLMTESTLAB_UPSTREAM_API_KEY="$OPENAI_API_KEY" \
pytest
```

The fixture health-checks `GET /healthz` before yielding and terminates the
started process at the end of the test session. Attached proxies are not
terminated.

### `lab`

Function-scoped fixture that creates a unique test session id, exposes OpenAI
client helpers, waits for quiescence before assertions, and deletes the session
after the test.

```python
def test_something(lab):
    assert lab.headers["X-Test-Session"] == lab.session_id
```

## Options and environment

| Pytest option | Environment | Purpose |
| --- | --- | --- |
| `--llmtestlab-bin` | `LLMTESTLAB_BIN` | Path to proxy binary |
| `--llmtestlab-proxy-url` | `LLMTESTLAB_PROXY_URL` | Attach to existing proxy |
| `--llmtestlab-upstream-url` | `LLMTESTLAB_UPSTREAM_BASE_URL` | Upstream OpenAI-compatible base URL |
| `--llmtestlab-upstream-api-key` | `LLMTESTLAB_UPSTREAM_API_KEY` | Upstream API key |
| `--llmtestlab-control-token` | `LLMTESTLAB_CONTROL_TOKEN` | Bearer token for control API |
| `--llmtestlab-download` | `LLMTESTLAB_DOWNLOAD=1` | Opt in to GitHub Releases download |
| `--llmtestlab-wait-timeout` | `LLMTESTLAB_WAIT_TIMEOUT` | Session wait timeout in seconds |
| `--llmtestlab-quiet-for` | `LLMTESTLAB_QUIET_FOR` | Quiet period before wait returns |

## Custom binary command

If the Go proxy binary needs a different CLI shape, set
`LLMTESTLAB_START_COMMAND`.

```bash
export LLMTESTLAB_START_COMMAND='{bin} serve --addr {listen} --upstream {upstream_url}'
```

Template variables:

- `{bin}`
- `{host}`
- `{port}`
- `{listen}`
- `{upstream_url}`
- `{upstream_api_key}`
- `{token}`

## Providing a fake upstream in tests

Projects can provide session-scoped fixtures named `llmtestlab_upstream_url`,
`llmtestlab_upstream_api_key`, and `llmtestlab_control_token`. The plugin uses
them in preference to CLI options.

```python
@pytest.fixture(scope="session")
def llmtestlab_upstream_url(fake_openai_server):
    return fake_openai_server.url
```

## OpenAI clients

### openai-python

```python
from openai import OpenAI


def test_openai(lab):
    client = OpenAI(
        api_key="test-key",
        base_url=lab.openai_base_url,
        default_headers=lab.headers,
    )
```

### httpx

```python
import httpx


def test_httpx(lab):
    with httpx.Client(base_url=lab.openai_base_url, headers=lab.headers) as client:
        client.post("/chat/completions", json={"model": "fake", "messages": []})
```

Use the `baggage` property if your app propagates session ids via W3C baggage:

```python
headers = {"baggage": lab.baggage}
```
