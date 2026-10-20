# API reference

## Core objects

### `ControlClient`

```python
from llmentary import ControlClient

control = ControlClient("http://127.0.0.1:7778", token=None)
control.health()
trace = control.wait_session("session-id", timeout=5, quiet_for=0.2)
control.delete_session("session-id")
otlp_bytes = control.otlp("session-id")
```

The Go proxy serves OpenAI-compatible traffic on `-listen` and the control API
on a separate `-control` address (default `127.0.0.1:7778`). The control client
uses:

- `GET /healthz`
- `GET /sessions/{id}`
- `GET /sessions/{id}/wait?timeout=...&quiet_for=...`
- fallback `POST /sessions/{id}/wait`
- `DELETE /sessions/{id}`
- `GET /sessions/{id}/otlp`

For non-loopback proxies, pass the bearer token as `token` or set
`LLMENTARY_CONTROL_TOKEN`.

### `Session`

The pytest `llmentary` fixture returns `llmentary.Session`.

Useful properties:

- `llmentary.session_id`
- `llmentary.base_url`: proxy URL
- `llmentary.openai_base_url`: proxy URL with `/v1`, suitable for `openai-python`
- `llmentary.headers`: `X-Test-Session` plus W3C `baggage` with `llmentary.session.id`
- `llmentary.baggage`: `llmentary.session.id=<session-id>`

Methods:

- `llmentary.wait(timeout=None, quiet_for=None)`: wait for quiescence and return a trace
- `llmentary.trace()`: fetch the current trace without waiting
- `llmentary.otlp()`: fetch OTLP bytes for the session
- `llmentary.cleanup()`: delete the session
- `llmentary.expect(soft=False)`: wait, then return an expectation object
- `llmentary.query(api_type=None, model=None)`: scope assertions by API type/model
- `llmentary.soft_expect()`: context manager that accumulates assertion failures

### `Query`

```python
llmentary.query(api_type="chat_completions", model="gpt-4.1-mini").expect().no_errors()
llmentary.query(api_type="responses").expect().tool_called("lookup")
```

Accepted API type spellings include `chat`, `chat_completions`, `chat.completions`,
`response`, `responses`, and the Go proxy's `api_family` values.

## Assertions

All assertion methods are chainable. Hard expectations raise `AssertionError` on
the first failure. Soft expectations collect failures and raise from
`assert_all()` or when leaving `llmentary.soft_expect()`.

### Tool calls

```python
expect.tool_called("get_weather")
expect.not_called("delete_user")
expect.called_times("search", 2)
expect.args_subset("get_weather", {"city": "Paris"})
expect.args_matcher("get_weather", lambda args: args["city"].startswith("Par"))
expect.order("search", "summarize")
expect.order("search", "summarize", strict=True)
expect.parallel_tool_calls()
expect.tool_result_fed_back("get_weather")
```

`args_subset` recursively checks only the expected keys. `args_matcher` accepts
either a mapping (same behavior as `args_subset`) or a callable returning bool.

### Request and response metadata

```python
expect.model("gpt-4.1-mini")
expect.request_param("temperature", 0)
expect.request_param("response_format.type", "json_schema")
expect.finish_reason("stop")
```

`request_param` supports dot notation for nested request JSON fields.

### Usage, latency, TTFT, retries

```python
expect.tokens(min_total=1, max_total=500)
expect.tokens(min_prompt=1, max_completion=100)
expect.latency(max_ms=2_000)
expect.ttft(max_ms=500)
expect.retries(max_count=1)
expect.no_retries()
```

The client accepts both Chat Completions token names
(`prompt_tokens`, `completion_tokens`, `total_tokens`) and Responses names
(`input_tokens`, `output_tokens`, `total_tokens`).

### Errors

```python
expect.no_errors()
expect.has_error("429")
expect.mid_stream_error("connection reset")
```

### Prompt and response content

```python
expect.prompt_contains("weather")
expect.prompt_regex(r"weather.+Paris")
expect.response_contains("sunny")
expect.response_regex(r"sunny|clear")
expect.response_json_valid()
expect.response_json_schema({"type": "object", "required": ["answer"]})
```

JSON assertions inspect model output text only (`output_text`, Chat
`choices[].message.content`, Responses `output[].content`), not the HTTP
response envelope.

## Trace payload handling

The client preserves the exact JSON returned by the Go proxy and normalizes the
common fields used for assertions:

- call list keys: `calls`, `requests`, `records`, `events`, `entries`
- API type keys and endpoint paths
- request/response `body` or `json`
- Chat Completions `choices[].message.tool_calls`
- Responses `output[]` function calls
- Chat `role=tool` results and Responses `function_call_output` results
- usage, latency, TTFT, retry, and error fields in the names used by the Go
  proxy and OpenAI APIs

Every assertion failure includes `SessionTrace.trace_summary()`, which lists each
captured call with API type, model, status, finish reason, timings, and tools.
