# Isolation vectors

Normative fixtures for `isolated_from` / `IsolatedFrom` / `isolatedFrom`.

Canonical copy: [`testdata/isolation`](https://github.com/barzoj-co/llmentary/tree/cursor/dog-29-isolation-spec-f2b6/testdata/isolation)
on `barzoj-co/llmentary` (proxy DOG-29 PR https://github.com/barzoj-co/llmentary/pull/35,
branch `cursor/dog-29-isolation-spec-f2b6`). After that PR merges, the same
path on `v{PROXY_VERSION}` is the pin.

Each `*.json` file is two recorded sessions (`x`, `y`) plus `expect.isolated`
and `expect.message_contains`. Clients vendor this directory and fail CI if
the copy drifts from that path.
