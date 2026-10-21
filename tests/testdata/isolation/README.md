# Isolation vectors

Normative fixtures for `isolated_from` / `IsolatedFrom` / `isolatedFrom`.

Canonical copy: [`testdata/isolation`](https://github.com/barzoj-co/llmentary/tree/main/testdata/isolation)
on `barzoj-co/llmentary` (proxy DOG-29 PR https://github.com/barzoj-co/llmentary/pull/35,
merged to `main` as `bb39133`). After a proxy release includes this path, the
same files on `v{PROXY_VERSION}` are the pin.

Each `*.json` file is two recorded sessions (`x`, `y`) plus `expect.isolated`
and `expect.message_contains`. Clients vendor this directory and fail CI if
the copy drifts from that path.
