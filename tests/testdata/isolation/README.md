# Isolation vectors

Normative fixtures for `isolated_from` / `IsolatedFrom` / `isolatedFrom`.

Canonical copy: [`testdata/isolation`](https://github.com/barzoj-co/llmentary/tree/main/testdata/isolation)
on `barzoj-co/llmentary` (proxy isolation spec + vectors, including DOG-32 /
DOG-33 from https://github.com/barzoj-co/llmentary/pull/42). After a proxy
release includes this path, the same files on `v{PROXY_VERSION}` are the pin.

Each `*.json` file is two recorded sessions (`x`, `y`) plus `expect.isolated`
and `expect.message_contains`. Clients vendor this directory and fail CI if
the copy drifts from that path.

DOG-32: `empty-other.json` (fail), `empty-other-allow.json` (pass),
`missing-other-id.json` (fail). DOG-33: `later-call-two-new-user-messages.json`
and `responses-later-call-two-new-user-messages.json` (pass).
