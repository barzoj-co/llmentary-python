#!/usr/bin/env python3
"""Copy testdata/isolation from the proxy DOG-29 branch (or the pin tag)."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROXY_VECTOR_REPO = "barzoj-co/llmentary"
PROXY_VECTOR_PATH = "testdata/isolation"
PROXY_VECTOR_BRANCH = "cursor/dog-29-isolation-spec-f2b6"


def _token() -> str:
    return (
        os.environ.get("LLMENTARY_REPO_TOKEN")
        or os.environ.get("LLMTESTLAB_REPO_TOKEN")
        or os.environ.get("GITHUB_TOKEN")
        or ""
    )


def _proxy_version(root: Path) -> str:
    path = root / "PROXY_VERSION"
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _get(url: str, token: str) -> object | None:
    request = urllib.request.Request(url)
    request.add_header("Authorization", f"Bearer {token}")
    request.add_header("Accept", "application/vnd.github+json")
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in {404, 403}:
            return None
        raise
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def _listing(token: str, ref: str) -> list[object] | None:
    url = (
        f"https://api.github.com/repos/{PROXY_VECTOR_REPO}/contents/"
        f"{PROXY_VECTOR_PATH}?ref={ref}"
    )
    payload = _get(url, token)
    return payload if isinstance(payload, list) else None


def _download(token: str, ref: str, name: str) -> bytes | None:
    url = (
        f"https://api.github.com/repos/{PROXY_VECTOR_REPO}/contents/"
        f"{PROXY_VECTOR_PATH}/{name}?ref={ref}"
    )
    payload = _get(url, token)
    if not isinstance(payload, dict):
        return None
    encoded = payload.get("content")
    if not isinstance(encoded, str):
        return None
    return base64.b64decode(encoded)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Directory to write the vectors")
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Client repo root (for PROXY_VERSION)",
    )
    args = parser.parse_args()
    token = _token()
    if not token:
        print("fetch_isolation_vectors: no repo token; skip")
        return 0
    refs = [PROXY_VECTOR_BRANCH]
    version = _proxy_version(Path(args.root))
    if version:
        refs.append(f"v{version}")
    listing = None
    ref = None
    for candidate in refs:
        listing = _listing(token, candidate)
        if listing:
            ref = candidate
            break
    if listing is None or ref is None:
        print("fetch_isolation_vectors: testdata/isolation not available; skip")
        return 0
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=True)
    written = 0
    for item in listing:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name:
            continue
        if not (name.endswith(".json") or name == "README.md"):
            continue
        body = _download(token, ref, name)
        if body is None:
            continue
        (dest / name).write_bytes(body)
        written += 1
    print(f"fetch_isolation_vectors: wrote {written} file(s) from {ref} -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
