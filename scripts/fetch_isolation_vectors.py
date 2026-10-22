#!/usr/bin/env python3
"""Copy testdata/isolation from proxy main (then the pin tag)."""

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
PROXY_VECTOR_BRANCH = "main"


def _token() -> str:
    """Optional GITHUB_TOKEN for API rate limits. Never required."""

    value = os.environ.get("GITHUB_TOKEN") or ""
    return value.strip()


def _proxy_version(root: Path) -> str:
    path = root / "PROXY_VERSION"
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _get(url: str, token: str) -> tuple[object | None, int | None]:
    request = urllib.request.Request(url)
    request.add_header("Accept", "application/vnd.github+json")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8")), 200
    except urllib.error.HTTPError as exc:
        if exc.code in {404, 403}:
            return None, exc.code
        raise
    except (urllib.error.URLError, TimeoutError, OSError):
        return None, None


def _listing(token: str, ref: str) -> tuple[list[object] | None, int | None]:
    url = (
        f"https://api.github.com/repos/{PROXY_VECTOR_REPO}/contents/"
        f"{PROXY_VECTOR_PATH}?ref={ref}"
    )
    payload, status = _get(url, token)
    if isinstance(payload, list):
        return payload, status
    return None, status


def _download(token: str, ref: str, name: str) -> bytes | None:
    url = (
        f"https://api.github.com/repos/{PROXY_VECTOR_REPO}/contents/"
        f"{PROXY_VECTOR_PATH}/{name}?ref={ref}"
    )
    payload, _status = _get(url, token)
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
    refs = [PROXY_VECTOR_BRANCH]
    version = _proxy_version(Path(args.root))
    if version:
        refs.append(f"v{version}")
    listing = None
    ref = None
    last_status: int | None = None
    transport_error = False
    for candidate in refs:
        listing, last_status = _listing(token, candidate)
        if listing:
            ref = candidate
            break
        if last_status is None:
            transport_error = True
    if listing is None or ref is None:
        if last_status in {404, 403}:
            print(
                "fetch_isolation_vectors: failed: HTTP "
                f"{last_status} fetching {PROXY_VECTOR_REPO} {PROXY_VECTOR_PATH} "
                f"from {', '.join(refs)}"
            )
        elif transport_error:
            print(
                "fetch_isolation_vectors: failed: transport error fetching "
                f"{PROXY_VECTOR_REPO} {PROXY_VECTOR_PATH}"
            )
        else:
            print(
                "fetch_isolation_vectors: failed: "
                f"{PROXY_VECTOR_PATH} not available from {', '.join(refs)}"
            )
        return 1
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
    json_count = len(list(dest.glob("*.json")))
    if json_count == 0:
        print(
            "fetch_isolation_vectors: failed: "
            f"{PROXY_VECTOR_PATH}@{ref} listed no isolation vectors"
        )
        return 1
    (dest / ".ref").write_text(f"{ref}\n", encoding="utf-8")
    print(
        f"isolation vector drift check: fetched {json_count} vector(s) from {ref} "
        f"({written} file(s) -> {dest})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
