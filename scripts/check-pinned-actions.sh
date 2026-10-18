#!/usr/bin/env bash
# Verify GitHub Actions pins: owner/repo[/path]@<40 hex> # v...
# Local ./... and docker://...@sha256:<64 hex> are also allowed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GOOD_DIR="${ROOT}/scripts/testdata/pinned-actions/good"
BAD_DIR="${ROOT}/scripts/testdata/pinned-actions/bad"

scan_paths() {
  local path
  for path in "$@"; do
    if [[ -d "$path" ]]; then
      find "$path" \( -name '*.yml' -o -name '*.yaml' \) -print
    elif [[ -f "$path" ]]; then
      printf '%s\n' "$path"
    fi
  done
}

check_files() {
  python3 - "$@" <<'PY'
import re
import sys
from pathlib import Path

USES = re.compile(r"^(?:-\s+)?uses:\s*(?:['\"]([^'\"]+)['\"]|(\S+))(.*)$")
GITHUB = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_./-]+)?@[0-9a-fA-F]{40}$")
DOCKER = re.compile(r"^docker://.+@sha256:[0-9a-fA-F]{64}$")
LOCAL = re.compile(r"^\./")
VERSION_COMMENT = re.compile(r"#\s*v\d")

offenders = []
for path in sys.argv[1:]:
    text = Path(path).read_text(encoding="utf-8")
    for lineno, raw in enumerate(text.splitlines(), 1):
        stripped = raw.lstrip()
        if stripped.startswith("#") or "uses:" not in stripped:
            continue
        # Mapping key `uses:` or list item `- uses:`.
        if not re.match(r"(?:-\s+)?uses:\s*", stripped):
            continue
        match = USES.match(stripped)
        if not match:
            offenders.append((path, lineno, raw.rstrip()))
            continue
        value = match.group(1) or match.group(2)
        rest = match.group(3) or ""
        if LOCAL.match(value):
            continue
        if DOCKER.match(value):
            continue
        if GITHUB.match(value) and VERSION_COMMENT.search(rest):
            continue
        offenders.append((path, lineno, raw.rstrip()))

if offenders:
    for path, lineno, line in offenders:
        print(f"{path}:{lineno}: unpinned or uncommented action: {line}")
    sys.exit(1)
print(f"pinned-actions: {len(sys.argv) - 1} file(s) ok")
PY
}

self_test() {
  local good_files bad_file
  mapfile -t good_files < <(scan_paths "$GOOD_DIR")
  if [[ ${#good_files[@]} -eq 0 ]]; then
    echo "self-test: no good fixtures under $GOOD_DIR" >&2
    exit 1
  fi
  check_files "${good_files[@]}"

  local failed=0
  local saw_bad=0
  while IFS= read -r bad_file; do
    saw_bad=1
    if output="$(check_files "$bad_file" 2>&1)"; then
      echo "self-test: expected $bad_file to fail" >&2
      failed=1
    else
      if ! grep -E -q "^${bad_file}:[0-9]+:" <<<"$output"; then
        echo "self-test: $bad_file was rejected without file:line" >&2
        echo "$output" >&2
        failed=1
      else
        echo "self-test: rejected $bad_file"
        echo "$output"
      fi
    fi
  done < <(scan_paths "$BAD_DIR")
  if [[ "$saw_bad" -eq 0 ]]; then
    echo "self-test: no bad fixtures under $BAD_DIR" >&2
    exit 1
  fi
  if [[ "$failed" -ne 0 ]]; then
    exit 1
  fi
  echo "self-test: ok"
}

if [[ "${1:-}" == "--self-test" ]]; then
  self_test
  exit 0
fi

files=()
if [[ -d "${ROOT}/.github/workflows" ]]; then
  mapfile -t files < <(scan_paths "${ROOT}/.github/workflows")
fi
if [[ -d "${ROOT}/.github/actions" ]]; then
  mapfile -t extra < <(find "${ROOT}/.github/actions" \( -name 'action.yml' -o -name 'action.yaml' \) -print)
  files+=("${extra[@]}")
fi
if [[ ${#files[@]} -eq 0 ]]; then
  echo "no workflow or action files found" >&2
  exit 1
fi
check_files "${files[@]}"
