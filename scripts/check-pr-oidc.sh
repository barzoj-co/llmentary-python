#!/usr/bin/env bash
# Reject id-token: write on pull_request / pull_request_target workflows.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GOOD_DIR="${ROOT}/scripts/testdata/pr-oidc/good"
BAD_DIR="${ROOT}/scripts/testdata/pr-oidc/bad"
EXPECTED_REPO="barzoj-co/llm-test-lab-python"

scan_paths() {
  local path
  for path in "$@"; do
    if [[ -d "${path}" ]]; then
      find "${path}" \( -name '*.yml' -o -name '*.yaml' \) -print | sort
    elif [[ -f "${path}" ]]; then
      printf '%s\n' "${path}"
    fi
  done
}

check_files() {
  EXPECTED_REPO="${EXPECTED_REPO}" python3 - "$@" <<'PY'
from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("PyYAML is required (pip install pyyaml)", file=sys.stderr)
    sys.exit(2)

EXPECTED_IF = f"github.repository == '{os.environ['EXPECTED_REPO']}'"
PR_EVENTS = {"pull_request", "pull_request_target"}
SKIP_KEYS = {"__line__", "__key_lines__"}


class Node(dict):
    def __init__(self, line: int = 1) -> None:
        super().__init__()
        self.line = line
        self.key_lines: dict = {}


class Seq(list):
    def __init__(self, line: int = 1) -> None:
        super().__init__()
        self.line = line


class Loader(yaml.SafeLoader):
    pass


Loader.yaml_implicit_resolvers = {
    ch: [(tag, rx) for tag, rx in lst if not str(tag).endswith("bool")]
    for ch, lst in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


def construct_mapping(loader, node):
    mapping = Node(line=node.start_mark.line + 1)
    loader.flatten_mapping(node)
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        value = loader.construct_object(value_node, deep=True)
        mapping[key] = value
        mapping.key_lines[key] = key_node.start_mark.line + 1
    return mapping


def construct_sequence(loader, node):
    seq = Seq(line=node.start_mark.line + 1)
    for child in node.value:
        seq.append(loader.construct_object(child, deep=True))
    return seq


Loader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
)
Loader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_SEQUENCE_TAG, construct_sequence
)


def load(path: Path):
    with path.open(encoding="utf-8") as fh:
        doc = yaml.load(fh, Loader)
    if not isinstance(doc, dict):
        raise ValueError("workflow must be a mapping")
    return doc


def lineno(obj, default: int = 1) -> int:
    return getattr(obj, "line", default)


def key_line(mapping, key, default: int = 1) -> int:
    if isinstance(mapping, Node):
        return mapping.key_lines.get(key, mapping.line)
    return default


def trigger_names(on) -> set[str]:
    if on is None:
        return set()
    if isinstance(on, str):
        return {on}
    if isinstance(on, list):
        return {str(item) for item in on}
    if isinstance(on, dict):
        return {str(k) for k in on if k not in SKIP_KEYS}
    return {str(on)}


def grants_id_token_write(perms) -> tuple[bool, int]:
    if perms == "write-all":
        return True, lineno(perms)
    if isinstance(perms, dict):
        value = perms.get("id-token")
        if str(value) == "write":
            return True, key_line(perms, "id-token", lineno(perms))
        star = perms.get("*")
        if str(star) in {"write", "write-all"}:
            return True, lineno(perms)
    return False, 1


errors: list[str] = []


def err(path: Path, line: int, message: str) -> None:
    errors.append(f"{path}:{line}: {message}")


def check_workflow(path: Path) -> None:
    try:
        doc = load(path)
    except Exception as exc:  # noqa: BLE001
        err(path, 1, f"failed to parse: {exc}")
        return
    on = doc.get("on")
    names = trigger_names(on)
    pr_triggered = bool(names & PR_EVENTS)
    call_triggered = "workflow_call" in names
    top_perms = doc.get("permissions")
    granted, line = grants_id_token_write(top_perms)
    if pr_triggered and granted:
        err(
            path,
            key_line(doc, "permissions", line),
            "id-token: write is not allowed on pull_request/pull_request_target workflows",
        )
    jobs = doc.get("jobs") or {}
    if not isinstance(jobs, dict):
        return
    for job_id, job in jobs.items():
        if job_id in SKIP_KEYS or not isinstance(job, dict):
            continue
        granted, perm_line = grants_id_token_write(job.get("permissions"))
        if not granted:
            continue
        job_line = key_line(jobs, job_id, lineno(job, perm_line))
        if pr_triggered:
            err(
                path,
                perm_line or job_line,
                f"job {job_id}: id-token: write is not allowed on pull_request/pull_request_target workflows",
            )
            continue
        if call_triggered:
            raw_if = str(job.get("if") or "")
            normalized = raw_if.replace("${{", "").replace("}}", "").strip()
            if normalized != EXPECTED_IF:
                err(
                    path,
                    key_line(job, "permissions", job_line),
                    f"job {job_id}: id-token: write on workflow_call must set if: {EXPECTED_IF}",
                )


for raw in sys.argv[1:]:
    check_workflow(Path(raw))

if errors:
    for item in errors:
        print(item)
    sys.exit(1)
print(f"pr-oidc: {len(sys.argv) - 1} file(s) ok")
PY
}

self_test() {
  local good_files bad_file output failed=0 saw_bad=0
  mapfile -t good_files < <(scan_paths "${GOOD_DIR}")
  if [[ ${#good_files[@]} -eq 0 ]]; then
    echo "self-test: no good fixtures under ${GOOD_DIR}" >&2
    exit 1
  fi
  check_files "${good_files[@]}"

  while IFS= read -r bad_file; do
    saw_bad=1
    if output="$(check_files "${bad_file}" 2>&1)"; then
      echo "self-test: expected ${bad_file} to fail" >&2
      failed=1
    else
      if ! grep -E -q "^${bad_file}:[0-9]+:" <<<"${output}"; then
        echo "self-test: ${bad_file} was rejected without file:line" >&2
        echo "${output}" >&2
        failed=1
      elif ! grep -F -q "id-token: write" <<<"${output}"; then
        echo "self-test: ${bad_file} was rejected without 'id-token: write'" >&2
        echo "${output}" >&2
        failed=1
      else
        echo "self-test: rejected ${bad_file}"
        echo "${output}"
      fi
    fi
  done < <(scan_paths "${BAD_DIR}")
  if [[ "${saw_bad}" -eq 0 ]]; then
    echo "self-test: no bad fixtures under ${BAD_DIR}" >&2
    exit 1
  fi
  if [[ "${failed}" -ne 0 ]]; then
    exit 1
  fi
  echo "self-test: ok"
}

if [[ "${1:-}" == "--self-test" ]]; then
  self_test
  exit 0
fi

if [[ $# -gt 0 ]]; then
  check_files "$@"
  exit 0
fi

files=()
if [[ -d "${ROOT}/.github/workflows" ]]; then
  mapfile -t files < <(scan_paths "${ROOT}/.github/workflows")
fi
if [[ ${#files[@]} -eq 0 ]]; then
  echo "no workflow files found" >&2
  exit 1
fi
check_files "${files[@]}"
