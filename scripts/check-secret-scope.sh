#!/usr/bin/env bash
# Enforce step-scoped secrets and a real e2e-gate job across workflow files.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GOOD_DIR="${ROOT}/scripts/testdata/secret-scope/good"
BAD_DIR="${ROOT}/scripts/testdata/secret-scope/bad"
E2E_JOB="e2e"

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
  local e2e_job="$1"
  shift
  E2E_JOB="${e2e_job}" python3 - "$@" <<'PY'
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("PyYAML is required (pip install pyyaml)", file=sys.stderr)
    sys.exit(2)

E2E_JOB = os.environ["E2E_JOB"]
E2E_ENVIRONMENT = "openai-e2e"
OPENAI_SECRET = "OPENAI_API_KEY"
SKIP_KEYS = {"__line__", "__key_lines__"}
RAW_SECRET_EXPR = re.compile(r"\$\{\{\s*secrets\.([A-Za-z0-9_]+)\s*\}\}")
GITHUB_TOKEN_EXPR = re.compile(r"\$\{\{\s*(?:github\.token|secrets\.GITHUB_TOKEN)\s*\}\}")
ANY_SECRET_EXPR = re.compile(r"\$\{\{\s*secrets\.")
INSTALL_RE = re.compile(
    r"(?:^|[;\n&|])\s*(?:npm ci\b|npm install\b|npm i |pnpm install\b|"
    r"yarn install\b|pip install\b|python -m pip install\b|uv sync\b|"
    r"uv pip install\b|poetry install\b|go mod download\b|go get\b)"
)


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


def as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return ""


def walk_strings(value):
    if isinstance(value, str):
        yield value, 1
        return
    if isinstance(value, dict):
        for key, inner in value.items():
            if key in SKIP_KEYS:
                continue
            line = key_line(value, key, lineno(value))
            if isinstance(inner, str):
                yield inner, line
            else:
                for text, inner_line in walk_strings(inner):
                    yield text, inner_line
        return
    if isinstance(value, list):
        for inner in value:
            yield from walk_strings(inner)


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


def trigger_line(on, name: str, fallback: int) -> int:
    if isinstance(on, Node):
        return on.key_lines.get(name, on.line)
    return lineno(on, fallback)


errors: list[str] = []


def err(path: Path, line: int, message: str) -> None:
    errors.append(f"{path}:{line}: {message}")


def raw_secrets(text: str) -> list[str]:
    if not text:
        return []
    names: list[str] = []
    for match in RAW_SECRET_EXPR.finditer(text):
        names.append(match.group(1))
    if GITHUB_TOKEN_EXPR.search(text) and "GITHUB_TOKEN" not in names:
        names.append("GITHUB_TOKEN")
    return names


def interpolates_secret(text: str) -> bool:
    if not text:
        return False
    if RAW_SECRET_EXPR.search(text) or GITHUB_TOKEN_EXPR.search(text):
        return True
    # ${{ secrets.FOO != '' }} is a boolean presence check, not interpolation.
    for match in ANY_SECRET_EXPR.finditer(text):
        rest = text[match.end() :]
        if not re.match(r"[A-Za-z0-9_]+\s*(?:!=|==)\s*''", rest):
            return True
    return False


def is_checkout_uses(uses: str) -> bool:
    return uses.startswith("actions/checkout@")


def needs_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def check_env_block(path: Path, env, reason: str) -> None:
    if not isinstance(env, dict):
        return
    for key, value in env.items():
        if key in SKIP_KEYS:
            continue
        text = as_text(value)
        if raw_secrets(text) or interpolates_secret(text):
            err(path, key_line(env, key, lineno(env)), reason)


def check_job(path: Path, job_id: str, job: dict) -> None:
    secrets = job.get("secrets")
    if secrets == "inherit":
        err(
            path,
            key_line(job, "secrets", lineno(job)),
            "secrets: inherit is forbidden; pass named secrets",
        )

    check_env_block(path, job.get("env"), "secret in job-level env")

    steps = job.get("steps") or []
    raw_steps: dict[str, set[int]] = {}
    if not isinstance(steps, list):
        return
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        step_line = lineno(step)
        run_text = as_text(step.get("run"))
        uses = as_text(step.get("uses"))
        env = step.get("env")
        with_block = step.get("with")

        for text, line in walk_strings(run_text):
            if interpolates_secret(text):
                err(
                    path,
                    key_line(step, "run", line or step_line),
                    "secret interpolated into run:; pass it via step env",
                )
                break

        if isinstance(with_block, dict):
            for key, value in with_block.items():
                if key in SKIP_KEYS:
                    continue
                text = as_text(value)
                if not raw_secrets(text):
                    continue
                if key == "token" and is_checkout_uses(uses):
                    continue
                err(
                    path,
                    key_line(with_block, key, step_line),
                    "secret in with:",
                )

        env_has_raw = False
        if isinstance(env, dict):
            for key, value in env.items():
                if key in SKIP_KEYS:
                    continue
                text = as_text(value)
                names = raw_secrets(text)
                if not names:
                    continue
                env_has_raw = True
                for name in names:
                    raw_steps.setdefault(name, set()).add(index)
        if isinstance(with_block, dict):
            for key, value in with_block.items():
                if key in SKIP_KEYS:
                    continue
                text = as_text(value)
                for name in raw_secrets(text):
                    raw_steps.setdefault(name, set()).add(index)
        for name in raw_secrets(run_text):
            raw_steps.setdefault(name, set()).add(index)

        if env_has_raw and INSTALL_RE.search(run_text):
            err(
                path,
                key_line(step, "env", step_line),
                "secret on an install step",
            )

    for name, step_ids in raw_steps.items():
        if name in {"GITHUB_TOKEN", "github.token"}:
            continue
        if len(step_ids) > 1:
            err(
                path,
                lineno(job),
                f"secret {name} used raw in more than one step of job {job_id}; "
                "presence checks must use the boolean form",
            )


def job_environment(job: dict) -> str:
    value = job.get("environment")
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return as_text(value.get("name"))
    return ""


def job_receives_openai(job: dict) -> bool:
    for text, _line in walk_strings(job):
        if OPENAI_SECRET in raw_secrets(text):
            return True
    return False


def check_e2e_job(path: Path, doc: dict) -> None:
    jobs = doc.get("jobs") or {}
    if not isinstance(jobs, dict):
        jobs = {}
    if path.name == "ci.yml":
        e2e = jobs.get(E2E_JOB)
        if not isinstance(e2e, dict) or job_environment(e2e) != E2E_ENVIRONMENT:
            line = (
                key_line(jobs, E2E_JOB, lineno(e2e) if isinstance(e2e, dict) else 1)
                if E2E_JOB in jobs
                else key_line(doc, "jobs", 1)
            )
            err(path, line, "live e2e job must set environment: openai-e2e")
    for job_id, job in jobs.items():
        if job_id in SKIP_KEYS or not isinstance(job, dict):
            continue
        env_name = job_environment(job)
        if job_id != E2E_JOB and env_name == E2E_ENVIRONMENT:
            err(
                path,
                key_line(job, "environment", lineno(job)),
                "environment: openai-e2e is only allowed on the live e2e job",
            )
        if job_id != E2E_JOB and job_receives_openai(job):
            err(
                path,
                lineno(job),
                "OPENAI_API_KEY is only allowed in the live e2e job",
            )


def check_e2e_gate(path: Path, doc: dict) -> None:
    if path.name != "ci.yml":
        return
    jobs = doc.get("jobs") or {}
    if not isinstance(jobs, dict) or "e2e-gate" not in jobs:
        err(path, key_line(doc, "jobs", 1), "missing or malformed e2e-gate job")
        return
    gate = jobs["e2e-gate"]
    if not isinstance(gate, dict):
        err(path, key_line(jobs, "e2e-gate", 1), "missing or malformed e2e-gate job")
        return
    if_expr = as_text(gate.get("if"))
    needs = needs_list(gate.get("needs"))
    if "always()" not in if_expr or E2E_JOB not in needs:
        err(
            path,
            key_line(jobs, "e2e-gate", lineno(gate)),
            "missing or malformed e2e-gate job",
        )


def check_workflow(path: Path) -> None:
    try:
        doc = load(path)
    except Exception as exc:  # noqa: BLE001
        err(path, 1, f"failed to parse: {exc}")
        return
    on = doc.get("on")
    names = trigger_names(on)
    if "pull_request_target" in names:
        err(
            path,
            trigger_line(on, "pull_request_target", key_line(doc, "on", 1)),
            "pull_request_target is forbidden",
        )
    check_env_block(path, doc.get("env"), "secret in workflow-level env")
    jobs = doc.get("jobs") or {}
    if isinstance(jobs, dict):
        for job_id, job in jobs.items():
            if job_id in SKIP_KEYS or not isinstance(job, dict):
                continue
            check_job(path, job_id, job)
    check_e2e_gate(path, doc)
    check_e2e_job(path, doc)


for raw in sys.argv[1:]:
    check_workflow(Path(raw))

if errors:
    for item in errors:
        print(item)
    sys.exit(1)
print(f"secret-scope: {len(sys.argv) - 1} file(s) ok")
PY
}

self_test() {
  local good_files bad_file output expect failed=0 saw_bad=0 first
  mapfile -t good_files < <(scan_paths "${GOOD_DIR}")
  if [[ ${#good_files[@]} -eq 0 ]]; then
    echo "self-test: no good fixtures under ${GOOD_DIR}" >&2
    exit 1
  fi
  check_files "${E2E_JOB}" "${good_files[@]}"

  while IFS= read -r bad_file; do
    saw_bad=1
    first="$(head -n 1 "${bad_file}")"
    expect="${first#\# expect: }"
    if [[ "${first}" != "# expect: "* || -z "${expect}" ]]; then
      echo "self-test: ${bad_file} is missing '# expect: <substring>'" >&2
      failed=1
      continue
    fi
    if output="$(check_files "${E2E_JOB}" "${bad_file}" 2>&1)"; then
      echo "self-test: expected ${bad_file} to fail" >&2
      failed=1
    else
      if ! grep -E -q "^${bad_file}:[0-9]+:" <<<"${output}"; then
        echo "self-test: ${bad_file} was rejected without file:line" >&2
        echo "${output}" >&2
        failed=1
      elif ! grep -F -q "${expect}" <<<"${output}"; then
        echo "self-test: ${bad_file} was rejected without expected reason: ${expect}" >&2
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

usage() {
  echo "usage: $0 [--e2e-job <id>] [--self-test] [files...]" >&2
}

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --e2e-job)
      if [[ $# -lt 2 ]]; then
        usage
        exit 2
      fi
      E2E_JOB="$2"
      shift 2
      ;;
    --self-test)
      self_test
      exit 0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

if [[ ${#POSITIONAL[@]} -gt 0 ]]; then
  check_files "${E2E_JOB}" "${POSITIONAL[@]}"
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
check_files "${E2E_JOB}" "${files[@]}"
