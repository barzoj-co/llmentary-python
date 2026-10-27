#!/usr/bin/env bash
# Enforce least-privilege permissions, timeouts, concurrency, and the release
# publish path described in docs/releasing.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKFLOWS="${ROOT}/.github/workflows"
BAD_DIR="${ROOT}/scripts/testdata/workflow-policy/bad"

check_workflows() {
  python3 - "$1" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path

WORKFLOWS = Path(sys.argv[1])

PR_GROUP = "${{ github.workflow }}-${{ github.event.pull_request.number || github.ref }}"
PR_CANCEL = "${{ github.event_name == 'pull_request' }}"

ALLOWED_PERM_KEYS = {
    "contents",
    "actions",
    "security-events",
    "id-token",
    "pull-requests",
    "issues",
    "attestations",
}

errors: list[str] = []


def err(path: Path, message: str) -> None:
    errors.append(f"{path}: {message}")


def parse_yaml(text: str) -> dict:
    """Indentation-based subset parser for GitHub Actions YAML."""

    # Strip full-line comments but keep values that contain #.
    raw_lines = text.splitlines()
    tokens: list[tuple[int, str]] = []
    for raw in raw_lines:
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        tokens.append((indent, raw.strip()))

    def strip_inline_comment(raw: str) -> str:
        in_single = False
        in_double = False
        brace = 0
        for i, ch in enumerate(raw):
            if ch == "'" and not in_double:
                in_single = not in_single
            elif ch == '"' and not in_single:
                in_double = not in_double
            elif raw.startswith("${{", i) and not in_single and not in_double:
                brace += 1
            elif raw.startswith("}}", i) and brace:
                brace -= 1
            elif ch == "#" and not in_single and not in_double and brace == 0:
                if i == 0 or raw[i - 1].isspace():
                    return raw[:i].rstrip()
        return raw

    def parse_value(raw: str):
        raw = strip_inline_comment(raw.strip())
        if raw == "" or raw in {"null", "~"}:
            return None
        if raw in {"true", "false"}:
            return raw == "true"
        if re.fullmatch(r"-?\d+", raw):
            return int(raw)
        if raw.startswith("{") and raw.endswith("}") and not raw[1:-1].strip():
            return {}
        if raw.startswith("[") and raw.endswith("]"):
            inner = raw[1:-1].strip()
            if not inner:
                return []
            return [parse_value(part) for part in _split_flow(inner)]
        if (raw.startswith('"') and raw.endswith('"')) or (
            raw.startswith("'") and raw.endswith("'")
        ):
            return raw[1:-1]
        return raw

    def _split_flow(inner: str) -> list[str]:
        parts: list[str] = []
        buf: list[str] = []
        quote = None
        for ch in inner:
            if quote:
                buf.append(ch)
                if ch == quote:
                    quote = None
                continue
            if ch in {"'", '"'}:
                quote = ch
                buf.append(ch)
                continue
            if ch == ",":
                parts.append("".join(buf).strip())
                buf = []
                continue
            buf.append(ch)
        if buf:
            parts.append("".join(buf).strip())
        return parts

    def parse_block(index: int, indent: int):
        mapping: dict = {}
        sequence: list = []
        is_list = False
        while index < len(tokens):
            cur_indent, content = tokens[index]
            if cur_indent < indent:
                break
            if cur_indent > indent:
                raise ValueError(f"unexpected indent at {content}")
            if content.startswith("- "):
                is_list = True
                item_raw = content[2:]
                if ":" in item_raw and not item_raw.startswith("${{"):
                    key, _, rest = item_raw.partition(":")
                    item = {key.strip(): parse_value(rest)}
                    nxt = index + 1
                    if nxt < len(tokens) and tokens[nxt][0] > cur_indent:
                        nested, nxt = parse_block(nxt, tokens[nxt][0])
                        if isinstance(nested, dict):
                            item.update(nested)
                    sequence.append(item)
                    index = nxt
                else:
                    sequence.append(parse_value(item_raw))
                    index += 1
                continue
            key, sep, rest = content.partition(":")
            if not sep:
                raise ValueError(f"cannot parse: {content}")
            key = key.strip()
            rest = rest.strip()
            index += 1
            if rest in {"|", ">", "|-", ">-", "|+", ">+"}:
                # Swallow the literal/folded block; we only need structure.
                block_lines: list[str] = []
                while index < len(tokens) and tokens[index][0] > cur_indent:
                    block_lines.append(tokens[index][1])
                    index += 1
                mapping[key] = "\n".join(block_lines)
                continue
            if rest:
                mapping[key] = parse_value(rest)
                continue
            if index < len(tokens) and tokens[index][0] > cur_indent:
                nested, index = parse_block(index, tokens[index][0])
                mapping[key] = nested
            else:
                mapping[key] = None
        return (sequence if is_list else mapping), index

    doc, _ = parse_block(0, tokens[0][0] if tokens else 0)
    if not isinstance(doc, dict):
        raise ValueError("workflow document must be a mapping")
    return doc


def as_on(value) -> dict:
    if value is None:
        return {}
    if isinstance(value, str):
        return {value: None}
    if isinstance(value, list):
        return {item: None for item in value}
    if isinstance(value, dict):
        return value
    return {}


def has_pull_request(on: dict) -> bool:
    return "pull_request" in on


def permission_map(value) -> dict[str, str]:
    if value is None:
        return {}
    if value == {}:
        return {}
    if isinstance(value, str):
        return {"*": value}
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    raise ValueError(f"unsupported permissions value: {value!r}")


def job_uses_codecov(job: dict) -> bool:
    for step in job.get("steps") or []:
        if not isinstance(step, dict):
            continue
        uses = str(step.get("uses") or "")
        if uses.startswith("codecov/codecov-action@"):
            return True
    return False


def job_is_publish(job_id: str, job: dict) -> bool:
    return job_id == "publish" or str(job.get("name") or "") == "publish"


def check_top_level_permissions(path: Path, doc: dict) -> None:
    if "permissions" not in doc:
        err(path, "missing top-level permissions")
        return
    perms = permission_map(doc["permissions"])
    if perms == {}:
        return
    allowed = {"contents": "read"}
    if perms != allowed:
        err(
            path,
            f"over-broad or unexpected top-level permissions: {perms} "
            "(want contents: read or {})",
        )


def check_job_permissions(
    path: Path, job_id: str, job: dict, workflow_name: str, on: dict
) -> None:
    perms = permission_map(job.get("permissions"))
    for key, value in perms.items():
        if key not in ALLOWED_PERM_KEYS:
            err(path, f"job {job_id}: permission {key} is not listed in the policy")
        if value not in {"read", "write"}:
            err(path, f"job {job_id}: permission {key} has unexpected value {value}")

    name = str(job.get("name") or "")
    is_scorecard = workflow_name == "Scorecard" or name == "Scorecard" or job_id == "scorecard"
    is_codeql = name.startswith("CodeQL") or job_id == "codeql"
    is_publish = job_is_publish(job_id, job)
    is_release_please = path.name == "release-please.yml"
    is_label_dry = name == "labels (dry run)"
    is_label_sync = path.name == "labels.yml"
    pr_triggered = has_pull_request(on) or "pull_request_target" in on

    if perms.get("id-token") == "write":
        if pr_triggered:
            err(
                path,
                f"job {job_id}: id-token: write is not allowed on PR-triggered workflows",
            )
        elif not (is_publish or is_scorecard):
            err(path, f"job {job_id}: id-token: write is only allowed on publish and Scorecard")
    if perms.get("attestations") == "write" and not is_publish:
        err(path, f"job {job_id}: attestations: write is only allowed on publish")
    if is_publish and perms.get("id-token") != "write":
        err(path, "publish job must have id-token: write")
    if is_scorecard:
        expected = {
            "contents": "read",
            "actions": "read",
            "security-events": "write",
            "id-token": "write",
        }
        if not expected.items() <= perms.items():
            err(path, f"Scorecard job permissions must include {expected}, got {perms}")
    if is_codeql:
        expected = {
            "contents": "read",
            "actions": "read",
            "security-events": "write",
        }
        if not expected.items() <= perms.items():
            err(path, f"CodeQL job permissions must include {expected}, got {perms}")
    if is_release_please and job_id == "release-please":
        expected = {
            "contents": "write",
            "pull-requests": "write",
            "issues": "write",
            "actions": "write",
        }
        if not expected.items() <= perms.items():
            err(path, f"release-please job permissions must include {expected}, got {perms}")
    if is_label_sync and not is_label_dry:
        if perms.get("issues") != "write":
            err(path, f"label sync job {job_id} must have issues: write")
    if is_label_dry:
        if perms.get("issues") != "read" or perms.get("contents") not in {None, "read"}:
            err(path, f"labels (dry run) must have contents: read and issues: read, got {perms}")


def check_concurrency(path: Path, doc: dict, on: dict) -> None:
    conc = doc.get("concurrency")
    release_workflow = path.name in {"release.yml", "release-please.yml"}
    if has_pull_request(on):
        if not isinstance(conc, dict):
            err(path, "PR-triggered workflow is missing the concurrency block")
            return
        if conc.get("group") != PR_GROUP:
            err(path, f"concurrency.group must be exactly {PR_GROUP!r}, got {conc.get('group')!r}")
        cancel = conc.get("cancel-in-progress")
        if cancel != PR_CANCEL and cancel is not True and cancel != "true":
            # Accept the expression form required by the ticket.
            if str(cancel) != PR_CANCEL:
                err(
                    path,
                    f"concurrency.cancel-in-progress must be {PR_CANCEL!r}, got {cancel!r}",
                )
    if release_workflow:
        if not isinstance(conc, dict):
            err(path, "release workflow is missing a fixed concurrency group")
            return
        if conc.get("cancel-in-progress") not in {False, "false", False}:
            if conc.get("cancel-in-progress") is not False:
                err(path, "release workflow must set cancel-in-progress: false")
        group = str(conc.get("group") or "")
        if PR_GROUP in group or "pull_request" in group:
            err(path, "release workflow concurrency group must be fixed, not PR-scoped")


def dump_contains(path: Path, needle: str) -> bool:
    return needle in path.read_text(encoding="utf-8")


def check_release_yml(path: Path, doc: dict) -> None:
    on = as_on(doc.get("on"))
    allowed_on = set(on)
    if allowed_on - {"push", "workflow_dispatch"}:
        err(path, f"release.yml triggers must only be v* tags and workflow_dispatch, got {sorted(on)}")
    push = on.get("push")
    tags = None
    if isinstance(push, dict):
        tags = push.get("tags")
    if tags != ["v*"] and tags != "v*":
        err(path, f"release.yml push.tags must be [v*], got {tags!r}")
    if "workflow_dispatch" not in on:
        err(path, "release.yml must allow workflow_dispatch")

    jobs = doc.get("jobs") or {}
    if "publish" not in jobs:
        err(path, "release.yml is missing job publish")
        return
    if "build" not in jobs:
        err(path, "release.yml is missing job build")

    text = path.read_text(encoding="utf-8")
    if 'version = "0.1.0"' in text and "replace" in text:
        err(path, "release.yml still rewrites version = \"0.1.0\"")
    if "pypa/gh-action-pypi-publish@release/v1" in text:
        err(path, "release.yml still uses pypa/gh-action-pypi-publish@release/v1")

    for job_id, job in jobs.items():
        if not isinstance(job, dict):
            continue
        if_expr = str(job.get("if") or "")
        if "refs/tags/v" not in if_expr:
            err(path, f"job {job_id} must be guarded by a refs/tags/v check")

    publish = jobs["publish"]
    if str(publish.get("environment") or "") != "pypi":
        err(path, "publish job must set environment: pypi")
    perms = permission_map(publish.get("permissions"))
    if perms.get("id-token") != "write":
        err(path, "publish job must grant id-token: write")
    if "PYPI_PUBLISH_ENABLED" not in str(publish.get("if") or ""):
        err(path, "publish job if: must require vars.PYPI_PUBLISH_ENABLED == 'true'")
    uses = []
    for step in publish.get("steps") or []:
        if isinstance(step, dict) and step.get("uses"):
            uses.append(str(step["uses"]))
    if not any(u.startswith("pypa/gh-action-pypi-publish@") and re.search(r"@[0-9a-fA-F]{40}$", u) for u in uses):
        err(path, "publish job must use SHA-pinned pypa/gh-action-pypi-publish")


def step_dispatches_release(step: dict) -> bool:
    run = str(step.get("run") or "")
    return "workflow run release.yml" in run


def nonempty_gh_repo(env) -> bool:
    if not isinstance(env, dict) or "GH_REPO" not in env:
        return False
    value = env.get("GH_REPO")
    if value is None:
        return False
    return bool(str(value).strip().strip("'\""))


def text_has_nonempty_gh_repo(text: str) -> bool:
    for match in re.finditer(r"GH_REPO:\s*(.*)$", text, re.M):
        raw = match.group(1).strip().strip("'\"")
        if raw:
            return True
    return False


def dispatch_targets_repo(step: dict, job: dict) -> bool:
    run = str(step.get("run") or "")
    if re.search(r"(^|\s)--repo(\s|=)", run):
        return True
    return nonempty_gh_repo(step.get("env")) or nonempty_gh_repo(job.get("env"))


def check_release_please_yml(path: Path, doc: dict) -> None:
    text = path.read_text(encoding="utf-8")
    if "gh workflow run release.yml" not in text:
        err(path, "release-please.yml must dispatch release.yml")
    if "release_created" not in text:
        err(path, "release-please.yml must dispatch only when release_created is true")

    found_dispatch = False
    jobs = doc.get("jobs") or {}
    if isinstance(jobs, dict):
        for job in jobs.values():
            if not isinstance(job, dict):
                continue
            for step in job.get("steps") or []:
                if not isinstance(step, dict) or not step_dispatches_release(step):
                    continue
                found_dispatch = True
                if not dispatch_targets_repo(step, job):
                    err(
                        path,
                        "release-please dispatch step must set GH_REPO or pass --repo",
                    )
    if "gh workflow run release.yml" in text and not found_dispatch:
        if not text_has_nonempty_gh_repo(text) and "--repo" not in text:
            err(path, "release-please dispatch step must set GH_REPO or pass --repo")


def check_workflow(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    try:
        doc = parse_yaml(text)
    except Exception as exc:  # noqa: BLE001
        err(path, f"failed to parse: {exc}")
        return
    on = as_on(doc.get("on"))
    check_top_level_permissions(path, doc)
    check_concurrency(path, doc, on)
    jobs = doc.get("jobs") or {}
    if not isinstance(jobs, dict) or not jobs:
        err(path, "no jobs")
        return
    workflow_name = str(doc.get("name") or "")
    for job_id, job in jobs.items():
        if not isinstance(job, dict):
            err(path, f"job {job_id} is not a mapping")
            continue
        if "timeout-minutes" not in job:
            err(path, f"job {job_id} is missing timeout-minutes")
        check_job_permissions(path, job_id, job, workflow_name, on)
    if path.name == "release.yml":
        check_release_yml(path, doc)
    if path.name == "release-please.yml":
        check_release_please_yml(path, doc)


paths = sorted(WORKFLOWS.rglob("*.yml")) + sorted(WORKFLOWS.rglob("*.yaml"))
if not paths:
    print(f"no workflows in {WORKFLOWS}", file=sys.stderr)
    sys.exit(1)
for path in paths:
    check_workflow(path)

if errors:
    print("workflow policy failures:", file=sys.stderr)
    for item in errors:
        print(item, file=sys.stderr)
    sys.exit(1)
print(f"workflow policy: {len(paths)} workflow(s) ok")
PY
}

self_test() {
  local output=""
  local rc=0
  if [[ ! -d "$BAD_DIR" ]]; then
    echo "self-test: missing $BAD_DIR" >&2
    exit 1
  fi
  set +e
  output="$(check_workflows "$BAD_DIR" 2>&1)"
  rc=$?
  set -e
  if [[ "$rc" -eq 0 ]]; then
    echo "self-test: expected $BAD_DIR to fail" >&2
    echo "$output" >&2
    exit 1
  fi
  if ! grep -Eq 'GH_REPO|--repo' <<<"$output"; then
    echo "self-test: expected a GH_REPO/--repo failure, got:" >&2
    echo "$output" >&2
    exit 1
  fi
  echo "self-test: rejected missing GH_REPO/--repo"
  echo "$output"
  echo "self-test: ok"
}

if [[ "${1:-}" == "--self-test" ]]; then
  self_test
  exit 0
fi

check_workflows "$WORKFLOWS"
