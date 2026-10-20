#!/usr/bin/env bash
# Print wheel filenames and tags the same way in release.yml and the dry-run job.
set -euo pipefail

shopt -s nullglob
wheels=(dist/*.whl)
if [[ ${#wheels[@]} -eq 0 ]]; then
  echo "no wheels in dist/" >&2
  exit 1
fi

for whl in "${wheels[@]}"; do
  echo "=== $(basename "${whl}") ==="
  python -m wheel info "${whl}"
  echo
done
