#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../common/run_task_eval_common.sh"

run_task_eval \
  "task1" \
  "dataset/task1/add" \
  "dataset/task1/del" \
  "dataset/task1/add-del" \
  "dataset/task1/change" \
  "dataset/task1/add-change" \
  "dataset/task1/del-change" \
  "$@"
