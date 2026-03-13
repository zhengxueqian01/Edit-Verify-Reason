#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../common/run_task_eval_common.sh"

run_task_eval \
  "task1" \
  "dataset/task1-mix-area/add" \
  "dataset/task1-mix-area/del" \
  "dataset/task1-mix-area/add-del" \
  "dataset/task1-mix-area/change" \
  "dataset/task1-mix-area/add-change" \
  "dataset/task1-mix-area/del-change" \
  "$@"
