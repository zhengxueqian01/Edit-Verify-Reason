#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../common/run_task_eval_common.sh"

run_task_eval \
  "task2" \
  "dataset/task2-line/del" \
  "dataset/task2-line/del-add" \
  "dataset/task2-line/del-change" \
  "$@"
