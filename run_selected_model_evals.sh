#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "${ROOT_DIR}/evaluation/common/run_task_eval_common.sh"

RUN_PREFIX="${RUN_PREFIX:-selected_eval_$(date +%Y%m%d_%H%M%S)}"
SESSION_ROOT="${SESSION_ROOT:-${ROOT_DIR}/output/dataset_records/${RUN_PREFIX}}"

echo "session_root=${SESSION_ROOT}"

run_task_eval \
  "task1" \
  "dataset/task1/add" \
  "dataset/task1/add-change" \
  "dataset/task1/add-del" \
  "dataset/task1/change" \
  "dataset/task1/del" \
  "dataset/task1/del-change" \
  --models "gpt" \
  --record-root "${SESSION_ROOT}" \
  "$@"

run_task_eval \
  "task2" \
  "dataset/task2-line/del" \
  "dataset/task2-line/del-add" \
  "dataset/task2-line/del-change" \
  --models "gpt" \
  --record-root "${SESSION_ROOT}" \
  "$@"

run_task_eval \
  "task3" \
  "dataset/task3-scatter-cluster" \
  --models "gpt" \
  --record-root "${SESSION_ROOT}" \
  "$@"

run_task_eval \
  "task1" \
  "dataset/task1/add" \
  "dataset/task1/add-change" \
  "dataset/task1/change" \
  "dataset/task1/del" \
  "dataset/task1/del-change" \
  --models "gemini" \
  --record-root "${SESSION_ROOT}" \
  "$@"

run_task_eval \
  "task2" \
  "dataset/task2-line/del" \
  "dataset/task2-line/del-add" \
  "dataset/task2-line/del-change" \
  --models "gemini" \
  --record-root "${SESSION_ROOT}" \
  "$@"

run_task_eval \
  "task3" \
  "dataset/task3-scatter-cluster" \
  --models "gemini" \
  --record-root "${SESSION_ROOT}" \
  "$@"

run_task_eval \
  "task1" \
  "dataset/task1/add" \
  "dataset/task1/add-change" \
  "dataset/task1/add-del" \
  "dataset/task1/del-change" \
  --models "claude" \
  --record-root "${SESSION_ROOT}" \
  "$@"

run_task_eval \
  "task2" \
  "dataset/task2-line/del" \
  "dataset/task2-line/del-add" \
  "dataset/task2-line/del-change" \
  --models "claude" \
  --record-root "${SESSION_ROOT}" \
  "$@"

run_task_eval \
  "task3" \
  "dataset/task3-scatter-cluster" \
  --models "claude" \
  --record-root "${SESSION_ROOT}" \
  "$@"

run_task_eval \
  "task1" \
  "dataset/task1/add-change" \
  "dataset/task1/change" \
  --models "qwen" \
  --record-root "${SESSION_ROOT}" \
  "$@"

run_task_eval \
  "task2" \
  "dataset/task2-line/del" \
  "dataset/task2-line/del-add" \
  "dataset/task2-line/del-change" \
  --models "qwen" \
  --record-root "${SESSION_ROOT}" \
  "$@"

run_task_eval \
  "task3" \
  "dataset/task3-scatter-cluster" \
  --models "qwen" \
  --record-root "${SESSION_ROOT}" \
  "$@"
