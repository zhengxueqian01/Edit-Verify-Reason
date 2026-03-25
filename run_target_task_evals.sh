#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "${ROOT_DIR}/evaluation/common/run_task_eval_common.sh"

# run_task_eval \
#   "task1" \
#   "dataset/task1-mix-area/add" \
#   --models "gpt gemini claude qwen" \
#   "$@"

# run_task_eval \
  # "task2" \
  # "dataset/task2-line/del" \
  # --models "gpt gemini claude qwen" \
  # "$@"

run_task_eval \
  "task3" \
  "dataset/task3-scatter-cluster" \
  --models "gpt gemini claude qwen" \
  "$@"
