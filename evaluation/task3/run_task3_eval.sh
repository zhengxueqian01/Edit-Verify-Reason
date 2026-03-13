#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../common/run_task_eval_common.sh"

run_task_eval \
  "task3" \
  "dataset/task3-scatter-cluster" \
  "$@"
