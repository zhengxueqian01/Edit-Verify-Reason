#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

"${ROOT_DIR}/evaluation/task1/run_task1_eval.sh" "$@"
"${ROOT_DIR}/evaluation/task2/run_task2_eval.sh" "$@"
"${ROOT_DIR}/evaluation/task3/run_task3_eval.sh" "$@"
