#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "${ROOT_DIR}/evaluation/common/run_task_eval_common.sh"

RUN_PREFIX="${RUN_PREFIX:-selected_eval_parallel_$(date +%Y%m%d_%H%M%S)}"
SESSION_ROOT="${SESSION_ROOT:-${ROOT_DIR}/output/dataset_records/${RUN_PREFIX}}"
LOG_DIR="${LOG_DIR:-${SESSION_ROOT}/logs}"

mkdir -p "${LOG_DIR}"

echo "session_root=${SESSION_ROOT}"
echo "log_dir=${LOG_DIR}"

run_gpt() {
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
}

run_gemini() {
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
}

run_claude() {
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
}

run_qwen() {
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
}

PIDS=()
NAMES=()

start_job() {
  local name="$1"
  shift
  (
    "$@"
  ) >"${LOG_DIR}/${name}.log" 2>&1 &
  PIDS+=("$!")
  NAMES+=("${name}")
}

start_job gpt run_gpt "$@"
start_job gemini run_gemini "$@"
start_job claude run_claude "$@"
start_job qwen run_qwen "$@"

echo "started_jobs="
for idx in "${!NAMES[@]}"; do
  echo "  ${NAMES[$idx]} pid=${PIDS[$idx]} log=${LOG_DIR}/${NAMES[$idx]}.log"
done

FAILURES=0
for idx in "${!PIDS[@]}"; do
  pid="${PIDS[$idx]}"
  name="${NAMES[$idx]}"
  if wait "${pid}"; then
    echo "[ok] ${name}"
  else
    echo "[fail] ${name} (see ${LOG_DIR}/${name}.log)"
    FAILURES=$((FAILURES + 1))
  fi
done

if [[ "${FAILURES}" -ne 0 ]]; then
  echo "parallel run finished with ${FAILURES} failure(s)"
  exit 1
fi

echo "parallel run finished successfully"
