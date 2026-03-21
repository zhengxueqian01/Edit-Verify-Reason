#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODELS="${MODELS:-gpt}"
LIMIT="${LIMIT:-5}"
RECORDS_BASE="${RECORDS_BASE:-${ROOT_DIR}/output/dataset_records}"
RUN_PREFIX="${RUN_PREFIX:-smoke5_all_tasks}"
RESUME="${RESUME:-1}"

if [[ $# -gt 0 ]]; then
  cat >&2 <<'EOF'
Usage:
  bash run_eval_session.sh

Defaults:
  MODELS=gpt
  LIMIT=5
  RUN_PREFIX=smoke5_all_tasks
  RESUME=1

This script runs every configured dataset task once:
  - task1: add, del, add-del, change, add-change, del-change
  - task2: del, del-add, del-change
  - task3: scatter-cluster

Useful examples:
  bash run_eval_session.sh
  MODELS="gpt gemini claude qwen" LIMIT=0 RUN_ROOT=/abs/path/to/existing/session bash run_eval_session.sh
  MODELS="gpt" LIMIT=5 RESUME=0 RUN_PREFIX="gpt_smoke5" bash run_eval_session.sh
EOF
  exit 2
fi

mkdir -p "${RECORDS_BASE}"

if [[ -n "${RUN_ROOT:-}" ]]; then
  SESSION_ROOT="${RUN_ROOT}"
elif [[ "${RESUME}" != "0" ]]; then
  SESSION_ROOT="$(find "${RECORDS_BASE}" -maxdepth 1 -type d -name "${RUN_PREFIX}_*" | sort | tail -n 1)"
  if [[ -z "${SESSION_ROOT}" ]]; then
    SESSION_ROOT="${RECORDS_BASE}/${RUN_PREFIX}_$(date +%Y%m%d_%H%M%S)"
  fi
else
  SESSION_ROOT="${RECORDS_BASE}/${RUN_PREFIX}_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "${SESSION_ROOT}"

EXTRA_ARGS=()
if [[ "${RESUME}" != "0" ]]; then
  EXTRA_ARGS+=(--resume)
fi

echo "session_root=${SESSION_ROOT}"
echo "models=${MODELS}"
echo "limit=${LIMIT}"

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  "${ROOT_DIR}/run_all_task_evals.sh" \
    --models "${MODELS}" \
    --limit "${LIMIT}" \
    --record-root "${SESSION_ROOT}" \
    "${EXTRA_ARGS[@]}"
else
  "${ROOT_DIR}/run_all_task_evals.sh" \
    --models "${MODELS}" \
    --limit "${LIMIT}" \
    --record-root "${SESSION_ROOT}"
fi
