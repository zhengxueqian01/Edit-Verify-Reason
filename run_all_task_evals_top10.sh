#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODELS="${MODELS:-gpt gemini claude qwen}"
LIMIT="${LIMIT:-10}"
RECORDS_BASE="${RECORDS_BASE:-${ROOT_DIR}/output/dataset_records}"
RUN_PREFIX="${RUN_PREFIX:-top10}"
RESUME="${RESUME:-1}"

if [[ $# -gt 0 ]]; then
  cat >&2 <<'EOF'
Usage:
  bash run_all_task_evals_top10.sh

This script always runs every configured dataset task once:
  - task1: add, del, add-del, change, add-change, del-change
  - task2: del, del-add, del-change
  - task3: scatter-cluster

Use environment variables to change behavior, for example:
  MODELS="gpt gemini" LIMIT=10 RESUME=0 bash run_all_task_evals_top10.sh
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
