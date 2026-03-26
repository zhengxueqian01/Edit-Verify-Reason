#!/usr/bin/env bash
# Usage examples:
#   bash run_eval_session.sh
#   bash run_eval_session.sh --models "gpt,gemini" --tasks "task1,task3" --limit 10
#   bash run_eval_session.sh --models gpt --tasks task2 --limit 20
#   bash run_eval_session.sh --output-root output/20260326/my_eval --models gpt --tasks task2
#   bash run_eval_session.sh --output-root output/20260326/my_eval --models gpt --tasks task2 --resume
#
# Notes:
#   - Default output directory: output/YYYYMMDD_HHMMSS
#   - Passing --output-root allows resume from an existing session directory
#   - Results are grouped by model under the session directory
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  PYTHON_BIN="python3"
fi

usage() {
  cat <<'EOF'
Usage:
  bash run_eval_session.sh [options] [-- extra args for src/run_dataset_via_main.py]

Options:
  --models "gpt gemini"   Space/comma separated model list. Default: gpt
  --tasks "task1 task3"   Space/comma separated task list. Default: task1 task2 task3
  --limit N               Per-dataset case limit. Default: 0 (all)
  --qa-index N            QA index. Default: 0
  --output-root DIR       Session output root. If omitted, use output/YYYYMMDD_HHMMSS
  --resume                Resume from existing result.json files. Default: on
  --no-resume             Disable resume and rerun selected cases
  --max-render-retries N  Forwarded to src/run_dataset_via_main.py. Default: 2
  -h, --help              Show this help

Examples:
  bash run_eval_session.sh
  bash run_eval_session.sh --models "gpt,gemini" --tasks "task1,task3" --limit 10
  bash run_eval_session.sh --output-root output/20260326/my_eval --models gpt --tasks task2
EOF
}

normalize_list() {
  local raw="$1"
  raw="${raw//,/ }"
  read -r -a items <<<"${raw}"
  printf '%s\n' "${items[@]}"
}

task_datasets() {
  case "$1" in
    task1)
      cat <<'EOF'
dataset/task1/add
dataset/task1/del
dataset/task1/add-del
dataset/task1/change
dataset/task1/add-change
dataset/task1/del-change
EOF
      ;;
    task2)
      cat <<'EOF'
dataset/task2-line/del
dataset/task2-line/del-add
dataset/task2-line/del-change
EOF
      ;;
    task3)
      cat <<'EOF'
dataset/task3-scatter-cluster
EOF
      ;;
    *)
      echo "error: unsupported task '$1'" >&2
      return 1
      ;;
  esac
}

MODELS_RAW="${MODELS:-gpt}"
TASKS_RAW="${TASKS:-task1 task2 task3}"
LIMIT="${LIMIT:-0}"
QA_INDEX="${QA_INDEX:-0}"
MAX_RENDER_RETRIES="${MAX_RENDER_RETRIES:-2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
RESUME="${RESUME:-1}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models)
      MODELS_RAW="${2:?missing value for --models}"
      shift 2
      ;;
    --tasks)
      TASKS_RAW="${2:?missing value for --tasks}"
      shift 2
      ;;
    --limit)
      LIMIT="${2:?missing value for --limit}"
      shift 2
      ;;
    --qa-index)
      QA_INDEX="${2:?missing value for --qa-index}"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="${2:?missing value for --output-root}"
      shift 2
      ;;
    --resume)
      RESUME="1"
      shift
      ;;
    --no-resume)
      RESUME="0"
      shift
      ;;
    --max-render-retries)
      MAX_RENDER_RETRIES="${2:?missing value for --max-render-retries}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        EXTRA_ARGS+=("$1")
        shift
      done
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

DATE_TIME_STAMP="$(date +%Y%m%d_%H%M%S)"
DEFAULT_OUTPUT_ROOT="${ROOT_DIR}/output/${DATE_TIME_STAMP}"

if [[ -n "${OUTPUT_ROOT}" ]]; then
  if [[ "${OUTPUT_ROOT}" = /* ]]; then
    SESSION_ROOT="${OUTPUT_ROOT}"
  else
    SESSION_ROOT="${ROOT_DIR}/${OUTPUT_ROOT}"
  fi
else
  SESSION_ROOT="${DEFAULT_OUTPUT_ROOT}"
fi

mkdir -p "${SESSION_ROOT}"

MODELS=()
while IFS= read -r model; do
  [[ -n "${model}" ]] && MODELS+=("${model}")
done < <(normalize_list "${MODELS_RAW}")

TASKS=()
while IFS= read -r task; do
  [[ -n "${task}" ]] && TASKS+=("${task}")
done < <(normalize_list "${TASKS_RAW}")

if [[ ${#MODELS[@]} -eq 0 ]]; then
  echo "error: no models selected" >&2
  exit 2
fi

if [[ ${#TASKS[@]} -eq 0 ]]; then
  echo "error: no tasks selected" >&2
  exit 2
fi

echo "session_root=${SESSION_ROOT}"
echo "models=${MODELS[*]}"
echo "tasks=${TASKS[*]}"
echo "limit=${LIMIT}"
echo "qa_index=${QA_INDEX}"
echo "resume=${RESUME}"

for model in "${MODELS[@]}"; do
  model_record_root="${SESSION_ROOT}/${model}"
  mkdir -p "${model_record_root}"
  for task in "${TASKS[@]}"; do
    while IFS= read -r dataset_dir; do
      [[ -n "${dataset_dir}" ]] || continue
      echo "===== model=${model} task=${task} dataset=${dataset_dir} ====="
      cmd=(
        "${PYTHON_BIN}"
        "${ROOT_DIR}/src/run_dataset_via_main.py"
        --input-dir "${dataset_dir}"
        --qa-index "${QA_INDEX}"
        --limit "${LIMIT}"
        --record-root "${model_record_root}"
        --max-render-retries "${MAX_RENDER_RETRIES}"
        --model "${model}"
      )
      if [[ "${RESUME}" != "0" ]]; then
        cmd+=(--resume)
      fi
      if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        cmd+=("${EXTRA_ARGS[@]}")
      fi
      "${cmd[@]}"
    done < <(task_datasets "${task}")
  done
done
