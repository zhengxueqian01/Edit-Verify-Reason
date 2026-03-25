#!/usr/bin/env bash
set -euo pipefail

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${COMMON_DIR}/../.." && pwd)"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  PYTHON_BIN="python3"
fi

MODEL_LIST_DEFAULT=(gpt gemini claude qwen)

usage_common() {
  cat <<'EOF'
Usage:
  run_task_eval <task-name> <dataset-dir> [dataset-dir ...] [--qa-index N] [--limit N] [--record-root DIR] [--models "gpt gemini"] [--resume|--no-resume] [extra args...]

Options:
  --qa-index N       QA index passed to src/run_dataset_via_main.py (default: 0)
  --limit N          Max cases per dataset folder (default: 0, all)
  --record-root DIR  Base output directory (default: output/dataset_records)
  --models "..."     Space/comma separated model list (default: gpt gemini claude qwen)
  --resume           Resume from existing result.json files when present (default: on)
  --no-resume        Disable resume and re-run all selected cases

Any remaining args are forwarded to src/run_dataset_via_main.py.
EOF
}

normalize_models() {
  local raw="$1"
  raw="${raw//,/ }"
  read -r -a models <<<"${raw}"
  if [[ ${#models[@]} -eq 0 ]]; then
    printf '%s\n' "${MODEL_LIST_DEFAULT[@]}"
    return
  fi
  printf '%s\n' "${models[@]}"
}

run_task_eval() {
  if [[ $# -lt 2 ]]; then
    usage_common >&2
    return 2
  fi

  local task_name="$1"
  shift

  local -a dataset_dirs=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --*)
        break
        ;;
      *)
        dataset_dirs+=("$1")
        shift
        ;;
    esac
  done

  if [[ ${#dataset_dirs[@]} -eq 0 ]]; then
    echo "error: at least one dataset dir is required for ${task_name}" >&2
    return 2
  fi

  local qa_index="0"
  local limit="0"
  local record_root_base="output/dataset_records"
  local models_raw="${MODEL_LIST_DEFAULT[*]}"
  local resume="1"
  local -a extra_args=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --qa-index)
        qa_index="${2:?missing value for --qa-index}"
        shift 2
        ;;
      --limit)
        limit="${2:?missing value for --limit}"
        shift 2
        ;;
      --record-root)
        record_root_base="${2:?missing value for --record-root}"
        shift 2
        ;;
      --models)
        models_raw="${2:?missing value for --models}"
        shift 2
        ;;
      --resume)
        resume="1"
        shift
        ;;
      --no-resume)
        resume="0"
        shift
        ;;
      --help|-h)
        usage_common
        return 0
        ;;
      *)
        extra_args+=("$1")
        shift
        ;;
    esac
  done

  local -a models=()
  while IFS= read -r model; do
    [[ -n "${model}" ]] && models+=("${model}")
  done < <(normalize_models "${models_raw}")

  local model
  local dataset_dir
  local record_root
  local -a cmd

  for model in "${models[@]}"; do
    record_root="${record_root_base%/}/${model}"
    echo "===== ${task_name}: model=${model}, record_root=${record_root} ====="
    for dataset_dir in "${dataset_dirs[@]}"; do
      echo "--- dataset=${dataset_dir}"
      cmd=(
        "${PYTHON_BIN}"
        "${PROJECT_ROOT}/src/run_dataset_via_main.py"
        --input-dir "${dataset_dir}"
        --qa-index "${qa_index}"
        --limit "${limit}"
        --record-root "${record_root}"
      )
      if [[ "${resume}" != "0" ]]; then
        cmd+=(--resume)
      fi
      if [[ ${#extra_args[@]} -gt 0 ]]; then
        cmd+=("${extra_args[@]}")
      fi
      env \
        DEFAULT_MODEL="${model}" \
        SPLITTER_MODEL="${model}" \
        PLANNER_MODEL="${model}" \
        TOOL_PLANNER_MODEL="${model}" \
        EXECUTOR_MODEL="${model}" \
        ROUTER_MODEL="${model}" \
        VALIDATOR_MODEL="${model}" \
        ANSWER_MODEL="${model}" \
        "${cmd[@]}"
    done
  done
}
