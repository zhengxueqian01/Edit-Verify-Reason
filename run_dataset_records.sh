#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage:
  ./run_dataset_records.sh <input-dir> [qa-index] [limit] [record-root] [extra args...]

Examples:
  ./run_dataset_records.sh dataset/task1-mix-area/add-change
  ./run_dataset_records.sh dataset/task1-mix-area/add-change 0 10
  ./run_dataset_records.sh dataset/task1-mix-area/add-change 0 0 output/dataset_records
EOF
  exit 1
fi

INPUT_DIR="$1"
QA_INDEX="${2:-0}"
LIMIT="${3:-0}"
RECORD_ROOT="${4:-output/dataset_records}"

EXTRA_ARGS=()
if [[ $# -gt 4 ]]; then
  EXTRA_ARGS=("${@:5}")
fi

CMD=(
  python3
  "$ROOT_DIR/src/run_dataset_via_main.py"
  --input-dir "$INPUT_DIR"
  --qa-index "$QA_INDEX"
  --limit "$LIMIT"
  --record-root "$RECORD_ROOT"
)

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
