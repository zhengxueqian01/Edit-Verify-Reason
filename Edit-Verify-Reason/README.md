# Edit-Verify-Reason

[中文说明](README_CN.md)

This repository contains the current Edit-Verify-Reason framework, evaluation scripts, and a local dataset snapshot for SVG chart editing and question answering.

Important: this README lives under `Edit-Verify-Reason/`, but the commands below are intended to be run from the repository root.

## What It Does

Edit-Verify-Reason processes a chart task in six stages:

1. Split the user request into update intent and QA target
2. Perceive the existing SVG chart structure
3. Plan the edit steps
4. Edit the SVG step by step
5. Render the result and validate it
6. Answer the question on the updated chart, optionally with visual tool augmentation

Currently supported chart types:

- `area`
- `line`
- `scatter`

## Current Layout

Main directories in the current repo:

- `Edit-Verify-Reason/`: framework code
- `Edit-Verify-Reason/main.py`: single-case entry
- `Edit-Verify-Reason/web_app.py`: local web entry
- `Edit-Verify-Reason/chart_agent/`: framework package
- `Edit-Verify-Reason/js/`: JS/Node helper scripts
- `Evaluation/`: batch runners, validation scripts, smoke/ablation helpers
- `CMod-QA/`: dataset snapshot currently present in this repo
- `Edit-Verify-Reason/tests/`: framework pytest tests
- `requirements.txt`: Python dependencies at repo root

## Setup

### Python

CLI and evaluation scripts run on the current Python environment. The web entry still requires Python 3.12 or below because it imports `cgi`.

Recommended setup from repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest
```

### `resvg`

SVG rendering depends on `resvg` being available on `PATH`.

macOS example:

```bash
brew install resvg
which resvg
```

### Model Credentials

Set keys through environment variables or `.env`:

```env
Aihubmix_API_KEY=
Siliconflow_API_KEY=
Doubao_API_KEY=
```

Common model and runtime variables:

```env
DEFAULT_MODEL=gpt

GPT_MODEL=gpt-5.2
GPT_BASE_URL=https://aihubmix.com/v1
GPT_TEMPERATURE=0.0

QWEN_MODEL=Qwen/Qwen3-VL-235B-A22B-Instruct
QWEN_BASE_URL=https://api.siliconflow.cn/v1
QWEN_TEMPERATURE=0.0

DOUBAO_MODEL=doubao-seed-1-6-251015
DOUBAO_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
DOUBAO_TEMPERATURE=0.0

CLAUDE_MODEL=claude-sonnet-4-6
CLAUDE_BASE_URL=https://aihubmix.com/v1
CLAUDE_TEMPERATURE=0.0

GEMINI_MODEL=gemini-2.5-flash
GEMINI_BASE_URL=https://aihubmix.com/v1
GEMINI_TEMPERATURE=0.0

SPLITTER_MODEL=gpt
PLANNER_MODEL=gpt
TOOL_PLANNER_MODEL=gpt
EXECUTOR_MODEL=gpt
VALIDATOR_MODEL=qwen
ANSWER_MODEL=gpt

PERCEPTION_MAX_RETRIES=2
SVG_PERCEPTION_MODE=rules
SVG_UPDATE_MODE=rules
CHART_WEB_HOST=127.0.0.1
CHART_WEB_PORT=8008
```

Available model keys:

- `gpt`
- `qwen`
- `doubao`
- `claude`
- `gemini`
- `qwen-aihubmix`

Mode normalization:

- `SVG_PERCEPTION_MODE`: `rules`, `llm_summary`, alias `llm`
- `SVG_UPDATE_MODE`: `rules`, `llm_intent`, `htn`, aliases `llm`, `hier`, `hierarchical`

## Data Layout

The dataset currently present in this repo is `CMod-QA/`, not `dataset/`.

Current task folders:

- `CMod-QA/task1-area/`
- `CMod-QA/task2-line/`
- `CMod-QA/task3-scatter/`

Each case folder typically contains:

- `{id}.svg`
- `{id}.png`
- `{id}.json`
- `{id}.csv`
- `{id}_aug.svg`
- `{id}_aug.png`

Important distinction:

- `Edit-Verify-Reason/main.py` only reads the question, image, and SVG you pass in
- `Evaluation/run_dataset_via_main.py` also reads the case JSON and injects `data_change`

That means dataset cases with phrases like `these points` or `listed value revisions` should usually be run through the batch runner or the Python API with `structured_update_context`, not through the bare single-case CLI alone.

## How To Use

### 1. Single-Case CLI

Use this when the update instruction is already fully expressed in the question text.

Command:

```bash
python Edit-Verify-Reason/main.py \
  --question "After deleting the category Aethelgard Medical Board, how many times do the lines for Lumina Health Authority and Stellar Public Health intersect?" \
  --image CMod-QA/task2-line/del/000/000.png \
  --svg CMod-QA/task2-line/del/000/000.svg \
  --experiment-mode full
```

Available CLI flags:

- `--question`
- `--image`
- `--svg`
- `--text_spec`
- `--experiment-mode`

Supported experiment modes:

- `full`
- `wo_question_decomposition`
- `wo_svg_update`

### 2. Dataset / Batch Runner

This is the recommended entry for dataset cases because it reads `data_change` from the case JSON.

Tested command:

```bash
python Evaluation/run_dataset_via_main.py \
  --input-dir CMod-QA/task1-area/add-change \
  --qa-index 0 \
  --limit 1 \
  --max-render-retries 2 \
  --record-root output/dataset_records_stage2_check
```

That command was verified in this repo and produced a successful case record for `000`.

Useful flags:

- `--input-dir`
- `--qa-index`
- `--max-render-retries`
- `--limit`
- `--record-root`
- `--resume`
- `--model`
- `--experiment-mode`

Another example:

```bash
python Evaluation/run_dataset_via_main.py \
  --input-dir CMod-QA/task2-line/del \
  --qa-index 0 \
  --limit 10 \
  --resume \
  --model gpt
```

### 3. Python API

Use `run_main()` directly if you need model overrides or want to inject structured update context yourself.

```python
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()
FRAMEWORK_ROOT = PROJECT_ROOT / "Edit-Verify-Reason"
sys.path.insert(0, str(FRAMEWORK_ROOT))

from main import run_main

case_dir = PROJECT_ROOT / "CMod-QA" / "task1-area" / "add-change" / "000"
payload = json.loads((case_dir / "000.json").read_text(encoding="utf-8"))
qa = payload["QA"][0]

result = run_main(
    {
        "question": qa["question"],
        "image_path": str(case_dir / "000.png"),
        "svg_path": str(case_dir / "000.svg"),
        "structured_update_context": {
            "data_change": payload.get("data_change", {}),
        },
        "max_render_retries": 2,
        "experiment_mode": "full",
        "model_overrides": {
            "splitter": "gpt",
            "planner": "gpt",
            "executor": "gpt",
            "answer": "gpt",
            "tool_planner": "gpt",
        },
    }
)
```

Common `run_main()` inputs:

- `question`
- `image_path`
- `svg_path`
- `text_spec`
- `max_render_retries`
- `render_output_dir`
- `experiment_mode`
- `svg_perception_mode`
- `svg_update_mode`
- `model_overrides`
- `structured_update_context`
- `update_question`
- `qa_question`
- `auto_split_question`
- `answer_image_path`

### 4. Web UI

Start from repo root:

```bash
python Edit-Verify-Reason/web_app.py
```

Default URL:

```text
http://127.0.0.1:8008
```

Web endpoints:

- `GET /`
- `GET /api/models`
- `POST /api/run`
- `POST /api/run/stream`
- `GET /files/...`

Current limitation:

- `Edit-Verify-Reason/web_app.py` imports `cgi`
- on Python 3.13+, it fails with `ModuleNotFoundError: No module named 'cgi'`
- use Python 3.12 or below for the web UI

### 5. Validation Scripts

Compare generated SVGs with ground truth:

```bash
python Evaluation/validate_svg_matches.py \
  --pred-root output/dataset_records_stage2_check/add_change \
  --dataset-dir CMod-QA/task1-area/add-change
```

Area numeric validation:

```bash
python Evaluation/validate_aug_svg.py \
  --pred-root output/dataset_records_stage2_check/add_change \
  --gt-root CMod-QA/task1-area/add-change \
  --mode top-only
```

Validate every task under one records root:

```bash
python Evaluation/run_validate_dataset_records.py \
  --records-root output/dataset_records_stage2_check \
  --limit 20
```

Re-answer from an existing tool-augmented image:

```bash
python Evaluation/run_tool_aug_answer.py \
  --case-dir output/dataset_records_stage2_check/add_change/000 \
  --resume
```

Optional visualization runner:

```bash
python Evaluation/run_dataset_vis_main.py \
  --task task1-area/add-change \
  --dataset-root CMod-QA \
  --limit 1 \
  --out output/vis_report.json
```

## Outputs

Typical single-run JSON fields:

- `output_image_path`
- `render_check`
- `attempt_logs`
- `operation_plan`
- `question_split`
- `perception_steps`
- `answer_original`
- `answer_initial`
- `answer_tool_augmented`
- `answer`
- `resolved_task_models`
- `experiment_mode`

Typical batch outputs:

- `<run_dir>/<case_id>/result.json`
- `<run_dir>/<case_id>/answer.txt`
- `<run_dir>/<case_id>/error.txt`
- `<run_dir>/<case_id>/<case_id>_updated.svg`
- `<run_dir>/<case_id>/<case_id>_updated.png`
- `<run_dir>/<case_id>/<case_id>_tool_aug.svg`
- `<run_dir>/<case_id>/<case_id>_tool_aug.png`
- `<run_dir>/summary.json`
- `<run_dir>/summary.txt`

Example output from the verified batch run:

- `output/dataset_records_stage2_check/add_change/000/result.json`
- `output/dataset_records_stage2_check/add_change/summary.txt`

## Visual Tool Phase

The current trigger threshold in `Edit-Verify-Reason/main.py` is:

- run tool augmentation when `answer_initial.confidence < 0.85`

It is also forced for:

- scatter clustering questions
- line intersection / crossing questions

Registered tools live in `Edit-Verify-Reason/chart_agent/core/vision_tool_phase.py`.

## Tests

Run all tests:

```bash
pytest -q Edit-Verify-Reason/tests
```

Useful focused tests:

```bash
pytest -q Edit-Verify-Reason/tests/test_main_question_split.py
pytest -q Edit-Verify-Reason/tests/test_main_step_retry.py
pytest -q Edit-Verify-Reason/tests/test_render_validator.py
pytest -q Edit-Verify-Reason/tests/test_area_svg_updater.py
pytest -q Edit-Verify-Reason/tests/test_scatter_svg_updater.py
```

## JS Helpers

JS dependencies live under `Edit-Verify-Reason/js/package.json`.

Install them with:

```bash
cd Edit-Verify-Reason/js
npm install
```

Current limitation:

- `Edit-Verify-Reason/js/run_dataset_case.py`
- `Edit-Verify-Reason/js/run_dataset_case.mjs`

still assume a top-level `dataset/` directory, so they are not directly usable with the current `CMod-QA/` snapshot unless you add a compatible dataset mirror or patch the paths.

## Current Limitations

- The web entry requires Python 3.12 or below
- Several convenience evaluation scripts still hardcode `dataset/...` task specs:
- `Evaluation/run_smoke_task1_task2_models.py`
- `Evaluation/run_ablation_gpt_10cases.py`
- `Evaluation/run_eval_session.sh`
- The repo snapshot currently contains `CMod-QA/`, not `dataset/`
- Bare single-case CLI does not automatically load `data_change` from case JSON
- Only framework tests are kept under `Edit-Verify-Reason/tests/`; evaluation-script tests were removed
- `requirements.txt` does not install `pytest`
