# ChartAgent

Agent-style chart editing and QA on SVG charts.

## Current Scope

- Main entry: `python -m src.main`
- Input mode: SVG-based update (`--svg` is required in current implementation)
- Supported chart types for SVG update: `scatter`, `line`, `area`
- Pipeline: question split -> perception -> planned update steps -> SVG/PNG render -> render validation -> QA answer generation
- Batch runner: `python -m src.run_dataset_via_main`
- Optional web server: `python -m src.web_app`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install `resvg` (required for SVG->PNG rendering):

```bash
brew install resvg
```

## Environment Variables

Set API keys in `.env` (or system env):

```env
Aihubmix_API_KEY_ZZT=
Siliconflow_API_KEY=
Doubao_API_KEY=
```

Optional model/runtime overrides:

```env
GPT_MODEL=gpt-5.2
GPT_BASE_URL=https://aihubmix.com/v1
PERCEPTION_MAX_RETRIES=2
SPLITTER_MODEL=gpt
PLANNER_MODEL=gpt
EXECUTOR_MODEL=gpt
ANSWER_MODEL=gpt
```

## Run Single Case

Line chart example:

```bash
python -m src.main \
  --question "Delete one series and tell me which year has the maximum value" \
  --image dataset/task2-line/094/094.png \
  --svg dataset/task2-line/094/094.svg
```

Scatter example:

```bash
python -m src.main \
  --question "Add points (10, 20) and (15, 18), then how many clusters are there?" \
  --image dataset/task3-scatter-cluster/094/094.png \
  --svg dataset/task3-scatter-cluster/094/094.svg
```

## Run Batch Dataset

```bash
python -m src.run_dataset_via_main \
  --input-dir dataset/task2-line \
  --qa-index 0 \
  --max-render-retries 2
```

Records are written under `output/dataset_records/<model>_<task>_<timestamp>/`.

## Run Web UI

```bash
python -m src.web_app
```

Then open `http://127.0.0.1:8008`.

## Output

- Rendered files: `output/<chart_type>/`
- Dataset run records: `output/dataset_records/...`
- `src.main` prints a JSON result including:
  - `output_image_path`
  - `render_check`
  - `attempt_logs`
  - `answer`, `answer_initial`, `answer_tool_augmented`
