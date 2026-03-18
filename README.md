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
Aihubmix_API_KEY=
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

Records are written under `output/dataset_records/<category>_<task>_<timestamp>/`, for example `task1_add_20260313_153000/`.

## Validate SVG Match

Run SVG-vs-ground-truth validation separately:

```bash
python -m src.validate_svg_matches \
  --pred-root output/dataset_records/task2_del_20260303_201506 \
  --dataset-dir dataset/task2-line
```

Notes:

- `--pred-root` and `--dataset-dir` must be passed explicitly.
- Default output path is `output/svg_match/<pred-root>.json`.
- To validate a single pair:

```bash
python -m src.validate_svg_matches \
  --pred-svg output/example.svg \
  --gt-svg dataset/task2-line/000/000_del.svg
```

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
  - `answer_initial`: the initial QA result
  - `answer_tool_augmented`: produced only when the initial answer has low confidence and tool-based image augmentation is triggered
  - `answer`: the final adopted answer; usually the tool-augmented answer if augmentation runs successfully, otherwise the initial answer

## Tool-Based Image Augmentation

- `src.main` first produces `answer_initial` and reads its `confidence`.
- Tool-based image augmentation is triggered only when `answer_initial.confidence` is low.
- `answer_initial.confidence` is now normalized to the `0-1` range before it is used for routing.
- The current automatic visual tools are `add_point`, `draw_line`, `highlight_rect`, `isolate_color_topology`, `isolate_all_color_topologies`, `draw_global_peak_crosshairs`, and `zoom_and_highlight_intersection`.
- `add_text` has been removed from the autonomous tool selection stage.
- `isolate_color_topology` is designed for single-color scatter questions: it fades non-target colors and draws convex hulls around DBSCAN clusters of the target-color points.
- `isolate_all_color_topologies` is designed for global scatter clustering questions: it clusters points separately for each detected color and draws hulls for all color-specific clusters.
- `draw_global_peak_crosshairs` is designed for area charts: it runs with zero parameters, scans global area vertices, and draws horizontal/vertical crosshairs at the absolute peak.
- `zoom_and_highlight_intersection` is designed for line charts: it resolves `line_A` and `line_B` from the legend, computes segment intersections, and injects red intersection markers.

## TODO

- Validate the match quality between generated SVGs and ground truth.
- Optimize the current dataset.
- Evaluate dataset performance across four models.
- Optimize the validation script.
- Add web-side image enhancement tools so users can manually improve chart visibility and send the enhanced image to the model for better results.
