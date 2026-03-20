# Pipeline Walkthrough

This document rewrites the pipeline explanation in full detail using the real case:

- `dataset/task2-line/del-add/001`

The goal is not just to describe the stages, but to list the concrete parameters entering and leaving each stage.

## 1. Source Dataset Case

Files under this case:

- `dataset/task2-line/del-add/001/001.json`
- `dataset/task2-line/del-add/001/001.svg`
- `dataset/task2-line/del-add/001/001.png`
- `dataset/task2-line/del-add/001/001.csv`
- `dataset/task2-line/del-add/001/001_aug.svg`
- `dataset/task2-line/del-add/001/001_aug.png`

The core dataset payload from `001.json` is:

```json
{
  "id": "001",
  "chart_type": "line",
  "operation": "del+add",
  "operation_target": {
    "category_name": ["Stellar Cultural Development Agency"],
    "add_category": "Mariner Systems"
  },
  "data_change": {
    "add": {
      "mode": "full_series",
      "years": [
        "1958", "1959", "1960", "1961", "1962",
        "1963", "1964", "1965", "1966", "1967",
        "1968", "1969", "1970", "1971", "1972"
      ],
      "values": [
        61656052.94, 29088578.52, 60841542.31, 27550184.99, 59772165.2,
        29469669.71, 64297153.54, 30732810.16, 64534250.46, 30543413.67,
        61012706.65, 30967944.43, 64010036.85, 29310612.84, 63498680.34
      ]
    }
  },
  "QA": [
    {
      "question": "After deleting the category Stellar Cultural Development Agency and adding the category Mariner Systems, how many times do the lines for Aethelgard Cultural Council and Mariner Systems intersect?",
      "answer": 14
    }
  ]
}
```

Meaning:

- Delete one existing line series: `Stellar Cultural Development Agency`
- Add one new line series: `Mariner Systems`
- Then answer an intersection-count question:
  `How many times do Aethelgard Cultural Council and Mariner Systems intersect?`

Ground truth answer:

- `14`

## 2. Batch Entry Command

The batch runner is:

- `python -m src.run_dataset_via_main`

Example command for this folder:

```bash
python -m src.run_dataset_via_main \
  --input-dir dataset/task2-line/del-add \
  --qa-index 0 \
  --max-render-retries 2
```

Runner arguments:

- `--input-dir`
  - type: string
  - meaning: dataset folder containing case subfolders such as `000`, `001`, ...
  - here: `dataset/task2-line/del-add`
- `--qa-index`
  - type: int
  - meaning: which QA entry in `QA[]` to use
  - here: `0`
- `--max-render-retries`
  - type: int
  - meaning: how many extra retries after the first attempt
  - here: `2`
- `--limit`
  - type: int
  - default: `0`
  - meaning: case cap, `0` means all
- `--record-root`
  - type: string
  - default: `output/dataset_records`
- `--resume`
  - type: bool flag
  - meaning: skip cases with an existing `result.json`

## 3. Case-Level Parameters Selected By The Batch Runner

Inside `src/run_dataset_via_main.py`, for case `001`, the runner resolves:

- `case_id = "001"`
- `json_path = dataset/task2-line/del-add/001/001.json`
- `svg_path = dataset/task2-line/del-add/001/001.svg`
- `image_path = dataset/task2-line/del-add/001/001.png`
- `case_out_dir = <record_root>/<run_dir>/001`

Then it reads the QA item with `qa_index = 0`:

- `qa_question`
  - value:
    `After deleting the category Stellar Cultural Development Agency and adding the category Mariner Systems, how many times do the lines for Aethelgard Cultural Council and Mariner Systems intersect?`
- `expected_answer`
  - value: `14`
- `qa_item`
  - value: the full first object inside `QA[]`

## 4. Structured Update Context

Current `build_structured_update_context()` keeps only execution-critical structured fields:

- `operation_target`
- `data_change`

For this case, the exact object is:

```json
{
  "operation_target": {
    "category_name": ["Stellar Cultural Development Agency"],
    "add_category": "Mariner Systems"
  },
  "data_change": {
    "add": {
      "mode": "full_series",
      "years": [
        "1958", "1959", "1960", "1961", "1962",
        "1963", "1964", "1965", "1966", "1967",
        "1968", "1969", "1970", "1971", "1972"
      ],
      "values": [
        61656052.94, 29088578.52, 60841542.31, 27550184.99, 59772165.2,
        29469669.71, 64297153.54, 30732810.16, 64534250.46, 30543413.67,
        61012706.65, 30967944.43, 64010036.85, 29310612.84, 63498680.34
      ]
    }
  }
}
```

Removed from this payload:

- `chart_type`
- `operation`
- `cluster_params`
- `task`

## 5. Exact `run_main()` Input

The batch runner calls `run_main()` with:

```json
{
  "question": "After deleting the category Stellar Cultural Development Agency and adding the category Mariner Systems, how many times do the lines for Aethelgard Cultural Council and Mariner Systems intersect?",
  "max_render_retries": 2,
  "svg_path": "dataset/task2-line/del-add/001/001.svg",
  "image_path": "dataset/task2-line/del-add/001/001.png",
  "structured_update_context": {
    "operation_target": {
      "category_name": ["Stellar Cultural Development Agency"],
      "add_category": "Mariner Systems"
    },
    "data_change": {
      "add": {
        "mode": "full_series",
        "years": [
          "1958", "1959", "1960", "1961", "1962",
          "1963", "1964", "1965", "1966", "1967",
          "1968", "1969", "1970", "1971", "1972"
        ],
        "values": [
          61656052.94, 29088578.52, 60841542.31, 27550184.99, 59772165.2,
          29469669.71, 64297153.54, 30732810.16, 64534250.46, 30543413.67,
          61012706.65, 30967944.43, 64010036.85, 29310612.84, 63498680.34
        ]
      }
    }
  },
  "text_spec": null
}
```

## 6. Runtime Stage 1: Model Setup

`run_main()` creates five LLM configs and instances:

- `splitter`
  - purpose: split full question into update part and QA part
- `planner`
  - purpose: plan operations and ordered edit steps
- `executor`
  - purpose: help SVG updater parse values/labels when regex is insufficient
- `answer`
  - purpose: answer questions on original or updated images
- `tool_planner`
  - purpose: decide visual augmentation tool calls

Other runtime-level parameters:

- `svg_update_mode`
  - source: `inputs.get("svg_update_mode")`
  - not explicitly set by this batch runner
- `svg_perception_mode`
  - source: `inputs.get("svg_perception_mode")`
  - not explicitly set by this batch runner
- `model_overrides`
  - source: `inputs.get("model_overrides")`
  - not explicitly set here

## 7. Runtime Stage 2: Question Split

Input to `_resolve_questions()`:

- `raw_question`
  - the full natural-language question from dataset QA
- `raw_update`
  - empty string in this case
- `raw_qa`
  - empty string in this case
- `structured_update_context`
  - the object shown above

Target output fields:

- `update_question`
  - only the chart-edit instruction
- `qa_question`
  - only the final analysis question
- `split_info`
  - metadata about the split
- `split_data_change`
  - any structured `data_change` parsed from the natural-language question

For this case, the intended logical result is:

- `update_question`
  - approximately:
    `Delete the category Stellar Cultural Development Agency. Add the category Mariner Systems.`
- `qa_question`
  - approximately:
    `How many times do the lines for Aethelgard Cultural Council and Mariner Systems intersect?`

Important:

- Even if the split is imperfect, `operation_target` and `data_change` are merged back afterward.
- This means the exact add-series values are still available as structured data.

## 8. Runtime Stage 3: Original-Image Answer

Before applying any edit, `run_main()` calls `answer_question()` on the original PNG.

Input object stored as `answer_original_input`:

```json
{
  "question": "How many times do the lines for Aethelgard Cultural Council and Mariner Systems intersect?",
  "output_image_path": "dataset/task2-line/del-add/001/001.png",
  "data_summary": {}
}
```

Call parameters:

- `qa_question`
- `data_summary = {}`
- `output_image_path = original PNG`
- `image_context_note = "This is the original chart image before any requested update is applied."`
- `llm = answer_llm`

Output field:

- `answer_original`

## 9. Runtime Stage 4: Attempt Loop

Attempt loop control parameters:

- `max_render_retries = 2`
- total possible attempts = `retries + 1 = 3`

Attempt-local mutable variables:

- `attempt`
- `planned_question`
- `retry_hint`
- `attempt_logs`
- `last_state`
- `last_output_image`
- `last_operation_plan`
- `last_render_check`
- `last_perception_steps`

Initially:

- `planned_question = update_question`
- `retry_hint = ""`

## 10. Attempt Stage A: `run_perception()`

For each attempt, `run_main()` prepares:

```json
{
  "question": "<planned_question>",
  "max_render_retries": 2,
  "svg_path": "dataset/task2-line/del-add/001/001.svg",
  "image_path": "dataset/task2-line/del-add/001/001.png",
  "structured_update_context": { "...": "..." },
  "text_spec": null
}
```

Then `run_perception(inputs)` executes these sub-actions:

1. `DETECT_INPUT_MODE`
2. `PARSE_QUESTION`
3. `PERCEIVE_IMAGE_SVG`
4. `SANITY_CHECK`

### 10.1 `PARSE_QUESTION`

`parse_question(question, llm, chart_type_hint)` returns:

- `update_spec`
- `issues`

Typical `update_spec` fields used by this system:

- `new_points`
- `raw`
- other parser-specific fields

For this line `del+add` case, `new_points` is not the key field; the structured add-series values come mainly from `data_change`.

### 10.2 `PERCEIVE_IMAGE_SVG`

`perceive_svg(svg_path, question, llm, perception_mode)` returns:

- `chart_type`
- `chart_type_confidence`
- `perception_mode`
- `mapping_ok`
- `mapping_confidence`
- `primitives_summary`
- `mapping_info`
- `issues`
- `suggested_next_actions`

For a line chart, `mapping_info` is especially important. Key fields:

- `axes_bounds`
- `x_ticks`
  - list of `(pixel_x, data_value)`
- `y_ticks`
  - list of `(pixel_y, data_value)`
- `existing_points_svg`
  - mainly for scatter, usually irrelevant here
- `existing_point_colors`
  - mainly for scatter
- `x_labels`
- `area_top_boundary`
  - mainly for area
- `area_fills`
  - mainly for area

For this case, line execution mainly depends on:

- `x_ticks`
- `y_ticks`
- line legend parsing later from SVG content

`primitives_summary` fields:

- `num_circles`
- `num_points`
- `num_xticks`
- `num_yticks`
- `num_areas`
- `num_lines`

Then `_resolve_supported_chart_type()` decides the effective chart type:

- final result for this case: `line`

## 11. Attempt Stage B: LLM Update Plan

`_llm_plan_update(question, chart_type, llm, retry_hint)` receives:

- `question = planned_question`
- `chart_type = "line"`
- `retry_hint = ""` on the first attempt

Requested JSON schema:

- `operation`
- `normalized_question`
- `steps`
- `new_points`
- `retry_hint`

Each `steps[]` item may contain:

- `operation`
- `question_hint`
- `operation_target`
- `data_change`
- `new_points`

For this case, the intended plan is logically:

```json
{
  "operation": "delete",
  "normalized_question": "Delete the category Stellar Cultural Development Agency. Add the category Mariner Systems.",
  "steps": [
    {
      "operation": "delete",
      "question_hint": "Delete the category Stellar Cultural Development Agency"
    },
    {
      "operation": "add",
      "question_hint": "Add the category Mariner Systems"
    }
  ],
  "new_points": []
}
```

The exact LLM output can vary, but the next stage enriches or rebuilds steps using structured data.

## 12. Attempt Stage C: Step Construction

`_operation_steps_from_plan(operation_plan, planned_question, structured_context)` builds the executable steps.

### 12.1 Structured Data Available

Structured inputs at this point:

- `operation_target = {"category_name": ["Stellar Cultural Development Agency"], "add_category": "Mariner Systems"}`
- `data_change.add.mode = "full_series"`
- `data_change.add.years = ["1958", ..., "1972"]`
- `data_change.add.values = [61656052.94, ..., 63498680.34]`

### 12.2 Final Executable Steps For This Case

The intended final step objects are:

Step 1:

```json
{
  "operation": "delete",
  "operation_target": {
    "category_name": "Stellar Cultural Development Agency"
  },
  "data_change": {},
  "question_hint": "Delete the category Stellar Cultural Development Agency",
  "new_points": []
}
```

Step 2:

```json
{
  "operation": "add",
  "operation_target": {
    "category_name": "Mariner Systems"
  },
  "data_change": {
    "mode": "full_series",
    "years": [
      "1958", "1959", "1960", "1961", "1962",
      "1963", "1964", "1965", "1966", "1967",
      "1968", "1969", "1970", "1971", "1972"
    ],
    "values": [
      61656052.94, 29088578.52, 60841542.31, 27550184.99, 59772165.2,
      29469669.71, 64297153.54, 30732810.16, 64534250.46, 30543413.67,
      61012706.65, 30967944.43, 64010036.85, 29310612.84, 63498680.34
    ]
  },
  "question_hint": "Add the category Mariner Systems",
  "new_points": []
}
```

## 13. Attempt Stage D: Rendered Step Questions

Before each step is executed, `_render_structured_step_question(step)` converts it to a natural-language execution question.

Step 1 rendered question:

```text
Delete the category/series "Stellar Cultural Development Agency"
```

Step 2 rendered question:

```text
Add the category/series "Mariner Systems" with values [61656052.94, 29088578.52, 60841542.31, 27550184.99, 59772165.2, 29469669.71, 64297153.54, 30732810.16, 64534250.46, 30543413.67, 61012706.65, 30967944.43, 64010036.85, 29310612.84, 63498680.34]
```

These two strings are the direct `question` arguments for the line SVG updater path.

## 14. Attempt Stage E: Step Output Paths

`_step_paths(svg_path, chart_type, idx, total)` uses `default_output_paths()` first.

For source SVG `dataset/task2-line/del-add/001/001.svg`:

- `case_id = 001`
- `chart_type = line`
- dataset JSON `operation = del+add`
- normalized operation = `del-add`

So the final output paths are:

- final SVG:
  `output/line/001_line_del-add_updated.svg`
- final PNG:
  `output/line/001_line_del-add_updated.png`

Because there are 2 steps:

- Step 1 output SVG:
  `output/line/001_line_del-add_updated_step1.svg`
- Step 1 output PNG:
  `output/line/001_line_del-add_updated_step1.png`
- Step 2 output SVG:
  `output/line/001_line_del-add_updated.svg`
- Step 2 output PNG:
  `output/line/001_line_del-add_updated.png`

## 15. Attempt Stage F: Step Execution Wrapper

For each step, `_execute_planned_steps()` prepares:

- `step_q`
  - rendered step question
- `step_inputs`
  - copy of `inputs`, except `step_inputs["svg_path"] = current_svg`
  - `step_inputs["question"] = step_q`

It then calls `run_perception(step_inputs)` again for the step-specific SVG.

Step-level stored outputs:

- `perception_steps[]`
  - contains:
    - `index`
    - `operation`
    - `question`
    - `question_hint`
    - `operation_target`
    - `data_change`
    - sanitized `perception`
- `step_logs[]`
  - contains:
    - `index`
    - `operation`
    - `question`
    - `output_svg_path`
    - `output_image_path`

## 16. Step 1 Execution: Delete Line Series

The dispatch branch is:

- `chart_type == "line"`
- call `update_line_svg(current_svg, step_q, mapping_info, output_path, svg_output_path, llm)`

Step 1 concrete parameters:

- `svg_path`
  - `dataset/task2-line/del-add/001/001.svg`
- `question`
  - `Delete the category/series "Stellar Cultural Development Agency"`
- `mapping_info`
  - from step-level `run_perception()`
  - key fields used later:
    - `x_ticks`
    - `y_ticks`
    - `axes_bounds`
- `output_path`
  - `output/line/001_line_del-add_updated_step1.png`
- `svg_output_path`
  - `output/line/001_line_del-add_updated_step1.svg`
- `llm`
  - executor LLM

Inside `update_line_svg()`:

1. `_resolve_line_ops(question)`
   - result: `["delete"]`
2. `_remove_line_series(...)`

### 16.1 `_remove_line_series()` Parameters

- `svg_path`
- `question`
- `mapping_info`
- `output_path`
- `svg_output_path`
- `llm`

### 16.2 `_remove_line_series()` Internal Derived Parameters

After reading SVG content:

- `root`
- `axes`
- `legend`
- `legend_items`
  - each item includes:
    - `label`
    - `stroke`
    - `text`
    - `patch`
- `labels`
  - all legend labels
- `labels_to_remove`
  - resolved from the question
  - expected here:
    - `["Stellar Cultural Development Agency"]`
- `strokes_by_label`
  - mapping from legend label to line stroke color
- `target_stroke`
  - the stroke color for `Stellar Cultural Development Agency`
- `line_group`
  - the `<g id="line2d_*">` matching that stroke

Then it performs:

- remove the line group from `axes`
- remove the matching legend item
- call `_rescale_line_chart_after_removal(root, axes, content, mapping_info)`

### 16.3 `_rescale_line_chart_after_removal()` Parameters

- `root`
- `axes`
- `content`
  - original SVG text content before rescale
- `mapping_info`

Purpose:

- re-read visible line values after the deletion
- recompute y-axis tick layout
- update y-axis tick labels and series coordinates to match the new scale

Then it writes:

- `output/line/001_line_del-add_updated_step1.svg`
- `output/line/001_line_del-add_updated_step1.png`

## 17. Step 2 Execution: Add Line Series

Now `current_svg` becomes:

- `output/line/001_line_del-add_updated_step1.svg`

Step 2 concrete parameters:

- `svg_path`
  - `output/line/001_line_del-add_updated_step1.svg`
- `question`
  - `Add the category/series "Mariner Systems" with values [61656052.94, ... , 63498680.34]`
- `mapping_info`
  - refreshed by step-level `run_perception()` on the step 1 SVG
- `output_path`
  - `output/line/001_line_del-add_updated.png`
- `svg_output_path`
  - `output/line/001_line_del-add_updated.svg`
- `llm`
  - executor LLM

Inside `update_line_svg()`:

1. `_resolve_line_ops(question)`
   - result: `["add"]`
2. `_add_line_series(...)`

### 17.1 `_add_line_series()` Parameters

- `svg_path`
- `question`
- `mapping_info`
- `output_path`
- `svg_output_path`
- `llm`

### 17.2 `_add_line_series()` Internal Derived Parameters

From `mapping_info`:

- `x_ticks`
- `y_ticks`

From question parsing:

- `values`
  - parsed from the bracketed values list
- `llm_meta`
  - parser metadata if LLM parsing is used

Then:

- `x_positions = _compute_x_positions(values, x_ticks)`
- `y_positions = [_data_to_pixel(val, y_ticks) for val in values]`
- `points_svg = list(zip(x_positions, y_positions))`

From existing SVG style extraction:

- `line_style`
  - contains:
    - `stroke`
    - `stroke_width`
    - `stroke_linecap`
    - `has_markers`
- `stroke`
  - chosen as an unused line color
- `stroke_width`
- `stroke_linecap`
- `use_markers`

Legend parsing:

- `label = "Mariner Systems"`
- `legend`
- `legend_items`

Then it performs:

- create or reuse `line2d_update`
- write the polyline path into `line_path.d`
- draw markers if needed
- append a legend item for `Mariner Systems`

Then it writes:

- `output/line/001_line_del-add_updated.svg`
- `output/line/001_line_del-add_updated.png`

## 18. Attempt Stage G: Render Validation

After all steps finish, `run_main()` calls:

```text
_validate_render_with_programmatic(
  output_image=last_png,
  chart_type="line",
  update_spec=<from perception>,
  step_logs=<two step logs>,
  llm=executor_llm,
  svg_perception_mode=<optional>
)
```

Arguments:

- `output_image`
  - `output/line/001_line_del-add_updated.png`
- `chart_type`
  - `line`
- `update_spec`
  - the latest perception update spec
- `step_logs`
  - contains both the delete step and the add step outputs
- `llm`
- `svg_perception_mode`

For line charts:

- no area-specific programmatic validator is applied
- validation is mainly the generic render validator path

Typical output shape:

```json
{
  "ok": true,
  "confidence": 0.0-1.0,
  "issues": []
}
```

If validation fails:

- `retry_hint` is set from the issues
- the attempt loop runs again

## 19. Runtime Stage 5: Cluster Logic

For this line case:

- `cluster_result = None`
- `cluster_params = {}`

Cluster parameter extraction only matters for scatter cluster questions. It is irrelevant here.

## 20. Runtime Stage 6: Answer On Updated Chart

After a successful render, `run_main()` builds:

```json
{
  "question": "How many times do the lines for Aethelgard Cultural Council and Mariner Systems intersect?",
  "chart_type": "line",
  "output_image_path": "output/line/001_line_del-add_updated.png",
  "data_summary": {
    "update_spec": "...",
    "cluster_result": null,
    "cluster_params": {},
    "mapping_info_summary": {
      "num_points": "...",
      "num_areas": "...",
      "num_lines": "..."
    },
    "operation_plan": "...",
    "perception_steps": "...",
    "latest_step_logs": "..."
  }
}
```

Then it calls `answer_question()` with:

- `qa_question`
- `data_summary`
- `output_image_path = output/line/001_line_del-add_updated.png`
- `image_context_note = "The requested chart update has already been applied to this image. Answer the QA question only based on the updated chart."`
- `llm = answer_llm`

Outputs:

- `answer_initial`
- `answer`

## 21. Runtime Stage 7: Forced Visual Tool Phase

This case asks about line intersections, so `_should_force_visual_tool_phase(question, chart_type)` returns `True`.

Trigger condition:

- `chart_type == "line"`
- `qa_question` contains one of:
  - `intersection`
  - `intersections`
  - `cross`
  - `crossing`
  - Chinese variants like `交点`

So even if `answer_initial.confidence` is high, the tool phase is forced for this case.

## 22. Tool Phase Input Parameters

`run_visual_tool_phase()` receives:

- `question`
  - `How many times do the lines for Aethelgard Cultural Council and Mariner Systems intersect?`
- `chart_type`
  - `line`
- `data_summary`
  - same `answer_data_summary` used above
- `image_path`
  - `output/line/001_line_del-add_updated.png`
- `svg_path`
  - final step SVG from `step_logs`
  - here:
    `output/line/001_line_del-add_updated.svg`
- `llm`
  - `tool_planner_llm`
- `svg_perception_mode`
  - optional
- `max_tool_calls`
  - default `6`

## 23. Tool Planner Output For This Case

Because this is a line intersection question, the tool planner has a built-in fallback:

- `_default_line_intersection_tool_calls(chart_type="line", question=question)`

It extracts labels from the QA question:

- `line_A = "Aethelgard Cultural Council"`
- `line_B = "Mariner Systems"`

So the default or preferred tool call is:

```json
[
  {
    "tool": "zoom_and_highlight_intersection",
    "args": {
      "line_A": "Aethelgard Cultural Council",
      "line_B": "Mariner Systems"
    }
  }
]
```

## 24. Tool Execution Parameters

`_execute_svg_tool_calls(...)` receives:

- source SVG
  - `output/line/001_line_del-add_updated.svg`
- output SVG
  - `output/line/001_line_del-add_updated_tool_aug.svg`
- tool calls
  - usually the single intersection-highlighting call shown above
- `max_tool_calls = 6`

For `zoom_and_highlight_intersection`, the tool uses:

- `line_A`
- `line_B`

Internal behavior:

- locate both line series from the legend
- extract their polyline points
- compute all polyline segment intersections
- add highlighted marks at each intersection

Then `render_svg_to_png()` creates:

- `output/line/001_line_del-add_updated_tool_aug.png`

Tool phase output fields:

- `ok`
- `tool_calls`
- `planner`
- `augmented_svg_path`
- `augmented_image_path`

## 25. Final Answer On Augmented Chart

If tool execution succeeds, `_apply_tool_augmented_answer()` calls `answer_question()` again with:

- `qa_question`
- `data_summary`
- `output_image_path = output/line/001_line_del-add_updated_tool_aug.png`
- `image_context_note = "The requested chart update has already been applied, and visual augmentation has also been added to help reasoning. Answer the QA question only based on this updated and enhanced chart."`
- `llm = answer_llm`

Outputs:

- `answer_input_tool_augmented`
- `answer_tool_augmented`

Then:

- final `answer = answer_tool_augmented`

if the augmented answer exists.

## 26. Record Artifacts Written By The Batch Runner

After `run_main()` returns, `run_dataset_via_main.py` copies artifacts into the case record folder:

- `001_updated.svg`
  - copied from final SVG
- `001_updated.png`
  - copied from final PNG
- `001_tool_aug.svg`
  - copied if tool phase created augmented SVG
- `001_tool_aug.png`
  - copied if tool phase created augmented PNG
- `result.json`
  - full structured runtime result

It also writes answer summary text and match statistics.

## 27. Full Parameter Flow Summary

For `task2-line/del-add/001`, the concrete flow is:

1. Read dataset JSON and choose `QA[0]`
2. Build `structured_update_context` with:
   - `operation_target.category_name = ["Stellar Cultural Development Agency"]`
   - `operation_target.add_category = "Mariner Systems"`
   - `data_change.add.values = 15 numeric values`
3. Call `run_main()` with:
   - original question
   - source SVG path
   - source PNG path
   - structured update context
   - `max_render_retries = 2`
4. Split the full question into:
   - update part
   - intersection QA part
5. Answer the original chart once
6. Run perception on the source SVG
7. Build two steps:
   - delete `Stellar Cultural Development Agency`
   - add `Mariner Systems` with 15 values
8. Execute step 1:
   - remove the old line
   - remove its legend item
   - rescale the y-axis
   - write `..._step1.svg/png`
9. Execute step 2:
   - parse the 15 values
   - compute SVG polyline coordinates from `x_ticks` and `y_ticks`
   - add the new line and legend item
   - write final `...updated.svg/png`
10. Validate the final render
11. Answer the updated chart
12. Force a visual tool phase because the QA asks about line intersections
13. Run `zoom_and_highlight_intersection` with:
   - `line_A = Aethelgard Cultural Council`
   - `line_B = Mariner Systems`
14. Render augmented SVG/PNG
15. Answer again on the augmented chart
16. Store artifacts and compare the final answer with ground truth `14`
