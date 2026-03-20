# Agent Flow

## 1. Overview

ChartAgent takes a chart input and one unified user input, updates the chart when needed, and answers the QA question on top of the updated chart.

The current end-to-end flow is:

1. Receive unified input text and chart assets.
2. Decompose the input into `operation` and `question`.
3. Perceive the chart structure from SVG.
4. Build executable update steps.
5. Apply chart updates step by step.
6. Validate the rendered result.
7. Answer the QA question on the updated chart.
8. Optionally run visual tool augmentation for hard questions.

## 2. Configurable Modes

Two env switches control the current behavior:

- `SVG_PERCEPTION_MODE`
- `SVG_UPDATE_MODE`

Default values:

```text
SVG_PERCEPTION_MODE=rules
SVG_UPDATE_MODE=rules
```

### 2.1 `SVG_PERCEPTION_MODE`

- `rules`: keep the existing primitive-based SVG perception path
- `llm`: keep lightweight SVG extraction, then feed a compact `SVG Summary` to the model for chart perception

At the current stage, both modes mainly affect chart perception, especially chart-type recognition.

### 2.2 `SVG_UPDATE_MODE`

- `rules`: use the current hybrid update flow
- `llm`: after the normal operation planning stage, let the model produce a more explicit edit intent, then pass that intent to the existing rule executor

This means the new update mode is still:

- model decides edit intent
- rules execute the actual SVG modification

The model does not directly rewrite SVG XML.

## 3. Inputs

### 3.1 Runtime Input

The agent conceptually receives:

- `input`: one unified input string
- `svg_path`: the chart SVG to be updated and reasoned over

Optionally:

- `image_path`: the original chart image, used for original-image QA evaluation only

The core rule is that the system treats the user request as one unified input, not as several independent user fields.

### 3.2 Evaluation Special Case

In evaluation, the single input may contain a structured tail copied from JSON, for example:

```text
After adding the category Regional Carriers and deleting the category Charter Flights, in which year does the overall maximum occur?
"operation_target": {
  "add_category": "Regional Carriers",
  "del_category": "Charter Flights"
},
"data_change": {
  "add": {
    "mode": "full_series",
    "years": ["2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"],
    "values": [128598, 186977, 205514, 136129, 226783, 246727, 170089, 154587, 195958, 176685]
  },
  "del": {
    "category": "Charter Flights"
  }
}
```

This is still treated as one unified input. The structured tail is only used as stronger grounding for the internal operation representation.

## 4. Question Decomposition

### 4.1 Goal

Question decomposition converts one unified input into two primary semantic parts:

- `operation`
- `question`

At the current implementation stage, the system may also attach internal structured fields such as:

- `operation_target`
- `data_change`

These structured fields are internal grounding data. They are not treated as independent user input channels.

### 4.2 Unified Principle

Both actual runtime input and evaluation input should converge to the same internal representation:

```json
{
  "operation": "...",
  "question": "...",
  "operation_target": {},
  "data_change": {}
}
```

The primary normalized result is always:

- one `operation`
- one `question`

Any structured information is only used to strengthen the internal representation of `operation`.

Here, `question` means the pure QA part only. It excludes the update operation itself and is intended for post-update answering.

### 4.3 Example A: Evaluation Input

Raw input:

```text
After adding the category Regional Carriers and deleting the category Charter Flights, in which year does the overall maximum occur?
"operation_target": {
  "add_category": "Regional Carriers",
  "del_category": "Charter Flights"
},
"data_change": {
  "add": {
    "mode": "full_series",
    "years": ["2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"],
    "values": [128598, 186977, 205514, 136129, 226783, 246727, 170089, 154587, 195958, 176685]
  },
  "del": {
    "category": "Charter Flights"
  }
}
```

Normalized internal representation:

```json
{
  "question": "In which year does the overall maximum occur?",
  "operation": "Add the category Regional Carriers and delete the category Charter Flights.",
  "operation_target": {
    "add_category": "Regional Carriers",
    "del_category": "Charter Flights"
  },
  "data_change": {
    "add": {
      "mode": "full_series",
      "years": ["2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"],
      "values": [128598, 186977, 205514, 136129, 226783, 246727, 170089, 154587, 195958, 176685]
    },
    "del": {
      "category": "Charter Flights"
    }
  }
}
```

### 4.4 Example B: Actual Runtime Input

Raw input:

```text
After adding a new category "Regional Carriers" with values across the years 2015-2024 (128,598; 186,977; 205,514; 136,129; 226,783; 246,727; 170,089; 154,587; 195,958; 176,685) and deleting the category "Charter Flights," in which year does the overall maximum occur?
```

Normalized internal representation:

```json
{
  "question": "In which year does the overall maximum occur?",
  "operation": "Add a new category Regional Carriers with values across the years 2015-2024 and delete the category Charter Flights.",
  "operation_target": {
    "add_category": "Regional Carriers",
    "del_category": "Charter Flights"
  },
  "data_change": {
    "add": {
      "mode": "full_series",
      "years": ["2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"],
      "values": [128598, 186977, 205514, 136129, 226783, 246727, 170089, 154587, 195958, 176685]
    }
  }
}
```

### 4.5 Execution Boundary

After decomposition:

- the answering stage should only consume the normalized `question`
- the update execution stage should consume the normalized `operation`
- internal structured fields may strengthen execution, but they do not replace `operation`

## 5. SVG Perception

SVG perception is still based on lightweight programmatic extraction. The two perception modes differ in what is sent to the model afterward.

### 5.1 `rules` Perception Mode

In `rules` mode, the model mainly sees a coarse primitive summary, for example:

```json
{
  "num_points": 0,
  "num_xticks": 14,
  "num_yticks": 5,
  "num_areas": 0,
  "num_lines": 14
}
```

Its main current role is chart-type recognition.

### 5.2 `llm` Perception Mode

In `llm` mode, the model sees a richer compact `SVG Summary`, for example:

```json
{
  "chart_type_guess": "unknown",
  "primitives_summary": {
    "num_points": 0,
    "num_xticks": 14,
    "num_yticks": 5,
    "num_areas": 0,
    "num_lines": 14
  },
  "axes_bounds": {
    "x_min": 41.880625,
    "x_max": 430.680625,
    "y_min": 37.249375,
    "y_max": 314.449375
  },
  "x_tick_values": [1968.0, 1969.0, 1970.0, "...", 1981.0],
  "y_tick_values": [5000000.0, 10000000.0, 15000000.0, 20000000.0, 25000000.0]
}
```

At the current stage, its main role is still chart-type perception.

## 6. SVG Update

### 6.1 Current Default Path

The current default update path is a hybrid flow:

1. LLM decomposes the user input.
2. LLM proposes operation steps.
3. Rule logic may enrich or repair incomplete targets and data.
4. Rule updaters execute the actual SVG modification.

So the current system is not a pure workflow, but also not a fully model-driven agent. It is a hybrid agent.

### 6.2 `llm` Update Path

The `llm` update path keeps all existing rule executors unchanged, but inserts an additional model-driven intent stage before execution:

```text
operation_text
-> normal operation planning
-> llm intent planning
-> structured steps
-> existing SVG updater execution
```

In this path:

- the model is encouraged to decide which object, series, or category should be modified
- the output is still structured step data
- the bottom rule executor still performs the actual SVG deletion, addition, or update

The chart-type-specific updaters can therefore be viewed as tool-like executors:

- `line` updater
- `area` updater
- `scatter` updater

In that abstraction:

- the model decides which tool should be used
- the model provides the target object and parameters
- the bottom rule executor performs the actual SVG mutation

因此，按图表类型划分的 updater 可以被视为一层“工具式执行器”：

- `line` updater
- `area` updater
- `scatter` updater

在这种抽象下：

- 模型决定应该调用哪个工具
- 模型提供目标对象和参数
- 底层规则执行器负责真正的 SVG 修改

So the new path is:

- model decides edit intent
- rules execute the edit

## 7. Validation

After execution, the agent validates the updated rendering.

Validation can include:

- basic render checks
- chart-type-specific programmatic checks
- retry when render output is invalid

The purpose is to ensure that downstream QA is performed on a trustworthy updated chart.

## 8. Answering

Once the updated chart is available, the agent answers the QA question against the updated chart, not the original one.

In evaluation, the system may also answer the same question on the original image as a baseline measurement.

The textual inputs for answering should be distinguished as follows:

1. Original-image answering uses `full_input`, because the chart has not been updated yet.
2. Updated-image answering uses `question`, because the chart update has already been applied.
3. Tool-augmented-image answering also uses `question`, because visual augmentation is applied after the chart update.

In short:

- Original image answering: `full_input` + `original_image_path`
- Updated image answering: `question` + `updated_image_path`
- Tool-augmented image answering: `question` + `augmented_image_path`

## 9. Visual Tool Augmentation

For difficult QA tasks such as intersection counting or cluster reasoning, the agent may run a visual tool augmentation phase.

This phase adds lightweight visual guides to the already updated chart in order to improve answer reliability, without changing the chart data itself.
