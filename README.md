# Chart Agent

This project implements an **agent-style** chart system (not a fixed workflow). The system behaves like an agent by:
- selecting perception/update actions based on policy,
- executing tool-like steps,
- performing self-checks with confidence and issues,
- retrying or degrading when needed,
- and suggesting next actions for downstream stages.

## Agent Flow (End-to-End)

**Phase 1: Perception (Agent Loop)**
- **Policy-driven action selection** based on current state and inputs.
- **Action execution**: parse question, detect input mode, read SVG primitives, classify chart type (LLM-first), etc.
- **Sanity checks**: validate mapping, confidence, required inputs, and structural consistency.
- **Retry / degrade**: retry with relaxed rules or fall back to alternate parsing when issues are detected.
- **Suggested next actions** are emitted for the next stage.

**Phase 2: Update & Render (Tool Actions)**
- Update chart data based on parsed intent and chart type.
- Render the updated chart to PNG.
- Output images are grouped by chart type under `output/<chart_type>/`.

**Phase 3: Post-Render Check (Agent-style)**
- Verify output image exists, size is valid, and pixels are non-empty.
- Confirm that updates are visually present (e.g., new points/areas/lines).
- If failed: retry render or fall back to alternate strategies.

**Phase 4: Answering / Reasoning (Future)**
- Use the updated chart/data to answer user questions.
- Provide reasoning and confidence.

---

## Current Implemented Capabilities

### Perception (LLM-first where applicable)
- Input mode detection (image+svg vs text-only)
- Question parsing with LLM + rules fallback
- SVG parsing for scatter, bar, and area
- LLM-first chart type classification
- Sanity checks with retry/degrade suggestions

### Update & Render
- **Scatter**: add points on image using SVG mapping
- **Bar**: add value to a category bar, color matched, width preserved
- **Area**: add new stacked area layer on top
- **Line**: add new line on top of existing chart
- **Text-only render**:
  - scatter (matplotlib)
  - bar (matplotlib)
  - line (matplotlib, multi-series supported)

---

## CLI Usage

Text-only (scatter):
```bash
python -m src.main --question "draw scatter" --text_spec "points: (1,2) (3,4)"
```

Image + SVG (scatter update):
```bash
python -m src.main --question "new points: (5,6)" --image dataset/task1/000.png --svg dataset/task1/000.svg
```

Text-only (bar):
```bash
python -m src.main --question "draw bar" --text_spec "Peanut:120 Tomato:300"
```

Image + SVG (bar update):
```bash
python -m src.main --question "Peanut 增加 2200" --image dataset/task4/000.png --svg dataset/task4/000.svg
```

Image + SVG (area update):
```bash
python -m src.main --question "values: [10,12,8,9,11,10]" --image dataset/task5/000.png --svg dataset/task5/000.svg
```

Text-only (line, multi-series):
```bash
python -m src.main --question "draw line" --text_spec "values: [{\"year\": 1951, \"A\": 1, \"B\": 2}]"
```

---

## Environment

```
OPENAI_API_KEY=
GPT_MODEL=
GPT_BASE_URL=
PERCEPTION_MAX_RETRIES=2
```

---

## Output

- All generated images are saved under `output/<chart_type>/`.
- The CLI prints JSON with:
  - `perception` state
  - `trace` (agent loop steps)
  - `output_image_path`
