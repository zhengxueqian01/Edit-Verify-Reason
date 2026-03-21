# Pipeline 完整流程说明

这份文档按你更正后的真实案例来写，并且把每一步里用到的参数都展开。

案例：

- `dataset/task2-line/del-add/001`

目标：

- 不是只讲“阶段”
- 而是把每个阶段的输入参数、派生参数、step 对象、updater 参数、输出路径都列清楚

## 1. 数据样例本体

这个 case 目录下的文件有：

- `dataset/task2-line/del-add/001/001.json`
- `dataset/task2-line/del-add/001/001.svg`
- `dataset/task2-line/del-add/001/001.png`
- `dataset/task2-line/del-add/001/001.csv`
- `dataset/task2-line/del-add/001/001_aug.svg`
- `dataset/task2-line/del-add/001/001_aug.png`

其中 `001.json` 的核心内容是：

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

这个样例表达的是：

- 删除一条现有折线：`Stellar Cultural Development Agency`
- 新增一条折线：`Mariner Systems`
- 然后回答：
  `Aethelgard Cultural Council` 和 `Mariner Systems` 这两条线相交多少次

标准答案：

- `14`

## 2. 批量入口命令

批量入口是：

- `python -m src.run_dataset_via_main`

如果只跑这个目录，典型命令是：

```bash
python -m src.run_dataset_via_main \
  --input-dir dataset/task2-line/del-add \
  --qa-index 0 \
  --max-render-retries 2
```

这些参数的含义：

- `--input-dir`
  - 类型：字符串
  - 含义：数据集任务目录
  - 当前值：`dataset/task2-line/del-add`
- `--qa-index`
  - 类型：整数
  - 含义：取 `QA[]` 中第几个问题
  - 当前值：`0`
- `--max-render-retries`
  - 类型：整数
  - 含义：第一次尝试失败后最多还能再重试几次
  - 当前值：`2`
- `--limit`
  - 类型：整数
  - 默认：`0`
  - 含义：最多跑多少个 case；`0` 表示不限制
- `--record-root`
  - 类型：字符串
  - 默认：`output/dataset_records`
- `--resume`
  - 类型：布尔 flag
  - 含义：如果已有 `result.json`，则跳过

## 3. 这个 case 在批量入口里被解析成什么参数

`src/run_dataset_via_main.py` 对 `001` 这个 case 会解析出：

- `case_id = "001"`
- `json_path = dataset/task2-line/del-add/001/001.json`
- `svg_path = dataset/task2-line/del-add/001/001.svg`
- `image_path = dataset/task2-line/del-add/001/001.png`
- `case_out_dir = <record_root>/<run_dir>/001`

然后取 `QA[0]`：

- `qa_question`
  - 值：
    `After deleting the category Stellar Cultural Development Agency and adding the category Mariner Systems, how many times do the lines for Aethelgard Cultural Council and Mariner Systems intersect?`
- `expected_answer`
  - 值：`14`
- `qa_item`
  - 值：`QA[0]` 对应的完整字典

## 4. Structured Update Context 现在长什么样

当前 `build_structured_update_context()` 只保留真正用于执行更新的结构化参数：

- `operation_target`
- `data_change`

对这个 case，构造出的对象是：

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

已经从这个 payload 中删除的字段：

- `chart_type`
- `operation`
- `cluster_params`
- `task`

## 5. 最终传给 `run_main()` 的完整输入

批量入口最后传给 `run_main()` 的参数是：

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

## 6. `run_main()` 第一阶段：模型初始化

`run_main()` 会构造五套模型配置和实例：

- `splitter`
  - 用途：把完整问题切成 update 部分和 QA 部分
- `planner`
  - 用途：生成编辑计划和 step 顺序
- `executor`
  - 用途：执行 SVG 更新时辅助解析 label/value
- `answer`
  - 用途：对原图、更新图、增强图回答问题
- `tool_planner`
  - 用途：决定是否调用视觉增强工具，以及调用什么工具

另外几个运行时参数：

- `svg_update_mode`
  - 来源：`inputs.get("svg_update_mode")`
  - 这个 batch case 没显式传
- `svg_perception_mode`
  - 来源：`inputs.get("svg_perception_mode")`
  - 这个 batch case 没显式传
- `model_overrides`
  - 来源：`inputs.get("model_overrides")`
  - 这里没传

## 7. 第二阶段：问题切分

`_resolve_questions()` 收到的输入核心是：

- `raw_question`
  - 就是完整自然语言问题
- `raw_update`
  - 空串
- `raw_qa`
  - 空串
- `structured_update_context`
  - 上面那份结构化执行参数

目标输出有四个：

- `update_question`
  - 只保留“怎么改图”
- `qa_question`
  - 只保留“最后问什么”
- `split_info`
  - 问题切分元信息
- `split_data_change`
  - 如果模型从问题里额外抽出了结构化数据，这里会带出来

对这个 case，理想的切分结果大概是：

- `update_question`
  - `Delete the category Stellar Cultural Development Agency. Add the category Mariner Systems.`
- `qa_question`
  - `How many times do the lines for Aethelgard Cultural Council and Mariner Systems intersect?`

注意：

- 即便模型切分不稳定，后面仍然会把 `operation_target` 和 `data_change` 再并回执行上下文
- 所以新增系列的 15 个值不会丢

## 8. 第三阶段：先在原图上回答一次

在真正修改图之前，`run_main()` 先对原图调用一次 `answer_question()`。

存入 `answer_original_input` 的参数是：

```json
{
  "question": "How many times do the lines for Aethelgard Cultural Council and Mariner Systems intersect?",
  "output_image_path": "dataset/task2-line/del-add/001/001.png",
  "data_summary": {}
}
```

调用参数：

- `qa_question`
- `data_summary = {}`
- `output_image_path = 原始 PNG`
- `image_context_note = "This is the original chart image before any requested update is applied."`
- `llm = answer_llm`

输出字段：

- `answer_original`

## 9. 第四阶段：进入 attempt 循环

这条主线会进入带重试的执行循环。

控制参数：

- `max_render_retries = 2`
- 总尝试次数 = `第一次 + 最多 2 次重试 = 3`

attempt 循环里维护的变量包括：

- `attempt`
- `planned_question`
- `retry_hint`
- `attempt_logs`
- `last_state`
- `last_output_image`
- `last_operation_plan`
- `last_render_check`
- `last_perception_steps`

初始化时：

- `planned_question = update_question`
- `retry_hint = ""`

## 10. attempt 内第 1 步：`run_perception()`

每次 attempt 开始时，`run_main()` 会准备一份 perception 输入：

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

然后 `run_perception(inputs)` 会顺序执行：

1. `DETECT_INPUT_MODE`
2. `PARSE_QUESTION`
3. `PERCEIVE_IMAGE_SVG`
4. `SANITY_CHECK`

### 10.1 `PARSE_QUESTION`

`parse_question(question, llm, chart_type_hint)` 会返回：

- `update_spec`
- `issues`

`update_spec` 常见字段：

- `new_points`
- `raw`
- 其他 parser 衍生字段

这个 `line del+add` case 里，`new_points` 不是关键；真正关键的新增序列数值还是来自 `data_change.add.values`。

### 10.2 `PERCEIVE_IMAGE_SVG`

`perceive_svg(svg_path, question, llm, perception_mode)` 的输出字段有：

- `chart_type`
- `chart_type_confidence`
- `perception_mode`
- `mapping_ok`
- `mapping_confidence`
- `primitives_summary`
- `mapping_info`
- `issues`
- `suggested_next_actions`

对于 line 图，`mapping_info` 里真正重要的是：

- `axes_bounds`
- `x_ticks`
  - 格式：`[(pixel_x, data_value), ...]`
- `y_ticks`
  - 格式：`[(pixel_y, data_value), ...]`
- `x_labels`

另外几个字段虽然也在 `mapping_info` 中，但对这个 case 不关键：

- `existing_points_svg`
- `existing_point_colors`
- `area_top_boundary`
- `area_fills`

`primitives_summary` 里有：

- `num_circles`
- `num_points`
- `num_xticks`
- `num_yticks`
- `num_areas`
- `num_lines`

然后 `_resolve_supported_chart_type()` 会决定最终图类型：

- 这个 case 的最终结果是：`line`

## 11. attempt 内第 2 步：LLM 规划更新

`_llm_plan_update(question, chart_type, llm, retry_hint)` 的输入是：

- `question = planned_question`
- `chart_type = "line"`
- `retry_hint = ""`

它要求 LLM 返回的 JSON 结构包括：

- `operation`
- `normalized_question`
- `steps`
- `new_points`
- `retry_hint`

其中 `steps[]` 里的每个 step 可以包含：

- `operation`
- `question_hint`
- `operation_target`
- `data_change`
- `new_points`

对这个 case，理想上的逻辑规划大致是：

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

实际 LLM 输出可以有波动，但下一阶段会用结构化数据做兜底补全。

## 12. attempt 内第 3 步：构造最终可执行 step

`_operation_steps_from_plan(operation_plan, planned_question, structured_context)` 会把前面的规划结果变成最终执行对象。

### 12.1 当前可用的结构化数据

这时手上的结构化参数是：

- `operation_target = {"category_name": ["Stellar Cultural Development Agency"], "add_category": "Mariner Systems"}`
- `data_change.add.mode = "full_series"`
- `data_change.add.years = ["1958", ..., "1972"]`
- `data_change.add.values = [61656052.94, ..., 63498680.34]`

### 12.2 这个 case 的最终 step 对象

Step 1：

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

Step 2：

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

## 13. attempt 内第 4 步：把 step 渲染成执行问题

在真正执行 step 之前，`_render_structured_step_question(step)` 会把结构化对象重新转成执行文本。

Step 1 的执行问题：

```text
Delete the category/series "Stellar Cultural Development Agency"
```

Step 2 的执行问题：

```text
Add the category/series "Mariner Systems" with values [61656052.94, 29088578.52, 60841542.31, 27550184.99, 59772165.2, 29469669.71, 64297153.54, 30732810.16, 64534250.46, 30543413.67, 61012706.65, 30967944.43, 64010036.85, 29310612.84, 63498680.34]
```

这两个字符串就是后面 `update_line_svg()` 收到的 `question` 参数。

## 14. attempt 内第 5 步：生成每个 step 的输出路径

`_step_paths(svg_path, chart_type, idx, total)` 会先调用 `default_output_paths()`。

对源 SVG `dataset/task2-line/del-add/001/001.svg`：

- `case_id = 001`
- `chart_type = line`
- 数据集 JSON 中的 `operation = del+add`
- 归一化后 operation = `del-add`

所以最终产物命名为：

- 最终 SVG：
  `output/line/001_line_del-add_updated.svg`
- 最终 PNG：
  `output/line/001_line_del-add_updated.png`

因为这个 case 有 2 个 step，所以路径分配是：

- Step 1 SVG：
  `output/line/001_line_del-add_updated_step1.svg`
- Step 1 PNG：
  `output/line/001_line_del-add_updated_step1.png`
- Step 2 SVG：
  `output/line/001_line_del-add_updated.svg`
- Step 2 PNG：
  `output/line/001_line_del-add_updated.png`

## 15. attempt 内第 6 步：step 执行包装层

`_execute_planned_steps()` 对每个 step 会做：

- 生成 `step_q`
  - 即刚才那条执行问题
- 生成 `step_inputs`
  - 是原始 `inputs` 的拷贝
  - 同时改写：
    - `step_inputs["svg_path"] = current_svg`
    - `step_inputs["question"] = step_q`

然后再对这个 step-specific SVG 跑一次 `run_perception(step_inputs)`。

每一步会记录两个结构：

- `perception_steps[]`
  - 包含：
    - `index`
    - `operation`
    - `question`
    - `question_hint`
    - `operation_target`
    - `data_change`
    - `perception`
- `step_logs[]`
  - 包含：
    - `index`
    - `operation`
    - `question`
    - `output_svg_path`
    - `output_image_path`

## 16. Step 1：删除折线

这一层的 dispatch 逻辑是：

- `chart_type == "line"`
- 调用：
  `update_line_svg(current_svg, step_q, mapping_info, output_path, svg_output_path, llm)`

Step 1 的具体参数：

- `svg_path`
  - `dataset/task2-line/del-add/001/001.svg`
- `question`
  - `Delete the category/series "Stellar Cultural Development Agency"`
- `mapping_info`
  - 来自 step 级别的 `run_perception()`
  - 这里后续真正会用到的是：
    - `x_ticks`
    - `y_ticks`
    - `axes_bounds`
- `output_path`
  - `output/line/001_line_del-add_updated_step1.png`
- `svg_output_path`
  - `output/line/001_line_del-add_updated_step1.svg`
- `llm`
  - executor 模型

在 `update_line_svg()` 内部：

1. `_resolve_line_ops(question)`
   - 结果：`["delete"]`
2. 然后走 `_remove_line_series(...)`

### 16.1 `_remove_line_series()` 的输入参数

- `svg_path`
- `question`
- `mapping_info`
- `output_path`
- `svg_output_path`
- `llm`

### 16.2 `_remove_line_series()` 内部推导出来的参数

SVG 读入之后会得到：

- `root`
- `axes`
- `legend`
- `legend_items`
  - 每个元素都带：
    - `label`
    - `stroke`
    - `text`
    - `patch`
- `labels`
  - 当前 legend 里的所有 label
- `labels_to_remove`
  - 从 question 里解析出来
  - 这里预期是：
    - `["Stellar Cultural Development Agency"]`
- `strokes_by_label`
  - `label -> stroke color`
- `target_stroke`
  - `Stellar Cultural Development Agency` 对应的颜色
- `line_group`
  - 在 `axes` 里找到的匹配那条线的 `<g id="line2d_*">`

然后它会做：

- 从 `axes` 里删除这条线
- 从 legend 里删掉对应 legend item
- 调用 `_rescale_line_chart_after_removal(root, axes, content, mapping_info)`

### 16.3 `_rescale_line_chart_after_removal()` 的参数

- `root`
- `axes`
- `content`
  - 当前 SVG 的文本内容
- `mapping_info`

作用：

- 重新读取删除后的可见 line 数据
- 重新计算 y 轴 ticks
- 重新更新 y 轴文本和每条折线的像素坐标

最后会写出：

- `output/line/001_line_del-add_updated_step1.svg`
- `output/line/001_line_del-add_updated_step1.png`

## 17. Step 2：新增折线

执行 Step 2 时，`current_svg` 已经变成：

- `output/line/001_line_del-add_updated_step1.svg`

Step 2 的具体参数：

- `svg_path`
  - `output/line/001_line_del-add_updated_step1.svg`
- `question`
  - `Add the category/series "Mariner Systems" with values [61656052.94, ... , 63498680.34]`
- `mapping_info`
  - 这是在 Step 1 结果图上重新 perception 得到的新映射信息
- `output_path`
  - `output/line/001_line_del-add_updated.png`
- `svg_output_path`
  - `output/line/001_line_del-add_updated.svg`
- `llm`
  - executor 模型

`update_line_svg()` 内部会：

1. `_resolve_line_ops(question)`
   - 结果：`["add"]`
2. 然后走 `_add_line_series(...)`

### 17.1 `_add_line_series()` 的输入参数

- `svg_path`
- `question`
- `mapping_info`
- `output_path`
- `svg_output_path`
- `llm`

### 17.2 `_add_line_series()` 内部推导出来的参数

从 `mapping_info` 中读：

- `x_ticks`
- `y_ticks`

从 question 中解析：

- `values`
  - 从方括号数值串里读出来
- `llm_meta`
  - 如果 regex 不够，需要 LLM 辅助时会有这个元信息

然后继续计算：

- `x_positions = _compute_x_positions(values, x_ticks)`
- `y_positions = [_data_to_pixel(val, y_ticks) for val in values]`
- `points_svg = list(zip(x_positions, y_positions))`

再从现有 SVG 风格中提取：

- `line_style`
  - 包含：
    - `stroke`
    - `stroke_width`
    - `stroke_linecap`
    - `has_markers`
- `stroke`
  - 自动选一个当前未被使用的线色
- `stroke_width`
- `stroke_linecap`
- `use_markers`

legend 相关参数：

- `label = "Mariner Systems"`
- `legend`
- `legend_items`

然后它会做：

- 创建或复用 `line2d_update`
- 把新增折线 polyline 写进 `line_path.d`
- 如果需要 marker，就画 marker
- 把 `Mariner Systems` 追加进 legend

最后写出：

- `output/line/001_line_del-add_updated.svg`
- `output/line/001_line_del-add_updated.png`

## 18. attempt 内第 7 步：渲染校验

两个 step 执行完后，`run_main()` 会调用：

```text
_validate_render_with_programmatic(
  output_image=最后的 PNG,
  chart_type="line",
  update_spec=<最新 perception 里的 update_spec>,
  step_logs=<两条 step log>,
  llm=executor_llm,
  svg_perception_mode=<可选>
)
```

具体参数：

- `output_image`
  - `output/line/001_line_del-add_updated.png`
- `chart_type`
  - `line`
- `update_spec`
  - 最新 perception 里解析出的更新 spec
- `step_logs`
  - 包含 Step 1 和 Step 2 的输出路径
- `llm`
- `svg_perception_mode`

对于 line 图：

- 不会走 area 专用的程序化 validator
- 主要走通用渲染校验

典型输出格式：

```json
{
  "ok": true,
  "confidence": 0.0,
  "issues": []
}
```

如果校验失败：

- `retry_hint` 会从 `issues` 里生成
- 然后重新进入下一次 attempt

## 19. 第五阶段：cluster 逻辑

这个 case 是 line 图，所以：

- `cluster_result = None`
- `cluster_params = {}`

cluster 参数逻辑只对 scatter cluster 问题有意义，这里不参与。

## 20. 第六阶段：对更新后的图回答 QA

渲染通过后，`run_main()` 会构造：

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

然后调用 `answer_question()`：

- `qa_question`
- `data_summary`
- `output_image_path = output/line/001_line_del-add_updated.png`
- `image_context_note = "The requested chart update has already been applied to this image. Answer the QA question only based on the updated chart."`
- `llm = answer_llm`

输出：

- `answer_initial`
- `answer`

## 21. 第七阶段：为什么这个 case 会强制进入工具增强

这个 case 问的是 line intersection，所以 `_should_force_visual_tool_phase(question, chart_type)` 会返回 `True`。

触发条件：

- `chart_type == "line"`
- `qa_question` 中包含以下关键词之一：
  - `intersection`
  - `intersections`
  - `cross`
  - `crossing`
  - 中文如 `交点`

所以这类 case 即使 `answer_initial.confidence` 不低，也会强制进入工具增强阶段。

## 22. 工具增强阶段的输入参数

`run_visual_tool_phase()` 收到的参数是：

- `question`
  - `How many times do the lines for Aethelgard Cultural Council and Mariner Systems intersect?`
- `chart_type`
  - `line`
- `data_summary`
  - 与前面 `answer_data_summary` 相同
- `image_path`
  - `output/line/001_line_del-add_updated.png`
- `svg_path`
  - 从 `step_logs` 里拿到的最终 step SVG
  - 即：
    `output/line/001_line_del-add_updated.svg`
- `llm`
  - `tool_planner_llm`
- `svg_perception_mode`
  - 可选
- `max_tool_calls`
  - 默认 `6`

## 23. 这个 case 的工具规划结果会是什么

因为这是 line intersection 问题，工具规划器有内置 fallback：

- `_default_line_intersection_tool_calls(chart_type="line", question=question)`

它会从 QA 问句中抽出两条线的 label：

- `line_A = "Aethelgard Cultural Council"`
- `line_B = "Mariner Systems"`

因此默认或优先的工具调用是：

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

## 24. 工具执行阶段的参数

`_execute_svg_tool_calls(...)` 收到：

- source SVG
  - `output/line/001_line_del-add_updated.svg`
- output SVG
  - `output/line/001_line_del-add_updated_tool_aug.svg`
- tool calls
  - 通常就是上面那个 intersection tool call
- `max_tool_calls = 6`

对 `zoom_and_highlight_intersection` 而言，真正有意义的参数是：

- `line_A`
- `line_B`

它内部会：

- 从 legend 里找到这两条线
- 取出两条 polyline 的点
- 计算所有线段交点
- 在交点处加高亮标记

然后再调用 `render_svg_to_png()` 渲染出：

- `output/line/001_line_del-add_updated_tool_aug.png`

工具阶段的输出字段：

- `ok`
- `tool_calls`
- `planner`
- `augmented_svg_path`
- `augmented_image_path`

## 25. 最后一轮：在增强图上再答一次

如果工具执行成功，`_apply_tool_augmented_answer()` 会再次调用 `answer_question()`：

- `qa_question`
- `data_summary`
- `output_image_path = output/line/001_line_del-add_updated_tool_aug.png`
- `image_context_note = "The requested chart update has already been applied, and visual augmentation has also been added to help reasoning. Answer the QA question only based on this updated and enhanced chart."`
- `llm = answer_llm`

对应输出：

- `answer_input_tool_augmented`
- `answer_tool_augmented`

然后：

- 如果增强回答存在，最终 `answer = answer_tool_augmented`

## 26. 批量入口最终会写哪些文件

`run_dataset_via_main.py` 在 `run_main()` 返回后，会把产物拷到 record 目录：

- `001_updated.svg`
  - 从最终 SVG 拷贝而来
- `001_updated.png`
  - 从最终 PNG 拷贝而来
- `001_tool_aug.svg`
  - 如果工具增强阶段生成了 SVG
- `001_tool_aug.png`
  - 如果工具增强阶段生成了 PNG
- `result.json`
  - 完整运行结果

同时还会写答案摘要和匹配统计。

## 27. 这个案例的完整参数流总结

`task2-line/del-add/001` 的完整参数流就是：

1. 读 `001.json`
2. 选中 `QA[0]`
3. 构造 `structured_update_context`：
   - `operation_target.category_name = ["Stellar Cultural Development Agency"]`
   - `operation_target.add_category = "Mariner Systems"`
   - `data_change.add.values = 15 个数`
4. 调用 `run_main()`
5. 把原问题切成：
   - update 部分
   - 交点 QA 部分
6. 先对原图回答一次
7. 跑 perception
8. 构造两个 step：
   - 删除 `Stellar Cultural Development Agency`
   - 用 15 个值新增 `Mariner Systems`
9. 执行 Step 1：
   - 删除原线
   - 删除 legend
   - 重新缩放 y 轴
   - 产出 `..._step1.svg/png`
10. 执行 Step 2：
   - 解析 15 个值
   - 结合 `x_ticks`、`y_ticks` 计算新折线的 SVG 坐标
   - 把新线和 legend 写入 SVG
   - 产出最终 `...updated.svg/png`
11. 做渲染校验
12. 在更新图上先回答一次
13. 因为问题是 intersection，所以强制进入工具增强
14. 调用 `zoom_and_highlight_intersection`
   - `line_A = Aethelgard Cultural Council`
   - `line_B = Mariner Systems`
15. 渲染增强版 SVG/PNG
16. 在增强图上再回答一次
17. 最后把结果和标准答案 `14` 做对比，并把所有产物写入 record 目录
