# Chart Agent（中文说明）

本项目实现的是**Agent 形式**的图表系统（不是固定的工作流），核心特征：
- 由策略选择感知/更新动作
- 执行动作并记录 trace
- 自检与置信度评估
- 失败可重试/降级
- 输出下一步建议

## Agent 全流程（概念版）

**阶段 1：感知（闭环）**
- 策略选择感知动作（policy-driven）
- 执行动作：输入判断、问题解析、SVG 结构解析、图表类型判断（LLM 优先）
- 自检与问题收集（issues）
- 失败重试或降级
- 给出下一步建议

**阶段 2：更新与渲染**
- 根据解析结果对图表数据进行更新
- 渲染输出 PNG
- 输出按图表类型分类：`output/<chart_type>/`

**阶段 3：渲染后检查（建议补充）**
- 检查输出文件是否存在、尺寸是否合理
- 检测是否真正绘制了新增内容
- 异常时重试或降级

**阶段 4：回答/推理（后续阶段）**
- 基于更新后的图像/数据回答问题

---

## 当前已实现能力

### 感知（LLM 优先 + 规则兜底）
- 输入类型识别（图像+SVG vs 纯文本）
- 问题解析（LLM 优先）
- SVG 解析：散点/柱状/面积
- 图表类型判断：LLM 优先
- 自检（sanity check）与建议

### 更新与渲染
- **散点图**：按坐标增加新点
- **柱状图**：指定类别数值增加/变更
- **面积图**：新增一层堆叠面积
- **折线图**：新增一条线
- **纯文本绘图**：
  - 散点图（matplotlib）
  - 柱状图（matplotlib）
  - 折线图（matplotlib，多系列）

---

## CLI 使用示例

纯文本散点图：
```bash
python -m src.main --question "draw scatter" --text_spec "points: (1,2) (3,4)"
```

图像 + SVG（散点更新）：
```bash
python -m src.main --question "new points: (5,6)" --image dataset/task1/000.png --svg dataset/task1/000.svg
```

纯文本柱状图：
```bash
python -m src.main --question "draw bar" --text_spec "Peanut:120 Tomato:300"
```

图像 + SVG（柱状更新）：
```bash
python -m src.main --question "Peanut 增加 2200" --image dataset/task4/000.png --svg dataset/task4/000.svg
```

图像 + SVG（面积更新）：
```bash
python -m src.main --question "values: [10,12,8,9,11,10]" --image dataset/task5/000.png --svg dataset/task5/000.svg
```

纯文本折线图（多系列）：
```bash
python -m src.main --question "draw line" --text_spec "values: [{\"year\": 1951, \"A\": 1, \"B\": 2}]"
```

---

## 环境变量

```
OPENAI_API_KEY=
GPT_MODEL=
GPT_BASE_URL=
PERCEPTION_MAX_RETRIES=2
```

---

## 输出
- 所有生成图片输出到：`output/<chart_type>/`
- CLI 输出 JSON 包含：
  - `perception`（感知结果）
  - `trace`（Agent 轨迹）
  - `output_image_path`
