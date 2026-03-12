# ChartAgent（中文说明）

一个基于 Agent 思路的 SVG 图表更新与问答项目。

## 当前支持范围

- 主入口：`python -m src.main`
- 输入模式：基于 SVG 的更新（当前实现要求传 `--svg`）
- 支持的 SVG 更新图表类型：`scatter`、`line`、`area`
- 执行链路：问题切分 -> 感知 -> 更新步骤规划 -> SVG/PNG 渲染 -> 渲染校验 -> QA 生成
- 批量运行入口：`python -m src.run_dataset_via_main`
- 可选 Web 服务：`python -m src.web_app`

## 环境准备

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

安装 `resvg`（SVG 渲染 PNG 必需）：

```bash
brew install resvg
```

## 环境变量

在 `.env`（或系统环境变量）中配置 API Key：

```env
Aihubmix_API_KEY_ZZT=
Siliconflow_API_KEY=
Doubao_API_KEY=
```

可选模型/运行参数：

```env
GPT_MODEL=gpt-5.2
GPT_BASE_URL=https://aihubmix.com/v1
PERCEPTION_MAX_RETRIES=2
SPLITTER_MODEL=gpt
PLANNER_MODEL=gpt
EXECUTOR_MODEL=gpt
ANSWER_MODEL=gpt
```

## 单例运行

折线图示例：

```bash
python -m src.main \
  --question "Delete one series and tell me which year has the maximum value" \
  --image dataset/task2-line/094/094.png \
  --svg dataset/task2-line/094/094.svg
```

散点图示例：

```bash
python -m src.main \
  --question "Add points (10, 20) and (15, 18), then how many clusters are there?" \
  --image dataset/task3-scatter-cluster/094/094.png \
  --svg dataset/task3-scatter-cluster/094/094.svg
```

## 批量跑数据集

```bash
python -m src.run_dataset_via_main \
  --input-dir dataset/task2-line \
  --qa-index 0 \
  --max-render-retries 2
```

运行记录输出到：`output/dataset_records/<model>_<task>_<timestamp>/`。

## 独立验证 SVG 匹配

单独执行生成 SVG 和 ground truth 的匹配验证：

```bash
python -m src.validate_svg_matches \
  --pred-root output/dataset_records/dataset_task2-line_20260303_201506 \
  --dataset-dir dataset/task2-line
```

说明：

- `--pred-root` 和 `--dataset-dir` 必须显式传入。
- 默认输出路径为 `output/svg_match/<pred-root>.json`。
- 如果只验证单个 SVG 对，可以执行：

```bash
python -m src.validate_svg_matches \
  --pred-svg output/example.svg \
  --gt-svg dataset/task2-line/000/000_del.svg
```

## 启动 Web 页面

```bash
python -m src.web_app
```

浏览器访问：`http://127.0.0.1:8008`。

## 架构解耦：Web 交互与批量测试流

ChartAgent 区分了纯粹的后台执行核心和多轮前端交互层，互不干扰：

### 1. 批量测试 / 无干预管线 (`src.main`)
`src.main` 作为最纯粹的核心执行引擎，负责“问题切分 -> 图片渲染 -> QA 问答”的标准闭环。
- 当跑自动化测试或批量执行 (`src.run_dataset_via_main`) 时，系统使用代码即时生成的最终 SVG 截图进行最后的 QA 回答。
- 系统会先生成一次初始回答 `answer_initial`，并读取其中的 `confidence`。
- 只有当 `answer_initial.confidence` 较低时，才会进入“模型自主选择工具增强图片”阶段，再次生成 `answer_tool_augmented`。
- `answer_initial.confidence` 现已统一规范到 `0-1` 区间。
- 当前自动视觉增强工具为 `add_point`、`draw_line`、`highlight_rect`、`isolate_color_topology`、`isolate_all_color_topologies`、`draw_global_peak_crosshairs`、`zoom_and_highlight_intersection`。
- `add_text` 已移除。
- `isolate_color_topology` 面向散点图单色问题：会虚化非目标颜色点，并对目标颜色点做 DBSCAN 聚类与凸包圈选。
- `isolate_all_color_topologies` 面向散点图全局簇问题：会按颜色分别聚类，并为图中所有颜色的簇绘制凸包圈选。
- `draw_global_peak_crosshairs` 面向面积图：零参数执行，扫描全局 area 顶点，锁定最高点并绘制水平/垂直十字锚线。
- `zoom_and_highlight_intersection` 面向折线图：根据 legend 中的 `line_A` / `line_B` 定位两条折线，计算线段交点并注入红色交点标记。
- 无任何前端侵入式逻辑。

### 2. 交互式 Web 管线 (`src.web_app` + `src/web/index.html`)
Web 端支持“人机协同修改”体验：
- 允许用户在模型返回的第一轮输出图片上**直接用画笔、红框进行标记**。
- `web_app.py` 负责在网络层拦截并处理这些“带有手绘批注”的增强图片 (`enhanced_image`)。
- 当用户提交纯视觉修改（即：输入框为空，但画板上有标记）时，Web 层会自动继承上一轮文字问题，并将批注过的原图强制注入给大模型做 QA 重新回答。

## 输出说明

- 渲染产物：`output/<chart_type>/`
- 批量运行记录：`output/dataset_records/...`
- `src.main` 的 JSON 输出重点字段：
  - `output_image_path`
  - `render_check`
  - `attempt_logs`
  - `answer_initial`：初始 QA 结果
  - `answer_tool_augmented`：仅在初始置信度较低且触发工具增强后才会生成
  - `answer`：最终采用的回答；若触发并完成工具增强，则通常为增强后的结果，否则为初始回答

## 待办

- 验证生成的 SVG 图和 ground truth 的匹配度
- 优化现在的数据集
- 数据集在 4 个模型上的效果
- 优化验证脚本
