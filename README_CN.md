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

## 输出说明

- 渲染产物：`output/<chart_type>/`
- 批量运行记录：`output/dataset_records/...`
- `src.main` 的 JSON 输出重点字段：
  - `output_image_path`
  - `render_check`
  - `attempt_logs`
  - `answer`、`answer_initial`、`answer_tool_augmented`

## 待办

- 优化现在模型自主调用工具的流程
- 验证生成的 SVG 图和 ground truth 的匹配度
- 优化现在的数据集
- 数据集在 4 个模型上的效果
- 优化验证脚本
