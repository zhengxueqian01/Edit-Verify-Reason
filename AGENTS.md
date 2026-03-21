# AGENTS.md

## 项目定位

ChartAgent 是一个基于 SVG 图表的更新与问答项目。当前主流程是：

1. 拆分用户请求中的“更新指令”和“问答目标”
2. 感知当前图表结构
3. 规划更新步骤
4. 对 SVG 逐步执行修改
5. 渲染并校验结果
6. 基于更新后的图表生成答案

当前重点支持的图表类型：

- `scatter`
- `line`
- `area`

## 关键目录

- `src/main.py`：单条样本主入口
- `src/run_dataset_via_main.py`：批量跑数据集入口
- `src/web_app.py`：本地 Web 交互入口
- `src/chart_agent/core/`：主流程编排、状态、策略、回答阶段
- `src/chart_agent/perception/`：SVG 感知、更新、渲染、校验
- `src/chart_agent/prompts/`：提示词与结构化 schema
- `src/chart_agent/tools/`：工具注册与工具实现
- `src/web/index.html`：Web 页面
- `tests/`：pytest 测试
- `dataset/`、`dataset0313/`、`dataset03212/`：数据集与评测输入
- `output/`：运行输出目录

## 环境准备

推荐命令：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

SVG 转 PNG 依赖 `resvg`：

```bash
brew install resvg
```

如果缺少模型密钥，请通过环境变量或 `.env` 注入，不要把密钥写进代码或测试：

```env
Aihubmix_API_KEY=
Siliconflow_API_KEY=
Doubao_API_KEY=
```

## 常用命令

单例运行：

```bash
python -m src.main \
  --question "Delete one series and tell me which year has the maximum value" \
  --image dataset/task2-line/094/094.png \
  --svg dataset/task2-line/094/094.svg
```

批量运行：

```bash
python -m src.run_dataset_via_main \
  --input-dir dataset/task2-line \
  --qa-index 0 \
  --max-render-retries 2
```

启动 Web：

```bash
python -m src.web_app
```

运行测试：

```bash
pytest -q
```

改动局部逻辑时，优先跑最相关测试，例如：

```bash
pytest -q tests/test_svg_perceiver.py
pytest -q tests/test_scatter_svg_updater.py
pytest -q tests/test_main_question_split.py
```

## 开发约定

- 先搜索再修改，不要假设文件路径、配置名或内部 API 存在。
- 优先做小而清晰的改动；不要顺手大规模重构。
- 涉及行为变化时，优先补对应的 `pytest` 测试。
- 保持现有代码风格；不要混入无关的命名体系或抽象层。
- 不要新增遥测、埋点、外部网络请求，除非任务明确要求。
- 不要提交数据集产物、临时图片、调试日志或 `output/` 下的大文件。
- 如果改动依赖模型输出，尽量把可验证部分下沉到纯函数或可测试模块。

## 修改建议

### 改问题拆分或流程编排时

重点查看：

- `src/main.py`
- `src/chart_agent/core/graph.py`
- `src/chart_agent/core/state.py`
- `src/chart_agent/core/policy.py`

优先补：

- `tests/test_main_question_split.py`
- 与流程输出相关的现有测试

### 改 SVG 感知或图表编辑时

重点查看：

- `src/chart_agent/perception/svg_perceiver.py`
- `src/chart_agent/perception/*_svg_updater.py`
- `src/chart_agent/perception/svg_renderer.py`
- `src/chart_agent/perception/render_validator.py`

优先补：

- 对应图表类型的 updater 测试
- 渲染或校验相关测试

### 改回答或视觉工具增强时

重点查看：

- `src/chart_agent/core/answerer.py`
- `src/chart_agent/core/vision_tool_phase.py`
- `src/chart_agent/prompts/answer_prompt.py`

优先补：

- `tests/test_main_tool_phase_output.py`
- `tests/test_vision_tool_phase.py`
- 回答器相关测试

## 输出与调试

- 渲染输出通常写到 `output/<chart_type>/`
- 批量运行记录通常写到 `output/dataset_records/...`
- `src.main` 会输出 JSON，常用字段包括：
  - `output_image_path`
  - `render_check`
  - `attempt_logs`
  - `answer_initial`
  - `answer_tool_augmented`
  - `answer`

调试时优先保留最小复现输入：一个 `question`、一张 `png`、一个 `svg`。

## 提交前检查

至少完成以下检查：

1. 改动是否只覆盖当前任务范围，没有顺带改无关逻辑。
2. 新增命令、路径、字段名是否都能在仓库里找到依据。
3. 相关 `pytest` 是否已运行。
4. 若依赖 `resvg` 或外部模型，是否在说明中写清前置条件。
5. 是否避免提交 `output/`、缓存文件、密钥和本地配置。

## 额外说明

- 根目录已有一个 [agent.md](/Users/xueqianzheng/code/ChartAgent/agent.md)，它更偏系统流程说明。
- 本文件面向在这个仓库中协作开发的 agent / 工程助手，目的是减少误改、补齐运行约定、提高修改可验证性。
