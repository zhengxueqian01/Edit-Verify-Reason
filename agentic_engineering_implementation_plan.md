# ChartAgent Agentic Engineering 实施方案

本文档面向当前仓库，目标是把论文《Edit, Render, Reason: SVG-Grounded Agent for Chart Modification Question Answering》的方法体系工程化，并进一步展开成适合自动复现、扩展实验、失败归因和自动报告的 Agentic Engineering 工作流。

## 1. 总体目标

整个实施分三层推进：

1. 把论文方法落成可稳定执行的工程流水线
2. 把复现、评测、归因、汇报等高重复工作交给 agent 自动完成
3. 在主实验稳定后，快速展开额外实验、消融实验与失败分析

建议的总路线：

`Paper Method -> Repro Pipeline -> Agentic Workflow -> Extended Experiments -> Auto Report`

对应到本项目，建议拆成四个阶段：

1. `M1 基线复现稳定化`
2. `M2 实验流水线标准化`
3. `M3 Agentic 自动化接管`
4. `M4 扩展实验与论文级分析`

原则是先打通最小可复现闭环，再逐步代理化，不先做复杂的多 agent 平台。

## 2. 论文方法到当前仓库的工程映射

按论文主流程，可以把系统拆成 6 个可观测阶段：

### 2.1 Question Split

重点路径：

- `src/main.py`
- `src/chart_agent/perception/question_parser.py`
- `tests/test_main_question_split.py`

目标：

- 从用户请求中拆出编辑指令和问答目标
- 输出结构化结果，供后续 perception 和 answering 使用

### 2.2 Chart Perception

重点路径：

- `src/chart_agent/perception/svg_perceiver.py`
- `tests/test_svg_perceiver.py`

目标：

- 感知 SVG 的图表结构
- 产出图元、坐标、系列、标签等可供编辑的结构化表示

### 2.3 Edit Planning / Policy

重点路径：

- `src/chart_agent/core/policy.py`
- `src/chart_agent/core/graph.py`
- `src/chart_agent/core/state.py`

目标：

- 根据编辑目标规划执行步骤
- 管理中间状态、策略分支和重试决策

### 2.4 SVG Update Execution

重点路径：

- `src/chart_agent/perception/scatter_svg_updater.py`
- `src/chart_agent/perception/line_svg_updater.py`
- `src/chart_agent/perception/area_svg_updater.py`

目标：

- 将编辑规划实际执行到 SVG 上
- 分 chart type 处理具体修改逻辑

### 2.5 Render + Validate

重点路径：

- `src/chart_agent/perception/svg_renderer.py`
- `src/chart_agent/perception/render_validator.py`
- `tests/test_render_validator.py`

目标：

- 将更新后的 SVG 渲染成图像
- 对渲染质量和修改结果做校验

### 2.6 Answering

重点路径：

- `src/chart_agent/core/answerer.py`
- `src/chart_agent/core/vision_tool_phase.py`
- `tests/test_vision_tool_phase.py`

目标：

- 基于修改后的图表生成答案
- 可选地通过 tool phase 增强回答质量

## 3. 统一工程约束

为了让后续 agent 可以稳定接管，每个阶段都应统一提供：

- 输入
- 输出
- 中间日志
- 失败原因
- 可选重试策略

这一步非常关键。没有结构化工件，agent 只能“重跑”，不能“诊断”和“对比”。

## 4. M1：自动化复现实施方案

目标：先做自动复现最小闭环，确保一个命令能稳定产出结果、记录过程、支持恢复。

### 4.1 统一实验输入输出协议

建议定义标准化 `case record`，至少保存以下字段：

- `case_id`
- `question`
- `input_png`
- `input_svg`
- `chart_type`
- `parsed_edit_instruction`
- `parsed_qa_goal`
- `perception_summary`
- `edit_plan`
- `updated_svg_path`
- `rendered_image_path`
- `render_check`
- `answer_initial`
- `answer_tool_augmented`
- `answer`
- `status`
- `failure_stage`
- `failure_reason`
- `timings`
- `model_config`

建议输出目录规范：

- `output/runs/<run_id>/cases/<case_id>/`
- `output/runs/<run_id>/summary.json`
- `output/runs/<run_id>/metrics.json`
- `output/runs/<run_id>/report.md`

### 4.2 封装单例运行器

建议保留现有 `src/main.py` 作为入口，但抽出一个更稳定的逻辑层，供单例、批量和 agent 统一调用。

建议新增：

- `src/chart_agent/core/case_runner.py`
- `src/chart_agent/core/run_recorder.py`

职责建议：

- `case_runner` 负责流程编排执行
- `run_recorder` 负责落盘、日志和结构化工件保存

### 4.3 封装批量运行器

现有 `src/run_dataset_via_main.py` 建议从“批量调 main”演进成“批量调 case_runner”。

建议支持：

- `--input-dir`
- `--qa-index`
- `--max-cases`
- `--resume`
- `--run-id`
- `--chart-type-filter`
- `--only-failed`
- `--export-report`

### 4.4 最小验证链

修改局部逻辑时，优先运行相关测试：

```bash
pytest -q tests/test_main_question_split.py
pytest -q tests/test_svg_perceiver.py
pytest -q tests/test_scatter_svg_updater.py
pytest -q tests/test_area_svg_updater.py
pytest -q tests/test_render_validator.py
pytest -q tests/test_vision_tool_phase.py
```

最小单例复现命令：

```bash
python -m src.main \
  --question "Delete one series and tell me which year has the maximum value" \
  --image dataset/task2-line/094/094.png \
  --svg dataset/task2-line/094/094.svg
```

批量 smoke test：

```bash
python -m src.run_dataset_via_main \
  --input-dir dataset/task2-line \
  --qa-index 0 \
  --max-render-retries 2
```

### 4.5 M1 交付物

- 统一 run record 格式
- 可恢复的批量运行机制
- 每 case 的中间产物与错误原因
- 一个基础 summary 报告生成脚本

## 5. M2：扩展实验实施方案

重点不是“多跑几次”，而是让实验维度可枚举、可比较、可汇总。

### 5.1 实验维度建议

#### 模块消融

- 去掉 `render_validator`
- 去掉 `vision_tool_phase`
- 去掉 question split，直接端到端
- 去掉结构化 perception，只保留文本化 perception
- 只做 edit，不做 answer augmentation

#### 流程策略变化

- 单次更新 vs 多步更新
- 严格验证重试 vs 无验证重试
- chart-type 专用 updater vs 通用 updater
- planner 显式计划 vs 直接执行

#### 数据维度

- `scatter / line / area`
- 不同任务类型
- 不同问题复杂度
- 不同编辑指令类型
- 不同 answer 目标类型

#### 模型维度

- 不同 LLM backend
- 不同 temperature
- 不同 prompt 版本
- 是否启用 tool phase

### 5.2 实验配置机制

建议新增：

- `experiments/configs/`
- `experiments/reports/`
- `experiments/prompts/`

建议配置文件采用 `yaml` 或 `json`，内容包括：

- 数据集范围
- 模型配置
- 功能开关
- 输出位置
- 指标定义

建议的配置示例：

- `experiments/configs/baseline_task2.yaml`
- `experiments/configs/ablation_no_validator.yaml`
- `experiments/configs/ablation_no_tool_phase.yaml`

### 5.3 实验执行器

建议新增：

- `src/chart_agent/experiments/experiment_runner.py`
- `src/chart_agent/experiments/experiment_registry.py`
- `src/chart_agent/experiments/metrics.py`

功能目标：

- 读取实验配置
- 派发 dataset run
- 汇总结果
- 产出可对比报告

## 6. M3：Agentic Engineering 方案

这里的 agent 指工程协作代理，不是论文里的推理代理。建议先做 4 类 agent。

### 6.1 Repro Agent

职责：

- 跑指定实验配置
- 检查依赖
- 发现失败 case
- 自动 resume

输入：

- 实验配置
- 数据范围
- 模型设置

输出：

- `summary.json`
- `failed_cases.json`
- `metrics.json`

### 6.2 Failure Triage Agent

职责：

- 读取失败 case record
- 为失败归因
- 输出错误分布与代表性案例

建议归因标签固定为：

- `question_split_error`
- `perception_error`
- `planning_error`
- `svg_update_error`
- `render_error`
- `answer_error`
- `unknown`

输出：

- 各类错误数量
- 代表性 case
- Top failure patterns

### 6.3 Experiment Agent

职责：

- 自动枚举 ablation / robustness 配置
- 跑多组实验
- 生成对比表格和差异分析

输出：

- `comparison.csv`
- `comparison.md`

### 6.4 Report Agent

职责：

- 将结果整理为论文风格摘要
- 输出表格和 case study
- 突出性能变化、错误模式和结论

输出：

- `report.md`
- `paper_tables.md`

## 7. 建议的 Agent 工作流

建议先采用线性工作流，而不是复杂的 agent-to-agent 协商：

1. `Repro Agent` 跑 baseline
2. `Failure Triage Agent` 分析 baseline 失败
3. `Experiment Agent` 基于 baseline 派生扩展实验
4. `Report Agent` 汇总并输出结果

每个 agent 都只读写标准化工件：

- `case records`
- `metrics`
- `reports`

这样后续无论是用脚本、Codex 还是调度系统，成本都比较低。

## 8. 指标设计

不建议只看最终 answer 是否正确。至少应拆成三层指标。

### 8.1 流程指标

- parse success rate
- perception success rate
- render success rate
- final answer success rate
- average retry count
- average latency per stage

### 8.2 编辑质量指标

- SVG 是否成功修改
- 修改后是否能正常渲染
- 目标图元是否被正确修改
- 非目标区域是否被误伤

### 8.3 QA 质量指标

- answer correctness
- tool augmentation gain
- after-edit reasoning correctness

如果论文已有主指标，则优先兼容论文主指标；若论文未充分展开中间质量指标，则在工程层补齐。

## 9. 建议的目录增量

建议做小步增量，不做大重构：

- `src/chart_agent/core/case_runner.py`
- `src/chart_agent/core/run_recorder.py`
- `src/chart_agent/experiments/experiment_runner.py`
- `src/chart_agent/experiments/metrics.py`
- `src/chart_agent/experiments/reporting.py`
- `experiments/configs/`
- `experiments/reports/`
- `scripts/run_baseline.sh`
- `scripts/run_ablation.sh`
- `scripts/analyze_failures.sh`

保留现有 `src/main.py` 和 `src/run_dataset_via_main.py` 作为入口层，不直接堆叠过多新逻辑。

## 10. 第一批最值得做的实验

按优先级建议如下：

### 10.1 Baseline Reproduction

目标：

- 确认主流程在 `scatter / line / area` 上可稳定跑完

### 10.2 No Validator Ablation

目标：

- 验证 `render_validator` 是否真正带来收益
- 量化其对重试成本和最终成功率的影响

### 10.3 No Tool Phase Ablation

目标：

- 验证 `vision_tool_phase` 对 answer 质量的真实贡献

### 10.4 Chart-Type Breakdown

目标：

- 对比三类图表的性能差异
- 找出结构性薄弱点

### 10.5 Failure Taxonomy Study

目标：

- 证明系统瓶颈在感知、编辑、渲染还是回答
- 避免只看端到端正确率

### 10.6 Prompt / Policy Variant

目标：

- 验证 prompt 或 policy 调整对某一阶段的影响
- 避免把偶然波动误判为系统提升

## 11. 建议实施顺序

建议按以下节奏推进：

### 第 1 周：复现稳定化

- 抽出 `case_runner`
- 落标准化记录
- 跑通单例和小规模批量
- 补关键测试

### 第 2 周：实验标准化

- 做实验配置文件
- 做 metrics 汇总
- 做对比报告
- 跑 baseline 与 1 到 2 个 ablation

### 第 3 周：失败归因自动化

- 固化 failure taxonomy
- 自动输出失败分布
- 生成 case study

### 第 4 周：Agentic 自动化

- 用 agent 接管 baseline、ablation、triage、report
- 形成“给定配置即可自动跑完并出报告”的闭环

## 12. 当前最适合优先完成的 5 件事

建议先做下面五件，不要同时铺开过多模块：

1. 定义 `case record` 数据结构
2. 抽出 `case_runner`，让单例与批量统一调用
3. 给 `run_dataset_via_main` 增加 `resume + report`
4. 做一个 `baseline_task2.yaml` 配置
5. 做一个基础的 `metrics/report` 生成器

这五件完成后，后续的 agent 化和扩展实验会顺很多。

## 13. 下一步建议

在这份方案基础上，下一步最值得继续细化的是一份“面向实现的任务拆解”，建议进一步明确：

- 每个新增文件的职责
- `case record` 的字段定义
- `experiment config` 的示例格式
- `report.md` 的模板
- 第一阶段可直接执行的改动清单

如果继续推进，可以直接按 M1 的最小闭环开始落代码，而不是先设计更复杂的平台。
