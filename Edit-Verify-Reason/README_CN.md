# ChartAgent

[English README](README.md)

这个仓库包含 ChartAgent 当前的主框架、评测脚本，以及一份本地数据集快照，用于 SVG 图表编辑与问答。

注意：这份 README 放在 `Edit-Verify-Reason/` 目录下，但下面的命令默认都应在仓库根目录执行。

## 1. 项目做什么

ChartAgent 当前流程分为六步：

1. 把用户请求拆成更新意图和问答目标
2. 感知现有 SVG 图表结构
3. 规划更新步骤
4. 逐步修改 SVG
5. 渲染并校验结果
6. 基于更新后的图表回答问题，必要时进入视觉增强工具阶段

当前重点支持的图表类型：

- `area`
- `line`
- `scatter`

## 2. 当前仓库结构

当前仓库里最重要的目录是：

- `Edit-Verify-Reason/`：主框架代码
- `Edit-Verify-Reason/main.py`：单例入口
- `Edit-Verify-Reason/web_app.py`：本地 Web 入口
- `Edit-Verify-Reason/chart_agent/`：框架包
- `Edit-Verify-Reason/js/`：JS / Node 辅助脚本
- `Evaluation/`：批量运行、验证、smoke/ablation 脚本
- `CMod-QA/`：当前仓库里实际存在的数据集快照
- `Edit-Verify-Reason/tests/`：框架 pytest 测试
- `requirements.txt`：仓库根目录 Python 依赖

## 3. 环境准备

### 3.1 Python

CLI 和评测脚本在当前 Python 环境下可以运行。Web 入口仍然要求 Python 3.12 或更低版本，因为它依赖 `cgi`。

推荐在仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest
```

### 3.2 安装 `resvg`

SVG 渲染依赖 `resvg`，并且要求它在 `PATH` 里。

macOS 示例：

```bash
brew install resvg
which resvg
```

### 3.3 模型密钥

通过环境变量或 `.env` 注入：

```env
Aihubmix_API_KEY=
Siliconflow_API_KEY=
Doubao_API_KEY=
```

常用模型与运行配置：

```env
DEFAULT_MODEL=gpt

GPT_MODEL=gpt-5.2
GPT_BASE_URL=https://aihubmix.com/v1
GPT_TEMPERATURE=0.0

QWEN_MODEL=Qwen/Qwen3-VL-235B-A22B-Instruct
QWEN_BASE_URL=https://api.siliconflow.cn/v1
QWEN_TEMPERATURE=0.0

DOUBAO_MODEL=doubao-seed-1-6-251015
DOUBAO_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
DOUBAO_TEMPERATURE=0.0

CLAUDE_MODEL=claude-sonnet-4-6
CLAUDE_BASE_URL=https://aihubmix.com/v1
CLAUDE_TEMPERATURE=0.0

GEMINI_MODEL=gemini-2.5-flash
GEMINI_BASE_URL=https://aihubmix.com/v1
GEMINI_TEMPERATURE=0.0

SPLITTER_MODEL=gpt
PLANNER_MODEL=gpt
TOOL_PLANNER_MODEL=gpt
EXECUTOR_MODEL=gpt
VALIDATOR_MODEL=qwen
ANSWER_MODEL=gpt

PERCEPTION_MAX_RETRIES=2
SVG_PERCEPTION_MODE=rules
SVG_UPDATE_MODE=rules
CHART_WEB_HOST=127.0.0.1
CHART_WEB_PORT=8008
```

当前可用模型 key：

- `gpt`
- `qwen`
- `doubao`
- `claude`
- `gemini`
- `qwen-aihubmix`

模式归一化规则：

- `SVG_PERCEPTION_MODE`：`rules`、`llm_summary`，以及别名 `llm`
- `SVG_UPDATE_MODE`：`rules`、`llm_intent`、`htn`，以及别名 `llm`、`hier`、`hierarchical`

## 4. 数据目录

当前仓库实际存在的是 `CMod-QA/`，不是 `dataset/`。

当前数据任务目录：

- `CMod-QA/task1-area/`
- `CMod-QA/task2-line/`
- `CMod-QA/task3-scatter/`

一个 case 目录通常包含：

- `{id}.svg`
- `{id}.png`
- `{id}.json`
- `{id}.csv`
- `{id}_aug.svg`
- `{id}_aug.png`

这里有一个关键区别：

- `Edit-Verify-Reason/main.py` 只会使用你手动传进去的 `question / image / svg`
- `Evaluation/run_dataset_via_main.py` 会额外读取 case 的 JSON，并自动注入 `data_change`

所以像 `these points`、`listed value revisions` 这种依赖 JSON 补充信息的数据集题目，通常应该优先用 batch runner，或者在 Python API 里手动传 `structured_update_context`，而不是直接裸跑单例 CLI。

## 5. 如何使用

### 5.1 单例 CLI

适合问题文本本身已经把更新信息说完整的场景。

示例：

```bash
python Edit-Verify-Reason/main.py \
  --question "After deleting the category Aethelgard Medical Board, how many times do the lines for Lumina Health Authority and Stellar Public Health intersect?" \
  --image CMod-QA/task2-line/del/000/000.png \
  --svg CMod-QA/task2-line/del/000/000.svg \
  --experiment-mode full
```

CLI 参数：

- `--question`
- `--image`
- `--svg`
- `--text_spec`
- `--experiment-mode`

当前支持的实验模式：

- `full`
- `wo_question_decomposition`
- `wo_svg_update`

### 5.2 批量 / 数据集入口

对数据集 case 来说，这是当前更推荐的入口，因为它会自动读取 `data_change`。

已实测通过的命令：

```bash
python Evaluation/run_dataset_via_main.py \
  --input-dir CMod-QA/task1-area/add-change \
  --qa-index 0 \
  --limit 1 \
  --max-render-retries 2 \
  --record-root output/dataset_records_stage2_check
```

这个命令在当前仓库里已经验证过，成功跑通了 `000` case。

常用参数：

- `--input-dir`
- `--qa-index`
- `--max-render-retries`
- `--limit`
- `--record-root`
- `--resume`
- `--model`
- `--experiment-mode`

另一个例子：

```bash
python Evaluation/run_dataset_via_main.py \
  --input-dir CMod-QA/task2-line/del \
  --qa-index 0 \
  --limit 10 \
  --resume \
  --model gpt
```

### 5.3 Python API

如果你要手动覆盖模型，或者自己注入结构化更新上下文，直接调用 `run_main()` 更合适。

```python
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()
FRAMEWORK_ROOT = PROJECT_ROOT / "Edit-Verify-Reason"
sys.path.insert(0, str(FRAMEWORK_ROOT))

from main import run_main

case_dir = PROJECT_ROOT / "CMod-QA" / "task1-area" / "add-change" / "000"
payload = json.loads((case_dir / "000.json").read_text(encoding="utf-8"))
qa = payload["QA"][0]

result = run_main(
    {
        "question": qa["question"],
        "image_path": str(case_dir / "000.png"),
        "svg_path": str(case_dir / "000.svg"),
        "structured_update_context": {
            "data_change": payload.get("data_change", {}),
        },
        "max_render_retries": 2,
        "experiment_mode": "full",
        "model_overrides": {
            "splitter": "gpt",
            "planner": "gpt",
            "executor": "gpt",
            "answer": "gpt",
            "tool_planner": "gpt",
        },
    }
)
```

`run_main()` 常见输入字段：

- `question`
- `image_path`
- `svg_path`
- `text_spec`
- `max_render_retries`
- `render_output_dir`
- `experiment_mode`
- `svg_perception_mode`
- `svg_update_mode`
- `model_overrides`
- `structured_update_context`
- `update_question`
- `qa_question`
- `auto_split_question`
- `answer_image_path`

### 5.4 Web

从仓库根目录启动：

```bash
python Edit-Verify-Reason/web_app.py
```

默认地址：

```text
http://127.0.0.1:8008
```

当前 Web 端点：

- `GET /`
- `GET /api/models`
- `POST /api/run`
- `POST /api/run/stream`
- `GET /files/...`

当前限制：

- `Edit-Verify-Reason/web_app.py` 依赖 `cgi`
- 在 Python 3.13+ 下会报 `ModuleNotFoundError: No module named 'cgi'`
- Web 建议使用 Python 3.12 或更低版本

### 5.5 验证脚本

比对生成 SVG 与 GT：

```bash
python Evaluation/validate_svg_matches.py \
  --pred-root output/dataset_records_stage2_check/add_change \
  --dataset-dir CMod-QA/task1-area/add-change
```

area 数值验证：

```bash
python Evaluation/validate_aug_svg.py \
  --pred-root output/dataset_records_stage2_check/add_change \
  --gt-root CMod-QA/task1-area/add-change \
  --mode top-only
```

验证整个 records 根目录：

```bash
python Evaluation/run_validate_dataset_records.py \
  --records-root output/dataset_records_stage2_check \
  --limit 20
```

对已有工具增强图重新回答：

```bash
python Evaluation/run_tool_aug_answer.py \
  --case-dir output/dataset_records_stage2_check/add_change/000 \
  --resume
```

可选的可视化批量入口：

```bash
python Evaluation/run_dataset_vis_main.py \
  --task task1-area/add-change \
  --dataset-root CMod-QA \
  --limit 1 \
  --out output/vis_report.json
```

## 6. 输出内容

单次运行常见 JSON 字段：

- `output_image_path`
- `render_check`
- `attempt_logs`
- `operation_plan`
- `question_split`
- `perception_steps`
- `answer_original`
- `answer_initial`
- `answer_tool_augmented`
- `answer`
- `resolved_task_models`
- `experiment_mode`

批量运行常见输出：

- `<run_dir>/<case_id>/result.json`
- `<run_dir>/<case_id>/answer.txt`
- `<run_dir>/<case_id>/error.txt`
- `<run_dir>/<case_id>/<case_id>_updated.svg`
- `<run_dir>/<case_id>/<case_id>_updated.png`
- `<run_dir>/<case_id>/<case_id>_tool_aug.svg`
- `<run_dir>/<case_id>/<case_id>_tool_aug.png`
- `<run_dir>/summary.json`
- `<run_dir>/summary.txt`

当前已验证的输出样例：

- `output/dataset_records_stage2_check/add_change/000/result.json`
- `output/dataset_records_stage2_check/add_change/summary.txt`

## 7. 视觉增强工具阶段

当前 `Edit-Verify-Reason/main.py` 中的触发阈值是：

- `answer_initial.confidence < 0.85`

以下问题类型还会强制进入工具阶段：

- scatter 聚类问题
- line 交点 / 相交问题

工具注册位置在 `Edit-Verify-Reason/chart_agent/core/vision_tool_phase.py`。

## 8. 测试

跑全量测试：

```bash
pytest -q Edit-Verify-Reason/tests
```

常用定向测试：

```bash
pytest -q Edit-Verify-Reason/tests/test_main_question_split.py
pytest -q Edit-Verify-Reason/tests/test_main_step_retry.py
pytest -q Edit-Verify-Reason/tests/test_render_validator.py
pytest -q Edit-Verify-Reason/tests/test_area_svg_updater.py
pytest -q Edit-Verify-Reason/tests/test_scatter_svg_updater.py
```

## 9. JS 辅助脚本

JS 依赖在 `Edit-Verify-Reason/js/package.json` 里。

安装方式：

```bash
cd Edit-Verify-Reason/js
npm install
```

当前限制：

- `Edit-Verify-Reason/js/run_dataset_case.py`
- `Edit-Verify-Reason/js/run_dataset_case.mjs`

这两个脚本仍然写死了顶层 `dataset/` 目录，所以在当前只包含 `CMod-QA/` 的仓库快照里不能直接开箱即用，除非你额外准备兼容目录或自行改路径。

## 10. 当前限制

- Web 入口要求 Python 3.12 或更低版本
- 多个便捷评测脚本仍然硬编码了 `dataset/...`：
- `Evaluation/run_smoke_task1_task2_models.py`
- `Evaluation/run_ablation_gpt_10cases.py`
- `Evaluation/run_eval_session.sh`
- 当前仓库里实际存在的是 `CMod-QA/`，不是 `dataset/`
- 单例 CLI 不会自动读取 case JSON 里的 `data_change`
- 现在只保留 `Edit-Verify-Reason/tests/` 下的框架测试，评测脚本测试已移除
- `requirements.txt` 不会自动安装 `pytest`
