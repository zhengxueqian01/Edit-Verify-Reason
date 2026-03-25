#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DATASET_SCRIPT = PROJECT_ROOT / "src" / "run_dataset_via_main.py"
DEFAULT_RECORD_ROOT = PROJECT_ROOT / "output" / "dataset_records"
DEFAULT_MODELS = ("claude", "gemini", "gpt", "qwen")
TASK_SPECS: tuple[tuple[str, str], ...] = (
    ("task1_add", "dataset/task1/add"),
    ("task1_add-change", "dataset/task1/add-change"),
    ("task1_add-del", "dataset/task1/add-del"),
    ("task1_change", "dataset/task1/change"),
    ("task1_del", "dataset/task1/del"),
    ("task1_del-change", "dataset/task1/del-change"),
    ("task2_del", "dataset/task2-line/del"),
    ("task2_del-add", "dataset/task2-line/del-add"),
    ("task2_del-change", "dataset/task2-line/del-change"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run task1/task2 smoke batches for multiple models with resume support."
    )
    parser.add_argument(
        "--session-name",
        default="smoke_task1_task2_10cases",
        help="Session folder name under record root.",
    )
    parser.add_argument(
        "--record-root",
        default=str(DEFAULT_RECORD_ROOT),
        help="Root directory for session outputs.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Model keys to run, e.g. claude gemini gpt qwen.",
    )
    parser.add_argument(
        "--cases-per-task",
        type=int,
        default=10,
        help="Max cases per task.",
    )
    parser.add_argument("--qa-index", type=int, default=0, help="QA index from JSON QA list.")
    parser.add_argument(
        "--max-render-retries",
        type=int,
        default=2,
        help="Retry times when render validation fails.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume existing session outputs and skip cases with result.json.",
    )
    return parser.parse_args()


def build_task_commands(
    *,
    python_executable: str,
    session_root: Path,
    models: list[str],
    cases_per_task: int,
    qa_index: int,
    max_render_retries: int,
    resume: bool,
) -> list[list[str]]:
    commands: list[list[str]] = []
    for model in models:
        model_root = session_root / model
        for _, input_dir in TASK_SPECS:
            cmd = [
                python_executable,
                str(RUN_DATASET_SCRIPT),
                "--input-dir",
                input_dir,
                "--qa-index",
                str(qa_index),
                "--max-render-retries",
                str(max_render_retries),
                "--limit",
                str(cases_per_task),
                "--record-root",
                str(model_root),
                "--model",
                model,
            ]
            if resume:
                cmd.append("--resume")
            commands.append(cmd)
    return commands


def main() -> None:
    args = parse_args()
    record_root = Path(args.record_root)
    if not record_root.is_absolute():
        record_root = (PROJECT_ROOT / record_root).resolve()
    session_root = record_root / args.session_name
    session_root.mkdir(parents=True, exist_ok=True)

    commands = build_task_commands(
        python_executable=sys.executable,
        session_root=session_root,
        models=[str(model).strip() for model in args.models if str(model).strip()],
        cases_per_task=args.cases_per_task,
        qa_index=args.qa_index,
        max_render_retries=args.max_render_retries,
        resume=args.resume,
    )

    for idx, cmd in enumerate(commands, start=1):
        task_input = cmd[cmd.index("--input-dir") + 1]
        model = cmd[cmd.index("--model") + 1]
        print(f"[{idx}/{len(commands)}] model={model} input={task_input}")
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


if __name__ == "__main__":
    main()
