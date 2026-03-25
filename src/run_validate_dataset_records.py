#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_RECORDS_ROOT = PROJECT_ROOT / "output" / "dataset_records"
SVG_MATCH_ROOT = PROJECT_ROOT / "output" / "svg_match"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SVG validation for each task under every model folder in output/dataset_records."
    )
    parser.add_argument(
        "--records-root",
        default=str(DATASET_RECORDS_ROOT),
        help="Dataset records root. Default: output/dataset_records",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Only validate the specified model folder(s). Repeatable.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Pass-through case limit for each task validation; 0 means all.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately when any task validation command fails.",
    )
    return parser.parse_args()


def iter_model_dirs(records_root: Path, selected_models: list[str]) -> list[Path]:
    model_names = set(selected_models)
    model_dirs: list[Path] = []
    for path in sorted(records_root.iterdir()):
        if not path.is_dir():
            continue
        if model_names and path.name not in model_names:
            continue
        if any(child.is_dir() and child.name.startswith("task") for child in path.iterdir()):
            model_dirs.append(path)
    return model_dirs


def _resolve_task1_dataset_dir(operation: str) -> str:
    for candidate in (f"task1/{operation}", f"task1-mix-area/{operation}"):
        if (PROJECT_ROOT / "dataset" / candidate).exists():
            return candidate
    return f"task1/{operation}"


def infer_dataset_dir(task_run_name: str) -> str:
    if task_run_name.startswith("task1_"):
        operation = task_run_name.removeprefix("task1_").rsplit("_", 2)[0]
        return _resolve_task1_dataset_dir(operation)
    if task_run_name.startswith("task2_"):
        operation = task_run_name.removeprefix("task2_").rsplit("_", 2)[0]
        return f"task2-line/{operation}"
    if task_run_name == "task3_scatter-cluster" or task_run_name.startswith("task3_scatter-cluster_"):
        return "task3-scatter-cluster"
    raise ValueError(f"Unsupported task run directory: {task_run_name}")


def build_output_path(pred_root: Path, records_root: Path) -> Path:
    return (SVG_MATCH_ROOT / records_root.name / pred_root.relative_to(records_root)).with_suffix(".json")


def run_validation(pred_root: Path, dataset_dir: str, limit: int, records_root: Path) -> dict[str, Any]:
    out_path = build_output_path(pred_root, records_root)
    cmd = [
        sys.executable,
        "-m",
        "src.validate_svg_matches",
        "--pred-root",
        str(pred_root),
        "--dataset-dir",
        dataset_dir,
        "--out",
        str(out_path),
    ]
    if limit:
        cmd.extend(["--limit", str(limit)])
    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    stdout = completed.stdout.strip()
    parsed: dict[str, Any] | None = None
    if stdout:
        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError:
            parsed = None
    return {
        "cmd": cmd,
        "returncode": completed.returncode,
        "stdout": stdout,
        "stderr": completed.stderr.strip(),
        "result": parsed,
        "out_path": str(out_path),
    }


def main() -> None:
    args = parse_args()
    records_root = Path(args.records_root).expanduser().resolve()
    if not records_root.exists() or not records_root.is_dir():
        raise SystemExit(f"Dataset records root not found: {records_root}")

    model_dirs = iter_model_dirs(records_root, args.model)
    if not model_dirs:
        raise SystemExit(f"No model directories found under: {records_root}")

    summary: list[dict[str, Any]] = []
    has_failures = False

    for model_dir in model_dirs:
        for task_dir in sorted(path for path in model_dir.iterdir() if path.is_dir() and path.name.startswith("task")):
            dataset_dir = infer_dataset_dir(task_dir.name)
            run_info = run_validation(task_dir, dataset_dir, args.limit, records_root)
            entry = {
                "model": model_dir.name,
                "task_run": task_dir.name,
                "dataset_dir": dataset_dir,
                "out_path": run_info["out_path"],
                "returncode": run_info["returncode"],
            }
            result = run_info.get("result")
            if isinstance(result, dict):
                entry["compared"] = result.get("compared")
                entry["average_score"] = result.get("average_score")
                entry["low_score_case_count"] = result.get("low_score_case_count")
                entry["failed"] = result.get("failed")
                entry["missing_pred"] = result.get("missing_pred")
                entry["missing_gt"] = result.get("missing_gt")
            if run_info["stderr"]:
                entry["stderr"] = run_info["stderr"]
            summary.append(entry)

            if run_info["returncode"] != 0:
                has_failures = True
                print(json.dumps(entry, ensure_ascii=False), file=sys.stderr)
                if args.fail_fast:
                    print(json.dumps(summary, ensure_ascii=False, indent=2))
                    raise SystemExit(run_info["returncode"])
            else:
                print(f"[ok] {model_dir.name}/{task_dir.name} -> {run_info['out_path']}", file=sys.stderr)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if has_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
