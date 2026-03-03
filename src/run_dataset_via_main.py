#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from main import run_main
from chart_agent.perception.svg_renderer import default_output_paths


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch run dataset folder via src/main.py run_main()."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Dataset folder path, e.g. dataset/task1-mix-area/add-change",
    )
    parser.add_argument("--qa-index", type=int, default=0, help="QA index from JSON QA list.")
    parser.add_argument(
        "--question-only",
        action="store_true",
        help="Use QA question as update instruction (skip synthesized update instruction).",
    )
    parser.add_argument(
        "--max-render-retries",
        type=int,
        default=2,
        help="Retry times when render validation fails.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max cases to run; 0 for all.")
    parser.add_argument(
        "--record-root",
        default="output/dataset_records",
        help="Root directory for run records.",
    )
    return parser.parse_args()


def resolve_input_dir(raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    candidate = (PROJECT_ROOT / p).resolve()
    if candidate.exists():
        return candidate
    return (DATASET_ROOT / raw).resolve()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def choose_qa(payload: dict[str, Any], qa_index: int) -> tuple[str, Any]:
    qa = payload.get("QA")
    if isinstance(qa, list) and qa:
        idx = max(0, min(qa_index, len(qa) - 1))
        item = qa[idx]
        if isinstance(item, dict):
            q = str(item.get("question") or "").strip()
            if q:
                return q, item.get("answer")
    q = str(payload.get("question") or "").strip()
    if q:
        return q, payload.get("answer")
    raise ValueError("No question found in JSON.")


def synthesize_instruction(payload: dict[str, Any]) -> str:
    op = str(payload.get("operation") or "").lower()
    target = payload.get("operation_target") or {}
    change = payload.get("data_change") or {}
    chart_type = str(payload.get("chart_type") or "").lower()
    parts: list[str] = []

    if chart_type == "scatter":
        points = change.get("points") if isinstance(change, dict) else None
        if isinstance(points, list) and points:
            text_points: list[str] = []
            for p in points:
                if not isinstance(p, dict):
                    continue
                try:
                    text_points.append(f"({float(p.get('x'))}, {float(p.get('y'))})")
                except Exception:
                    continue
            if text_points:
                parts.append("Add points " + ", ".join(text_points))

    if "delete" in op or "del" in op:
        names = target.get("category_name") if isinstance(target, dict) else None
        if not names and isinstance(target, dict):
            names = target.get("del_category")
        if isinstance(names, list) and names:
            parts.append("Delete categories " + ", ".join(f"\"{n}\"" for n in names))
        elif isinstance(names, str) and names.strip():
            parts.append(f"Delete category \"{names.strip()}\"")

    add_block = change.get("add") if isinstance(change, dict) else None
    if isinstance(add_block, dict):
        add_name = target.get("add_category") if isinstance(target, dict) else None
        values = add_block.get("values")
        if isinstance(values, list) and values:
            values_text = ", ".join(str(v) for v in values)
            if isinstance(add_name, str) and add_name.strip():
                parts.append(f"Add series \"{add_name.strip()}\" : [{values_text}]")
            else:
                parts.append(f"Add series: [{values_text}]")

    change_block = change.get("change") if isinstance(change, dict) else None
    if isinstance(change_block, dict):
        change_name = target.get("change_category") if isinstance(target, dict) else None
        years = change_block.get("years")
        values = change_block.get("values")
        year = years[0] if isinstance(years, list) and years else None
        value = values[0] if isinstance(values, list) and values else None
        if year is not None and value is not None:
            if isinstance(change_name, str) and change_name.strip():
                parts.append(f"Change \"{change_name.strip()}\" at year {year} to {value}")
            else:
                parts.append(f"Change value at year {year} to {value}")

    return "; then ".join(parts).strip()


def list_case_dirs(input_dir: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(input_dir.iterdir()):
        if not p.is_dir() or not p.name.isdigit():
            continue
        json_path = p / f"{p.name}.json"
        svg_path = p / f"{p.name}.svg"
        if json_path.exists() and svg_path.exists():
            out.append(p)
    return out


def extract_answer_text(result: dict[str, Any]) -> str:
    answer = result.get("answer")
    if isinstance(answer, dict):
        return str(answer.get("answer") or "").strip()
    if isinstance(answer, str):
        return answer.strip()
    return ""


def _extract_years(text: str) -> list[str]:
    return re.findall(r"\b\d{4}\b", text or "")


def _extract_numbers(text: str) -> list[float]:
    raw = re.findall(r"-?\d+(?:\.\d+)?", text or "")
    out: list[float] = []
    for item in raw:
        try:
            out.append(float(item))
        except ValueError:
            continue
    return out


def is_answer_match(expected: Any, answer_text: str) -> bool:
    if expected is None:
        return False
    actual = (answer_text or "").strip()
    if not actual:
        return False
    if isinstance(expected, (int, float)):
        nums = _extract_numbers(actual)
        if not nums:
            return False
        return abs(nums[0] - float(expected)) < 1e-6
    exp = str(expected).strip()
    if not exp:
        return False
    exp_years = _extract_years(exp)
    if exp_years:
        act_years = _extract_years(actual)
        if not act_years:
            return False
        return act_years[0] == exp_years[0]
    if exp in actual:
        return True
    exp_nums = _extract_numbers(exp)
    act_nums = _extract_numbers(actual)
    if len(exp_nums) == 1 and act_nums:
        return abs(exp_nums[0] - act_nums[0]) < 1e-6
    return False


def extract_final_svg_path(result: dict[str, Any], source_svg: Path, chart_type: str) -> Path | None:
    attempt_logs = result.get("attempt_logs")
    if isinstance(attempt_logs, list):
        for attempt in reversed(attempt_logs):
            if not isinstance(attempt, dict):
                continue
            step_logs = attempt.get("step_logs")
            if not isinstance(step_logs, list):
                continue
            for step in reversed(step_logs):
                if not isinstance(step, dict):
                    continue
                p = step.get("output_svg_path")
                if isinstance(p, str) and p.strip():
                    path = Path(p.strip())
                    if path.exists():
                        return path

    default_svg, _ = default_output_paths(str(source_svg), chart_type)
    fallback = Path(default_svg)
    if fallback.exists():
        return fallback
    return None


def main() -> None:
    args = parse_args()
    input_dir = resolve_input_dir(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input folder not found: {input_dir}")

    case_dirs = list_case_dirs(input_dir)
    if not case_dirs:
        raise SystemExit(f"No valid case folders found under: {input_dir}")

    parent_name = input_dir.parent.name.strip() if input_dir.parent else ""
    child_name = input_dir.name.strip() or "dataset_run"
    if parent_name:
        rel_name = f"{parent_name}_{child_name}"
    else:
        rel_name = child_name
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    record_root = Path(args.record_root)
    if not record_root.is_absolute():
        record_root = PROJECT_ROOT / record_root
    run_dir = record_root / f"{rel_name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, Any]] = []
    total = 0
    success = 0
    failed = 0
    answer_evaluated = 0
    answer_matched = 0

    for case_dir in case_dirs:
        if args.limit and total >= args.limit:
            break
        total += 1
        case_id = case_dir.name
        json_path = case_dir / f"{case_id}.json"
        svg_path = case_dir / f"{case_id}.svg"
        case_out_dir = run_dir / case_id
        case_out_dir.mkdir(parents=True, exist_ok=True)

        payload = load_json(json_path)
        qa_question, expected_answer = choose_qa(payload, args.qa_index)
        update_question = qa_question if args.question_only else (synthesize_instruction(payload) or qa_question)

        result: dict[str, Any]
        err = ""
        try:
            result = run_main(
                {
                    "question": f"{update_question}. {qa_question}",
                    "update_question": update_question,
                    "qa_question": qa_question,
                    "chart_type_hint": str((payload or {}).get("chart_type") or "").lower(),
                    "max_render_retries": args.max_render_retries,
                    "svg_path": str(svg_path),
                    "text_spec": None,
                    "image_path": None,
                }
            )
            result["case"] = case_id
            result["case_dir"] = str(case_dir)
            result["json_path"] = str(json_path)
            result["svg_path"] = str(svg_path)

            chart_type = str((payload or {}).get("chart_type") or "unknown")
            final_svg = extract_final_svg_path(result, svg_path, chart_type)
            if final_svg and final_svg.exists():
                shutil.copy2(final_svg, case_out_dir / f"{case_id}_updated.svg")
                result["record_svg_path"] = str(case_out_dir / f"{case_id}_updated.svg")
            output_image_path = Path(str(result.get("output_image_path") or "")).expanduser()
            if output_image_path.exists() and output_image_path.is_file():
                png_target = case_out_dir / f"{case_id}_updated{output_image_path.suffix.lower() or '.png'}"
                shutil.copy2(output_image_path, png_target)
                result["record_image_path"] = str(png_target)

            answer_text = extract_answer_text(result)
            answer_match = is_answer_match(expected_answer, answer_text)
            answer_evaluated += 1
            if answer_match:
                answer_matched += 1
            (case_out_dir / "answer.txt").write_text(answer_text + "\n", encoding="utf-8")
            (case_out_dir / "result.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            success += 1
        except Exception as exc:
            failed += 1
            err = f"{type(exc).__name__}: {exc}"
            (case_out_dir / "error.txt").write_text(err + "\n", encoding="utf-8")
            result = {
                "case": case_id,
                "error": err,
                "case_dir": str(case_dir),
                "json_path": str(json_path),
                "svg_path": str(svg_path),
            }

        summary.append(
            {
                "case": case_id,
                "ok": not bool(err),
                "question": qa_question,
                "update_question": update_question,
                "expected_answer": expected_answer,
                "answer": extract_answer_text(result) if not err else "",
                "answer_match": (is_answer_match(expected_answer, extract_answer_text(result)) if not err else False),
                "record_dir": str(case_out_dir),
                "error": err,
            }
        )
        status = "OK" if not err else "FAIL"
        print(f"[{status}] {case_id} -> {case_out_dir}")

    summary_payload = {
        "input_dir": str(input_dir),
        "run_dir": str(run_dir),
        "total": total,
        "success": success,
        "failed": failed,
        "answer_evaluated": answer_evaluated,
        "answer_matched": answer_matched,
        "answer_accuracy": (answer_matched / answer_evaluated if answer_evaluated else 0.0),
        "items": summary,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
