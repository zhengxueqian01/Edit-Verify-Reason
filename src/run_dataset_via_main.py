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
from chart_agent.config import get_task_model_config
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


def extract_answer_text_by_key(result: dict[str, Any], key: str) -> str:
    value = result.get(key)
    if isinstance(value, dict):
        return str(value.get("answer") or "").strip()
    if isinstance(value, str):
        return value.strip()
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
    if parent_name == "dataset":
        try:
            parent_name = get_task_model_config("answer").name
        except Exception:
            parent_name = "answer"
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
    answer_original_evaluated = 0
    answer_original_matched = 0
    answer_initial_evaluated = 0
    answer_initial_matched = 0

    for case_dir in case_dirs:
        if args.limit and total >= args.limit:
            break
        total += 1
        case_id = case_dir.name
        json_path = case_dir / f"{case_id}.json"
        svg_path = case_dir / f"{case_id}.svg"
        image_path = case_dir / f"{case_id}.png"
        case_out_dir = run_dir / case_id
        case_out_dir.mkdir(parents=True, exist_ok=True)

        payload = load_json(json_path)
        qa_question, expected_answer = choose_qa(payload, args.qa_index)
        split_update_question = ""

        result: dict[str, Any]
        err = ""
        try:
            result = run_main(
                {
                    # Always pass the original QA sentence and let run_main split it into
                    # update_question / qa_question with the splitter model.
                    "question": qa_question,
                    "max_render_retries": args.max_render_retries,
                    "svg_path": str(svg_path),
                    "text_spec": None,
                }
            )
            split_update_question = str(result.get("update_question") or "")
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
            tool_phase = result.get("tool_phase")
            if isinstance(tool_phase, dict):
                aug_svg_raw = str(tool_phase.get("augmented_svg_path") or "").strip()
                if aug_svg_raw:
                    aug_svg_path = Path(aug_svg_raw).expanduser()
                    if aug_svg_path.exists() and aug_svg_path.is_file():
                        aug_svg_target = case_out_dir / f"{case_id}_tool_aug.svg"
                        shutil.copy2(aug_svg_path, aug_svg_target)
                        result["record_tool_aug_svg_path"] = str(aug_svg_target)
                aug_raw = str(tool_phase.get("augmented_image_path") or "").strip()
                if aug_raw:
                    aug_path = Path(aug_raw).expanduser()
                    if aug_path.exists() and aug_path.is_file():
                        aug_suffix = aug_path.suffix.lower() or ".png"
                        aug_target = case_out_dir / f"{case_id}_tool_aug{aug_suffix}"
                        shutil.copy2(aug_path, aug_target)
                        result["record_tool_aug_image_path"] = str(aug_target)

            answer_initial_text = extract_answer_text_by_key(result, "answer_initial")
            answer_initial_match = is_answer_match(expected_answer, answer_initial_text)
            answer_original_text = extract_answer_text_by_key(result, "answer_original")
            answer_original_match = is_answer_match(expected_answer, answer_original_text)
            answer_tool_aug_text = extract_answer_text(result)
            answer_tool_aug_match = is_answer_match(expected_answer, answer_tool_aug_text)
            answer_evaluated += 1
            if answer_tool_aug_match:
                answer_matched += 1
            answer_original_evaluated += 1
            if answer_original_match:
                answer_original_matched += 1
            answer_initial_evaluated += 1
            if answer_initial_match:
                answer_initial_matched += 1
            answer_lines = [
                f"Question: {qa_question}",
                f"Expected: {expected_answer}",
                f"Original Image Answer: {answer_original_text}",
                f"Original Image Match: {answer_original_match}",
                f"Initial Answer: {answer_initial_text}",
                f"Initial Match: {answer_initial_match}",
                f"Tool Augmented Answer: {answer_tool_aug_text}",
                f"Tool Augmented Match: {answer_tool_aug_match}",
                "",
                "=== Prompt: answer_original ===",
                str(((result.get("answer_original") or {}).get("prompt")) if isinstance(result.get("answer_original"), dict) else ""),
                "",
                "=== Prompt: answer_initial ===",
                str(((result.get("answer_initial") or {}).get("prompt")) if isinstance(result.get("answer_initial"), dict) else ""),
                "",
                "=== Prompt: answer_tool_augmented ===",
                str(((result.get("answer_tool_augmented") or {}).get("prompt")) if isinstance(result.get("answer_tool_augmented"), dict) else ""),
                "",
                "=== LLM Raw: answer_original ===",
                str(((result.get("answer_original") or {}).get("llm_raw")) if isinstance(result.get("answer_original"), dict) else ""),
                "",
                "=== LLM Raw: answer_initial ===",
                str(((result.get("answer_initial") or {}).get("llm_raw")) if isinstance(result.get("answer_initial"), dict) else ""),
                "",
                "=== LLM Raw: answer_tool_augmented ===",
                str(((result.get("answer_tool_augmented") or {}).get("llm_raw")) if isinstance(result.get("answer_tool_augmented"), dict) else ""),
                "",
                "=== LLM Raw: answer(final) ===",
                str(((result.get("answer") or {}).get("llm_raw")) if isinstance(result.get("answer"), dict) else ""),
                "",
                "=== LLM Raw: tool_planner ===",
                str(
                    (
                        ((result.get("tool_phase") or {}).get("planner") or {}).get("llm_raw")
                        if isinstance(result.get("tool_phase"), dict)
                        else ""
                    )
                ),
            ]
            (case_out_dir / "answer.txt").write_text("\n".join(answer_lines) + "\n", encoding="utf-8")
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
                "update_question": split_update_question,
                "expected_answer": expected_answer,
                "answer_original": (extract_answer_text_by_key(result, "answer_original") if not err else ""),
                "answer_original_match": (
                    is_answer_match(expected_answer, extract_answer_text_by_key(result, "answer_original"))
                    if not err
                    else False
                ),
                "answer_initial": (extract_answer_text_by_key(result, "answer_initial") if not err else ""),
                "answer_initial_match": (
                    is_answer_match(expected_answer, extract_answer_text_by_key(result, "answer_initial"))
                    if not err
                    else False
                ),
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
        "answer_original_evaluated": answer_original_evaluated,
        "answer_original_matched": answer_original_matched,
        "answer_original_accuracy": (
            answer_original_matched / answer_original_evaluated if answer_original_evaluated else 0.0
        ),
        "answer_initial_evaluated": answer_initial_evaluated,
        "answer_initial_matched": answer_initial_matched,
        "answer_initial_accuracy": (
            answer_initial_matched / answer_initial_evaluated if answer_initial_evaluated else 0.0
        ),
        "answer_evaluated": answer_evaluated,
        "answer_matched": answer_matched,
        "answer_accuracy": (answer_matched / answer_evaluated if answer_evaluated else 0.0),
        "answer_tool_aug_evaluated": answer_evaluated,
        "answer_tool_aug_matched": answer_matched,
        "answer_tool_aug_accuracy": (answer_matched / answer_evaluated if answer_evaluated else 0.0),
        "items": summary,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
