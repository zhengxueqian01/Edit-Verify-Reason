#貌似没用的，不知道什么时候生成的，先保留着吧
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from main import run_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch run dataset cases via src/main.py:run_main()."
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Task path under dataset, e.g. task1-mix-area/add-change, task2-line, task3-scatter-cluster.",
    )
    parser.add_argument(
        "--dataset-root",
        default="dataset",
        help="Dataset root path (relative to repo root by default).",
    )
    parser.add_argument("--qa-index", type=int, default=0, help="QA index from JSON QA list.")
    parser.add_argument(
        "--question-only",
        action="store_true",
        help="Use QA question directly instead of synthesized update instruction.",
    )
    parser.add_argument(
        "--max-render-retries",
        type=int,
        default=2,
        help="Retry times when render validation fails.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max cases to run; 0 for all.")
    parser.add_argument(
        "--out",
        default="output/dataset_eval_results.txt",
        help="Output report path.",
    )
    parser.add_argument(
        "--write-jsonl",
        action="store_true",
        help="Also write per-case raw output jsonl beside --out.",
    )
    return parser.parse_args()


def choose_qa(payload: dict[str, Any], qa_index: int) -> tuple[str, Any]:
    qa = payload.get("QA")
    if isinstance(qa, list) and qa:
        idx = max(0, min(qa_index, len(qa) - 1))
        item = qa[idx]
        if isinstance(item, dict):
            question = str(item.get("question") or "").strip()
            if question:
                return question, item.get("answer")
    fallback_q = str(payload.get("question") or "").strip()
    if fallback_q:
        return fallback_q, payload.get("answer")
    raise ValueError("No question found in payload.")


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
        add_blocks = [add_block]
    elif isinstance(add_block, list):
        add_blocks = [item for item in add_block if isinstance(item, dict)]
    else:
        add_blocks = []
    if add_blocks:
        add_name = target.get("add_category") if isinstance(target, dict) else None
        add_names = add_name if isinstance(add_name, list) else [add_name]
        for idx, block in enumerate(add_blocks):
            values = block.get("values")
            if not isinstance(values, list) or not values:
                continue
            values_text = ", ".join(str(v) for v in values)
            name = add_names[idx] if idx < len(add_names) else None
            if isinstance(name, str) and name.strip():
                parts.append(f"Add series \"{name.strip()}\" : [{values_text}]")
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


def extract_years(text: str) -> list[str]:
    return re.findall(r"\b\d{4}\b", text or "")


def extract_braced(text: str) -> list[str]:
    return [m.group(1).strip() for m in re.finditer(r"\{([^{}]+)\}", text or "")]


def extract_numbers(text: str) -> list[float]:
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text or "")
    out: list[float] = []
    for n in numbers:
        try:
            out.append(float(n))
        except ValueError:
            continue
    return out


def is_match(output_text: str, expected: Any) -> bool:
    answer_text = (output_text or "").strip()
    if expected is None:
        return False

    if isinstance(expected, (int, float)) and answer_text:
        nums = extract_numbers(answer_text)
        if not nums:
            return False
        return abs(nums[0] - float(expected)) < 1e-6

    expected_str = str(expected).strip()
    if not expected_str:
        return False
    expected_years = extract_years(expected_str)
    if expected_years:
        answer_years = extract_years(answer_text)
        if not answer_years:
            return False
        return answer_years[0] == expected_years[0]
    if expected_str in answer_text:
        return True
    expected_parts = extract_braced(expected_str)
    if expected_parts:
        return any(part in answer_text for part in expected_parts)

    nums_expected = extract_numbers(expected_str)
    nums_answer = extract_numbers(answer_text)
    if len(nums_expected) == 1 and nums_answer:
        return abs(nums_expected[0] - nums_answer[0]) < 1e-6
    return False


def get_answer_text(run_output: dict[str, Any]) -> str:
    answer = run_output.get("answer")
    if isinstance(answer, dict):
        v = answer.get("answer")
        return str(v or "").strip()
    if isinstance(answer, str):
        return answer.strip()
    return ""


def find_case_dirs(task_dir: Path) -> list[Path]:
    case_dirs: list[Path] = []
    for p in sorted(task_dir.rglob("*")):
        if not p.is_dir():
            continue
        if not p.name.isdigit():
            continue
        json_path = p / f"{p.name}.json"
        svg_path = p / f"{p.name}.svg"
        if json_path.exists() and svg_path.exists():
            case_dirs.append(p)
    return case_dirs


def main() -> None:
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = PROJECT_ROOT / dataset_root

    task_dir = dataset_root / args.task
    if not task_dir.exists():
        raise SystemExit(f"Task path not found: {task_dir}")

    case_dirs = find_case_dirs(task_dir)
    if not case_dirs:
        raise SystemExit(f"No valid case directories found under: {task_dir}")

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_path.with_suffix(out_path.suffix + ".jsonl")

    tested = 0
    matched = 0
    errors = 0

    with out_path.open("w", encoding="utf-8") as out_file:
        jsonl_file = jsonl_path.open("w", encoding="utf-8") if args.write_jsonl else None
        try:
            for case_dir in case_dirs:
                case_id = case_dir.name
                json_path = case_dir / f"{case_id}.json"
                svg_path = case_dir / f"{case_id}.svg"

                payload = json.loads(json_path.read_text(encoding="utf-8"))
                qa_question, expected = choose_qa(payload, args.qa_index)
                update_question = (
                    qa_question if args.question_only else (synthesize_instruction(payload) or qa_question)
                )

                inputs = {
                    "question": update_question,
                    "update_question": update_question,
                    "qa_question": qa_question,
                    "max_render_retries": args.max_render_retries,
                    "svg_path": str(svg_path),
                    "text_spec": None,
                    "image_path": None,
                }

                ok = False
                error_msg = ""
                answer_text = ""
                run_output: dict[str, Any] | None = None
                try:
                    run_output = run_main(inputs)
                    answer_text = get_answer_text(run_output)
                    ok = is_match(answer_text, expected)
                except Exception as exc:
                    errors += 1
                    error_msg = f"{type(exc).__name__}: {exc}"

                if ok:
                    matched += 1
                tested += 1

                line1 = f"[{case_id}] Q: {qa_question}"
                line2 = f"Update question: {update_question}"
                line3 = f"Expected: {expected}"
                line4 = f"Answer: {answer_text}" if not error_msg else f"Error: {error_msg}"
                line5 = f"Match: {'yes' if ok else 'no'}"
                sep = "-" * 60
                print(line1)
                print(line2)
                print(line3)
                print(line4)
                print(line5)
                print(sep)
                out_file.write(line1 + "\n")
                out_file.write(line2 + "\n")
                out_file.write(line3 + "\n")
                out_file.write(line4 + "\n")
                out_file.write(line5 + "\n")
                out_file.write(sep + "\n")

                if jsonl_file is not None:
                    record = {
                        "task": args.task,
                        "case": case_id,
                        "json_path": str(json_path),
                        "svg_path": str(svg_path),
                        "qa_question": qa_question,
                        "update_question": update_question,
                        "expected": expected,
                        "answer_text": answer_text,
                        "match": ok,
                        "error": error_msg,
                        "run_output": run_output,
                    }
                    jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")

                if args.limit and tested >= args.limit:
                    break

            acc = (matched / tested * 100.0) if tested else 0.0
            summary = f"Matched {matched}/{tested}"
            acc_line = f"Accuracy: {acc:.2f}%"
            err_line = f"Errors: {errors}"
            print(summary)
            print(acc_line)
            print(err_line)
            out_file.write(summary + "\n")
            out_file.write(acc_line + "\n")
            out_file.write(err_line + "\n")
        finally:
            if jsonl_file is not None:
                jsonl_file.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception:
        traceback.print_exc()
        raise
