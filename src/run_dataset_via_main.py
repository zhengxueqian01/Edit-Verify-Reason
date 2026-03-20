#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_SRC = os.path.abspath(os.path.dirname(__file__))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume in the existing task record directory and skip cases with result.json.",
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


def build_run_dir_name(input_dir: Path) -> str:
    try:
        rel = input_dir.resolve().relative_to(DATASET_ROOT.resolve())
        parts = list(rel.parts)
    except Exception:
        parts = [input_dir.name]

    if not parts:
        prefix = "dataset_run"
    elif len(parts) == 1:
        first = parts[0].strip() or "dataset"
        category = first.split("-", 1)[0] or first
        task = first.split("-", 1)[1] if "-" in first else first
        prefix = f"{category}_{task}"
    else:
        first = parts[0].strip() or "dataset"
        category = first.split("-", 1)[0] or first
        task = parts[-1].strip() or "run"
        prefix = f"{category}_{task}"
    return re.sub(r"[^A-Za-z0-9_-]+", "_", prefix).strip("_") or "dataset_run"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def choose_qa(payload: dict[str, Any], qa_index: int) -> tuple[str, Any, dict[str, Any]]:
    qa = payload.get("QA")
    if isinstance(qa, list) and qa:
        idx = max(0, min(qa_index, len(qa) - 1))
        item = qa[idx]
        if isinstance(item, dict):
            q = str(item.get("question") or "").strip()
            if q:
                return q, item.get("answer"), item
    q = str(payload.get("question") or "").strip()
    if q:
        return q, payload.get("answer"), {}
    raise ValueError("No question found in JSON.")


def build_structured_update_context(payload: dict[str, Any], qa_item: dict[str, Any] | None = None) -> dict[str, Any]:
    del qa_item
    target = payload.get("operation_target")
    data_change = payload.get("data_change")
    return {
        "operation_target": target if isinstance(target, dict) else {},
        "data_change": data_change if isinstance(data_change, dict) else {},
    }


def build_full_input(payload: dict[str, Any], qa_question: str) -> str:
    question = str(qa_question or "").strip()
    prefix = _synthesize_operation_prefix(payload)
    if prefix and not _question_already_has_update_context(question):
        if question:
            question = f"{prefix}, {question[:1].lower()}{question[1:]}" if len(question) > 1 else f"{prefix}, {question.lower()}"
        else:
            question = prefix
    return _append_structured_tail(
        question,
        operation_target=payload.get("operation_target"),
        data_change=payload.get("data_change"),
    )


def _question_already_has_update_context(question: str) -> bool:
    text = str(question or "").strip().lower()
    if not text:
        return False
    if any(token in text for token in ("after ", "before ", "following ", "once ", "when ")):
        return True
    return bool(re.search(r"\b(add|delete|remove|drop|change|update|modify|insert|append)\w*\b", text))


def _synthesize_operation_prefix(payload: dict[str, Any]) -> str:
    operation = str(payload.get("operation") or "").strip().lower()
    operation_target = payload.get("operation_target")
    data_change = payload.get("data_change")
    if not isinstance(operation_target, dict):
        operation_target = {}
    if not isinstance(data_change, dict):
        data_change = {}

    clauses: list[str] = []
    ordered_ops = [part.strip() for part in operation.split("+") if part.strip()] if operation else []
    if not ordered_ops:
        ordered_ops = _infer_ops_from_payload(operation_target, data_change)

    for op in ordered_ops:
        normalized = _normalize_eval_operation_token(op)
        if normalized == "add":
            label = _first_non_empty_string(operation_target.get("add_category"), operation_target.get("category_name"))
            if label:
                clauses.append(f'adding the category "{label}"')
            else:
                clauses.append("adding the requested category")
        elif normalized == "delete":
            label = _first_non_empty_string(operation_target.get("del_category"), operation_target.get("category_name"))
            if label:
                clauses.append(f'deleting the category "{label}"')
            else:
                clauses.append("deleting the requested category")
        elif normalized == "change":
            label = _first_non_empty_string(operation_target.get("change_category"), operation_target.get("category_name"))
            change_root = data_change.get("change") if isinstance(data_change.get("change"), dict) else data_change
            years = change_root.get("years") if isinstance(change_root, dict) else None
            values = change_root.get("values") if isinstance(change_root, dict) else None
            if label and isinstance(years, list) and years and isinstance(values, list) and values:
                clauses.append(f'changing "{label}" in {years[0]} to {values[0]}')
            elif label:
                clauses.append(f'changing the category "{label}"')
            else:
                clauses.append("applying the requested value changes")

    if not clauses:
        return ""
    if len(clauses) == 1:
        return f"After {clauses[0]}"
    return "After " + ", ".join(clauses[:-1]) + f", and {clauses[-1]}"


def _infer_ops_from_payload(operation_target: dict[str, Any], data_change: dict[str, Any]) -> list[str]:
    ops: list[str] = []
    if any(operation_target.get(key) for key in ("del_category", "del_categories")) or data_change.get("del"):
        ops.append("delete")
    if any(operation_target.get(key) for key in ("add_category",)) or data_change.get("add"):
        ops.append("add")
    if any(operation_target.get(key) for key in ("change_category",)) or data_change.get("change"):
        ops.append("change")
    return ops


def _normalize_eval_operation_token(token: str) -> str:
    lowered = str(token or "").strip().lower()
    if lowered in {"add", "append", "insert"}:
        return "add"
    if lowered in {"del", "delete", "remove", "drop"}:
        return "delete"
    if lowered in {"change", "update", "modify", "set"}:
        return "change"
    return lowered


def _first_non_empty_string(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    return item.strip()
    return ""


def _append_structured_tail(question: str, *, operation_target: Any, data_change: Any) -> str:
    text = str(question or "").strip()
    tail_parts: list[str] = []
    if isinstance(operation_target, dict) and operation_target:
        tail_parts.append(f'"operation_target": {json.dumps(operation_target, ensure_ascii=False)}')
    if isinstance(data_change, dict) and data_change:
        tail_parts.append(f'"data_change": {json.dumps(data_change, ensure_ascii=False)}')
    if not tail_parts:
        return text
    if not text:
        return ",\n".join(tail_parts)
    return text + "\n" + ",\n".join(tail_parts)


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


def extract_answer_payload(result: dict[str, Any], key: str) -> dict[str, Any] | None:
    value = result.get(key)
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return {"answer": value}
    return None


def format_answer_section(
    *,
    title: str,
    payload: dict[str, Any] | None,
    answer_text: str,
    matched: bool | None,
) -> list[str]:
    lines = [f"=== {title} ==="]
    lines.append(f"Answer Text: {answer_text}" if answer_text else "Answer Text: ")
    if matched is not None:
        lines.append(f"Match: {matched}")
    if payload is None:
        lines.append("Model Output: null")
    else:
        lines.append("Model Output:")
        lines.append(json.dumps(_sanitize_report_payload(payload), ensure_ascii=False, indent=2))
    lines.append("")
    return lines


def _sanitize_report_payload(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"labels", "labels_by_color"}:
                continue
            sanitized[key] = _sanitize_report_payload(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_report_payload(item) for item in value]
    return value


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


def build_summary_entry(
    *,
    case_id: str,
    case_dir: Path,
    case_out_dir: Path,
    qa_question: str,
    expected_answer: Any,
    result: dict[str, Any],
    err: str,
) -> dict[str, Any]:
    return {
        "case": case_id,
        "ok": not bool(err),
        "question": qa_question,
        "update_question": str(result.get("update_question") or ""),
        "expected_answer": expected_answer,
        "answer_original": (extract_answer_text_by_key(result, "answer_original") if not err else ""),
        "answer_original_match": (
            is_answer_match(expected_answer, extract_answer_text_by_key(result, "answer_original"))
            if not err
            else False
        ),
        "answer_modified": (extract_answer_text_by_key(result, "answer_initial") if not err else ""),
        "answer_modified_match": (
            is_answer_match(expected_answer, extract_answer_text_by_key(result, "answer_initial"))
            if not err
            else False
        ),
        "answer_tool_augmented": (
            extract_answer_text_by_key(result, "answer_tool_augmented") if not err else ""
        ),
        "answer_tool_augmented_match": (
            (
                is_answer_match(expected_answer, extract_answer_text_by_key(result, "answer_tool_augmented"))
                if extract_answer_payload(result, "answer_tool_augmented") is not None
                else False
            )
            if not err
            else False
        ),
        "answer_final": extract_answer_text(result) if not err else "",
        "answer_final_match": (
            is_answer_match(expected_answer, extract_answer_text(result)) if not err else False
        ),
        "record_dir": str(case_out_dir),
        "case_dir": str(case_dir),
        "error": err,
    }


def main() -> None:
    args = parse_args()
    input_dir = resolve_input_dir(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input folder not found: {input_dir}")

    case_dirs = list_case_dirs(input_dir)
    if not case_dirs:
        raise SystemExit(f"No valid case folders found under: {input_dir}")

    record_root = Path(args.record_root)
    if not record_root.is_absolute():
        record_root = PROJECT_ROOT / record_root
    run_dir = record_root / build_run_dir_name(input_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, Any]] = []
    total = 0
    success = 0
    failed = 0
    answer_modified_evaluated = 0
    answer_modified_matched = 0
    answer_original_evaluated = 0
    answer_original_matched = 0
    answer_tool_aug_evaluated = 0
    answer_tool_aug_matched = 0
    answer_effective_evaluated = 0
    answer_effective_matched = 0

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
        qa_question, expected_answer, qa_item = choose_qa(payload, args.qa_index)
        existing_result_path = case_out_dir / "result.json"
        if args.resume and existing_result_path.exists():
            existing_result = load_json(existing_result_path)
            summary.append(
                build_summary_entry(
                    case_id=case_id,
                    case_dir=case_dir,
                    case_out_dir=case_out_dir,
                    qa_question=qa_question,
                    expected_answer=expected_answer,
                    result=existing_result,
                    err="",
                )
            )
            answer_original_text = extract_answer_text_by_key(existing_result, "answer_original")
            answer_original_match = is_answer_match(expected_answer, answer_original_text)
            answer_modified_text = extract_answer_text_by_key(existing_result, "answer_initial")
            answer_modified_match = is_answer_match(expected_answer, answer_modified_text)
            answer_tool_aug_payload = extract_answer_payload(existing_result, "answer_tool_augmented")
            answer_tool_aug_text = extract_answer_text_by_key(existing_result, "answer_tool_augmented")
            answer_tool_aug_match = (
                is_answer_match(expected_answer, answer_tool_aug_text) if answer_tool_aug_payload else None
            )
            success += 1
            answer_original_evaluated += 1
            if answer_original_match:
                answer_original_matched += 1
            answer_modified_evaluated += 1
            if answer_modified_match:
                answer_modified_matched += 1
            if answer_tool_aug_payload is not None:
                answer_tool_aug_evaluated += 1
                if answer_tool_aug_match:
                    answer_tool_aug_matched += 1
                answer_effective_evaluated += 1
                if answer_tool_aug_match:
                    answer_effective_matched += 1
            else:
                answer_effective_evaluated += 1
                if answer_modified_match:
                    answer_effective_matched += 1
            print(f"[SKIP] {case_id} -> {case_out_dir}")
            continue

        split_update_question = ""
        structured_update_context = build_structured_update_context(payload, qa_item)
        full_input = build_full_input(payload, qa_question)

        result: dict[str, Any]
        err = ""
        try:
            result = run_main(
                {
                    "question": full_input,
                    "max_render_retries": args.max_render_retries,
                    "svg_path": str(svg_path),
                    "image_path": str(image_path) if image_path.exists() else None,
                    "structured_update_context": structured_update_context,
                    "text_spec": None,
                }
            )
            split_update_question = str(result.get("update_question") or split_update_question)
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

            answer_original_text = extract_answer_text_by_key(result, "answer_original")
            answer_original_match = is_answer_match(expected_answer, answer_original_text)
            answer_modified_text = extract_answer_text_by_key(result, "answer_initial")
            answer_modified_match = is_answer_match(expected_answer, answer_modified_text)
            answer_tool_aug_payload = extract_answer_payload(result, "answer_tool_augmented")
            answer_tool_aug_text = extract_answer_text_by_key(result, "answer_tool_augmented")
            answer_tool_aug_match = (
                is_answer_match(expected_answer, answer_tool_aug_text) if answer_tool_aug_payload else None
            )
            answer_original_evaluated += 1
            if answer_original_match:
                answer_original_matched += 1
            answer_modified_evaluated += 1
            if answer_modified_match:
                answer_modified_matched += 1
            if answer_tool_aug_payload is not None:
                answer_tool_aug_evaluated += 1
                if answer_tool_aug_match:
                    answer_tool_aug_matched += 1
                answer_effective_evaluated += 1
                if answer_tool_aug_match:
                    answer_effective_matched += 1
            else:
                answer_effective_evaluated += 1
                if answer_modified_match:
                    answer_effective_matched += 1
            answer_lines = [
                f"Question: {qa_question}",
                f"Update Question: {split_update_question}",
                f"Expected: {expected_answer}",
                "",
            ]
            answer_lines.extend(
                format_answer_section(
                    title="Original Image Answer",
                    payload=extract_answer_payload(result, "answer_original"),
                    answer_text=answer_original_text,
                    matched=answer_original_match,
                )
            )
            answer_lines.extend(
                format_answer_section(
                    title="Modified Image Answer",
                    payload=extract_answer_payload(result, "answer_initial"),
                    answer_text=answer_modified_text,
                    matched=answer_modified_match,
                )
            )
            answer_lines.extend(
                format_answer_section(
                    title="Visual Tool Answer",
                    payload=answer_tool_aug_payload,
                    answer_text=answer_tool_aug_text,
                    matched=answer_tool_aug_match,
                )
            )
            tool_phase = result.get("tool_phase") if isinstance(result.get("tool_phase"), dict) else None
            answer_lines.extend(
                [
                    "=== Tool Phase ===",
                    json.dumps(tool_phase, ensure_ascii=False, indent=2) if tool_phase is not None else "null",
                    "",
                ]
            )
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
            build_summary_entry(
                case_id=case_id,
                case_dir=case_dir,
                case_out_dir=case_out_dir,
                qa_question=qa_question,
                expected_answer=expected_answer,
                result=result,
                err=err,
            )
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
        "answer_modified_evaluated": answer_modified_evaluated,
        "answer_modified_matched": answer_modified_matched,
        "answer_modified_accuracy": (
            answer_modified_matched / answer_modified_evaluated if answer_modified_evaluated else 0.0
        ),
        "answer_tool_aug_evaluated": answer_tool_aug_evaluated,
        "answer_tool_aug_matched": answer_tool_aug_matched,
        "answer_tool_aug_accuracy": (
            answer_tool_aug_matched / answer_tool_aug_evaluated if answer_tool_aug_evaluated else 0.0
        ),
        "answer_effective_evaluated": answer_effective_evaluated,
        "answer_effective_matched": answer_effective_matched,
        "answer_effective_accuracy": (
            answer_effective_matched / answer_effective_evaluated if answer_effective_evaluated else 0.0
        ),
        "items": summary,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_lines = [
        f"Input Dir: {input_dir}",
        f"Run Dir: {run_dir}",
        f"Total: {total}",
        f"Success: {success}",
        f"Failed: {failed}",
        "",
        "Accuracy",
        (
            f"Original Image Answer: {answer_original_matched}/{answer_original_evaluated} "
            f"= {summary_payload['answer_original_accuracy']:.4f}"
        ),
        (
            f"Modified Image Answer: {answer_modified_matched}/{answer_modified_evaluated} "
            f"= {summary_payload['answer_modified_accuracy']:.4f}"
        ),
        (
            f"Visual Tool Answer: {answer_tool_aug_matched}/{answer_tool_aug_evaluated} "
            f"= {summary_payload['answer_tool_aug_accuracy']:.4f}"
        ),
        (
            f"Effective Answer: {answer_effective_matched}/{answer_effective_evaluated} "
            f"= {summary_payload['answer_effective_accuracy']:.4f}"
        ),
        "",
        "Cases",
    ]
    for item in summary:
        summary_lines.append(
            " | ".join(
                [
                    str(item.get("case") or ""),
                    f"ok={item.get('ok')}",
                    f"original={item.get('answer_original_match')}",
                    f"modified={item.get('answer_modified_match')}",
                    f"tool={item.get('answer_tool_augmented_match')}",
                    f"dir={item.get('record_dir')}",
                ]
            )
        )
    (run_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
