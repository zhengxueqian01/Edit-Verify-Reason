from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
from typing import Any

PROJECT_SRC = os.path.abspath(os.path.dirname(__file__))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

from chart_agent.config import get_task_model_config
from chart_agent.core.perception_graph import run_perception
from chart_agent.core.trace import append_trace
from chart_agent.core.answerer import answer_question
from chart_agent.core.vision_tool_phase import run_visual_tool_phase
from chart_agent.core.clusterer import run_dbscan, svg_points_to_data
from chart_agent.llm_factory import make_llm
from chart_agent.perception.scatter_svg_updater import update_scatter_svg
from chart_agent.perception.area_svg_updater import update_area_svg
from chart_agent.perception.line_svg_updater import update_line_svg
from chart_agent.perception.render_validator import validate_render
from chart_agent.perception.svg_renderer import default_output_paths
from chart_agent.perception.svg_perceiver import perceive_svg
from chart_agent.perception import area_svg_updater as area_ops


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chart agent perception CLI")
    parser.add_argument("--question", required=True, help="User question")
    parser.add_argument("--text_spec", default="", help="Text specification")
    parser.add_argument("--image", dest="image_path", default="", help="Path to image")
    parser.add_argument("--svg", dest="svg_path", default="", help="Path to SVG")
    return parser.parse_args()


SUPPORTED_SVG_CHART_TYPES = {"scatter", "line", "area"}
TOOL_AUG_CONFIDENCE_THRESHOLD = 0.7


def run_main(inputs: dict[str, Any]) -> dict[str, Any]:
    # Disable LangSmith tracing in this project runtime unless the user re-enables it explicitly.
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    splitter_llm = make_llm(get_task_model_config("splitter"))
    planner_llm = make_llm(get_task_model_config("planner"))
    executor_llm = make_llm(get_task_model_config("executor"))
    answer_llm = make_llm(get_task_model_config("answer"))
    tool_planner_llm = make_llm(get_task_model_config("tool_planner"))
    update_question, qa_question, split_info = _resolve_questions(inputs, splitter_llm)
    structured_context = _normalize_structured_context(inputs.get("structured_update_context"))
    update_question = _enrich_update_question(
        update_question=update_question,
        structured_context=structured_context,
    )
    original_combined_question = _build_combined_qa_question(
        update_question=update_question,
        qa_question=qa_question,
    )
    original_image_path = _resolve_original_image_path(inputs)
    original_chart_type = str(inputs.get("chart_type_hint") or "unknown").strip().lower() or "unknown"
    answer_original_input = {
        "question": original_combined_question,
        "chart_type": original_chart_type,
        "output_image_path": original_image_path,
        "data_summary": {},
    }
    answer_original = answer_question(
        qa_question=original_combined_question,
        chart_type=original_chart_type,
        data_summary={},
        output_image_path=original_image_path,
        llm=answer_llm,
    )
    if not inputs.get("svg_path"):
        raise ValueError("This entry only supports SVG-based updates for line/area/scatter.")

    max_render_retries = int(inputs.get("max_render_retries", 2))
    retries = max(0, max_render_retries)
    attempt_logs: list[dict[str, Any]] = []
    last_state = None
    last_output_image = None
    last_operation_plan: dict[str, Any] | None = None
    last_render_check: dict[str, Any] = {"ok": False, "issues": ["not_run"], "confidence": 0.0}
    retry_hint = ""
    planned_question = update_question
    used_scatter_points: list[dict[str, float]] = []
    last_perception_steps: list[dict[str, Any]] = []

    for attempt in range(1, retries + 2):
        perception_inputs = dict(inputs)
        perception_inputs["question"] = planned_question
        if inputs.get("chart_type_hint"):
            perception_inputs["chart_type_hint"] = inputs.get("chart_type_hint")
        state = run_perception(perception_inputs)
        last_state = state
        chart_type = _resolve_supported_chart_type(state.perception)
        state.perception["chart_type"] = chart_type
        mapping_info = state.perception.get("mapping_info", {})
        if chart_type not in SUPPORTED_SVG_CHART_TYPES:
            raise ValueError(
                f"Unsupported chart_type '{chart_type}'. Only {sorted(SUPPORTED_SVG_CHART_TYPES)} are supported."
            )

        operation_plan = _llm_plan_update(planned_question, str(chart_type), planner_llm, retry_hint=retry_hint)
        last_operation_plan = operation_plan
        normalized_question = str(operation_plan.get("normalized_question") or planned_question)
        planned_question = _choose_planned_question(planned_question, normalized_question)
        plan_steps = _operation_steps_from_plan(operation_plan, planned_question, structured_context)
        has_executable_plan = bool(plan_steps)
        if not operation_plan.get("llm_success") and not has_executable_plan:
            render_check = {
                "ok": False,
                "confidence": 0.0,
                "issues": ["llm_plan_failed"],
            }
            attempt_logs.append(
                {
                    "attempt": attempt,
                    "chart_type": chart_type,
                    "operation_plan": operation_plan,
                    "planned_question": planned_question,
                    "output_image_path": None,
                    "render_check": render_check,
                }
            )
            last_output_image = None
            last_render_check = render_check
            retry_hint = "llm planning failed"
            continue

        output_image = None
        step_logs: list[dict[str, Any]] = []
        perception_steps: list[dict[str, Any]] = []
        try:
            output_image, step_logs, perception_steps, last_state, used_scatter_points = _execute_planned_steps(
                inputs=inputs,
                planned_question=planned_question,
                operation_plan=operation_plan,
                structured_context=structured_context,
                chart_type=chart_type,
                llm=executor_llm,
                used_scatter_points=used_scatter_points,
            )
        except Exception as exc:
            render_check = {
                "ok": False,
                "confidence": 0.0,
                "issues": [f"operation_step_failed: {exc}"],
            }
            attempt_logs.append(
                {
                    "attempt": attempt,
                    "chart_type": chart_type,
                    "operation_plan": operation_plan,
                    "planned_question": planned_question,
                    "output_image_path": None,
                    "step_logs": step_logs,
                    "render_check": render_check,
                }
            )
            last_output_image = None
            last_render_check = render_check
            retry_hint = str(exc)
            continue

        render_check = _validate_render_with_programmatic(
            output_image=output_image,
            chart_type=chart_type,
            update_spec=(last_state.perception if last_state else state.perception).get("update_spec", {}),
            step_logs=step_logs,
            llm=executor_llm,
        )
        append_trace(
            state.trace,
            node="render_check",
            action="RENDER_VALIDATE",
            rationale="Validate rendered image output.",
            inputs_summary={"output_image_path": output_image, "attempt": attempt},
            outputs_summary={"ok": render_check.get("ok")},
            error=None,
        )

        attempt_logs.append(
            {
                "attempt": attempt,
                "chart_type": chart_type,
                "operation_plan": operation_plan,
                "planned_question": planned_question,
                "output_image_path": output_image,
                "step_logs": step_logs,
                "render_check": render_check,
            }
        )
        last_output_image = output_image
        last_render_check = render_check
        last_perception_steps = perception_steps
        if _is_render_check_passed(render_check):
            break
        retry_hint = "; ".join(render_check.get("issues", [])[:4]) or "render validation failed"

    if last_state is None:
        raise RuntimeError("No perception state produced.")

    chart_type = last_state.perception.get("chart_type", "unknown")
    output: dict[str, Any] = {
        "trace": [record.__dict__ for record in last_state.trace],
        "output_image_path": last_output_image,
        "operation_plan": last_operation_plan,
        "render_check": last_render_check,
        "attempt_logs": attempt_logs,
        "qa_question": qa_question,
        "update_question": update_question,
        "question_split": split_info,
        "perception_steps": last_perception_steps,
        "answer_original_input": answer_original_input,
        "answer_original": answer_original,
    }

    if not _is_render_check_passed(last_render_check):
        output["answer"] = {
            "answer": "Render validation failed after retries; QA skipped.",
            "confidence": 0.0,
            "issues": ["render_validation_failed"] + list(last_render_check.get("issues", [])),
        }
        return output

    cluster_result = None
    if chart_type == "scatter" and last_output_image:
        mapping_info = last_state.perception.get("mapping_info", {})
        svg_points = mapping_info.get("existing_points_svg", [])
        x_ticks = mapping_info.get("x_ticks", [])
        y_ticks = mapping_info.get("y_ticks", [])
        existing_data = svg_points_to_data(svg_points, x_ticks, y_ticks)
        new_data = [(p["x"], p["y"]) for p in used_scatter_points]
        all_points = existing_data + new_data
        if all_points:
            cluster_result = run_dbscan(all_points, qa_question)
            output["cluster_result"] = _sanitize_cluster_result(cluster_result)

    answer_data_summary: dict[str, Any] = {
        "update_spec": last_state.perception.get("update_spec", {}),
        "cluster_result": cluster_result,
        "mapping_info_summary": {
            "num_points": last_state.perception.get("primitives_summary", {}).get("num_points"),
            "num_bars": last_state.perception.get("primitives_summary", {}).get("num_bars"),
            "num_areas": last_state.perception.get("primitives_summary", {}).get("num_areas"),
            "num_lines": last_state.perception.get("primitives_summary", {}).get("num_lines"),
        },
        "operation_plan": last_operation_plan,
        "perception_steps": last_perception_steps,
        "latest_step_logs": (attempt_logs[-1].get("step_logs", []) if attempt_logs else []),
    }
    output["answer_input"] = {
        "question": qa_question,
        "chart_type": chart_type,
        "output_image_path": last_output_image,
        "data_summary": answer_data_summary,
    }

    final_eval_image = str(inputs.get("answer_image_path") or "").strip() or last_output_image

    initial_answer = answer_question(
        qa_question=qa_question,
        chart_type=chart_type,
        data_summary=answer_data_summary,
        output_image_path=final_eval_image,
        llm=answer_llm,
    )
    output["answer_initial"] = initial_answer
    output["answer"] = initial_answer

    final_step_svg_path = ""
    latest_step_logs = answer_data_summary.get("latest_step_logs", [])
    if isinstance(latest_step_logs, list):
        for step in reversed(latest_step_logs):
            if not isinstance(step, dict):
                continue
            p = str(step.get("output_svg_path") or "").strip()
            if p:
                final_step_svg_path = p
                break

    initial_confidence_raw = initial_answer.get("confidence", 0.0) if isinstance(initial_answer, dict) else 0.0
    try:
        initial_confidence = float(initial_confidence_raw)
    except (TypeError, ValueError):
        initial_confidence = 0.0

    if initial_confidence < TOOL_AUG_CONFIDENCE_THRESHOLD:
        tool_phase = run_visual_tool_phase(
            question=qa_question,
            chart_type=chart_type,
            data_summary=answer_data_summary,
            image_path=last_output_image,
            svg_path=final_step_svg_path or None,
            llm=tool_planner_llm,
        )
    else:
        tool_phase = {
            "ok": False,
            "reason": "skipped_high_confidence",
            "confidence_threshold": TOOL_AUG_CONFIDENCE_THRESHOLD,
            "initial_confidence": initial_confidence,
            "tool_calls": [],
            "augmented_svg_path": None,
            "augmented_image_path": last_output_image,
        }
    output["tool_phase"] = tool_phase

    augmented_path = str(tool_phase.get("augmented_image_path") or "").strip()
    if tool_phase.get("ok") and augmented_path:
        output["answer_input_tool_augmented"] = {
            "question": qa_question,
            "chart_type": chart_type,
            "output_image_path": augmented_path,
            "data_summary": answer_data_summary,
        }
        output["answer_tool_augmented"] = answer_question(
            qa_question=qa_question,
            chart_type=chart_type,
            data_summary=answer_data_summary,
            output_image_path=augmented_path,
            llm=answer_llm,
        )
        output["answer"] = output["answer_tool_augmented"]
    else:
        output["answer_tool_augmented"] = None
    return output


def _resolve_questions(inputs: dict[str, Any], splitter_llm: Any) -> tuple[str, str, dict[str, Any]]:
    raw_question = str(inputs.get("question") or "").strip()
    raw_update = str(inputs.get("update_question") or "").strip()
    raw_qa = str(inputs.get("qa_question") or "").strip()
    auto_split = bool(inputs.get("auto_split_question", True))

    split_info: dict[str, Any] = {
        "used": False,
        "reason": "not_needed",
        "source_question": raw_question,
    }

    if not raw_question and not raw_update and not raw_qa:
        raise ValueError("At least one of question/update_question/qa_question is required.")

    update_question = raw_update or raw_question
    qa_question = raw_qa or raw_question
    if not auto_split:
        split_info["reason"] = "disabled"
        return update_question, qa_question, split_info

    # Prefer model-based split whenever a full question is available.
    split_source = raw_question or f"{update_question}. {qa_question}".strip()
    if not split_source:
        split_info["reason"] = "empty_split_source"
        return update_question, qa_question, split_info

    split_payload = _llm_split_update_and_qa(split_source, splitter_llm)
    cand_update = str(split_payload.get("update_question") or "").strip()
    cand_qa = str(split_payload.get("qa_question") or "").strip()
    llm_success = bool(split_payload.get("llm_success"))
    if not (cand_update and cand_qa):
        heuristic_update, heuristic_qa = _heuristic_split_update_and_qa(split_source)
        cand_update = cand_update or heuristic_update
        cand_qa = cand_qa or heuristic_qa
    split_info = {
        "used": llm_success and bool(cand_update or cand_qa),
        "reason": "llm_split",
        "source_question": split_source,
        "llm_success": llm_success,
    }
    if cand_update:
        update_question = cand_update
    if cand_qa:
        qa_question = cand_qa
    if not llm_success:
        split_info["reason"] = "llm_split_failed_fallback"
    return update_question, qa_question, split_info


def _heuristic_split_update_and_qa(question: str) -> tuple[str, str]:
    text = (question or "").strip()
    if not text:
        return "", ""
    match = re.match(r"^(?:after|once|when)\s+(.+?),\s*(.+)$", text, flags=re.IGNORECASE)
    if not match:
        return "", ""
    raw_update = match.group(1).strip()
    qa_question = match.group(2).strip()
    update_question = _normalize_gerund_clause(raw_update)
    return update_question, qa_question


def _normalize_gerund_clause(text: str) -> str:
    cleaned = (text or "").strip().rstrip(".")
    replacements = {
        "adding ": "Add ",
        "deleting ": "Delete ",
        "removing ": "Remove ",
        "changing ": "Change ",
        "updating ": "Update ",
        "applying ": "Apply ",
    }
    lowered = cleaned.lower()
    for prefix, replacement in replacements.items():
        if lowered.startswith(prefix):
            cleaned = replacement + cleaned[len(prefix):]
            break
    return cleaned if cleaned.endswith("?") else f"{cleaned}."


def _enrich_update_question(
    *,
    update_question: str,
    structured_context: Any,
) -> str:
    parts = [f"Operation: {(update_question or '').strip()}"]
    operation_target_text, data_change_text = _format_structured_update_context(structured_context)
    if operation_target_text:
        parts.append(f"Operation target: {operation_target_text}")
    if data_change_text:
        parts.append(f"Data change: {data_change_text}")
    return "\n".join(part for part in parts if part.strip())


def _build_combined_qa_question(*, update_question: str, qa_question: str) -> str:
    update_text = (update_question or "").strip()
    qa_text = (qa_question or "").strip()
    if update_text and qa_text:
        return f"{update_text}. {qa_text}".strip()
    return update_text or qa_text


def _normalize_structured_context(structured_context: Any) -> dict[str, Any]:
    if not isinstance(structured_context, dict):
        return {}
    out: dict[str, Any] = {}
    chart_type = str(structured_context.get("chart_type") or "").strip().lower()
    operation = str(structured_context.get("operation") or "").strip().lower()
    operation_target = structured_context.get("operation_target")
    data_change = structured_context.get("data_change")
    if chart_type:
        out["chart_type"] = chart_type
    if operation:
        out["operation"] = operation
    out["operation_target"] = operation_target if isinstance(operation_target, dict) else {}
    out["data_change"] = data_change if isinstance(data_change, dict) else {}
    return out


def _format_structured_update_context(structured_context: Any) -> tuple[str, str]:
    if not isinstance(structured_context, dict):
        return "", ""
    operation_target = structured_context.get("operation_target")
    data_change = structured_context.get("data_change")
    operation_target_text = ""
    data_change_text = ""
    if operation_target:
        try:
            operation_target_text = json.dumps(operation_target, ensure_ascii=False)
        except TypeError:
            operation_target_text = str(operation_target)
    if data_change:
        try:
            data_change_text = json.dumps(data_change, ensure_ascii=False)
        except TypeError:
            data_change_text = str(data_change)
    return operation_target_text, data_change_text


def _resolve_original_image_path(inputs: dict[str, Any]) -> str | None:
    image_path = str(inputs.get("image_path") or "").strip()
    if image_path and Path(image_path).suffix.lower() == ".png" and Path(image_path).exists():
        return image_path
    return None


def main() -> None:
    args = _parse_args()
    inputs = {
        "question": args.question,
        "text_spec": args.text_spec or None,
        "image_path": args.image_path or None,
        "svg_path": args.svg_path or None,
    }
    output = run_main(inputs)
    print(json.dumps(output, indent=2, ensure_ascii=False))


def _sanitize_perception(perception: dict[str, object]) -> dict[str, object]:
    sanitized = dict(perception)
    mapping_info = perception.get("mapping_info")
    if isinstance(mapping_info, dict):
        scrubbed = dict(mapping_info)
        for key in ("existing_points_svg", "bars", "area_top_boundary", "area_fills"):
            scrubbed.pop(key, None)
        sanitized["mapping_info"] = scrubbed
    update_spec = perception.get("update_spec")
    if isinstance(update_spec, dict) and "new_points" in update_spec:
        update_copy = dict(update_spec)
        update_copy["new_points"] = []
        sanitized["update_spec"] = update_copy
    return sanitized


def _sanitize_cluster_result(cluster_result: dict[str, object] | None) -> dict[str, object] | None:
    if not cluster_result:
        return cluster_result
    sanitized = dict(cluster_result)
    sanitized.pop("labels", None)
    return sanitized


def _llm_plan_update(question: str, chart_type: str, llm: Any, retry_hint: str = "") -> dict[str, Any]:
    prompt = (
        "You are planning chart-edit operations.\n"
        f"Chart type: {chart_type}\n"
        "Return JSON only with keys:\n"
        "- operation: one of add|delete|change|unknown\n"
        "- normalized_question: concise imperative update instruction in English\n"
        "- steps: array of step objects in execution order; each step has operation and optional question_hint fields\n"
        "- new_points: list of {x:number,y:number} (required for scatter add, else empty list)\n"
        f"- retry_hint: {retry_hint or 'none'}\n"
        "Rules:\n"
        "- Do not rewrite or summarize structured data payloads.\n"
        "- question_hint is only a short execution hint, not the source of truth for values.\n"
        "Question:\n"
        f"{question}"
    )
    try:
        response = llm.invoke(prompt)
    except Exception:
        return _heuristic_plan_update(question)
    content = getattr(response, "content", "")
    payload = _safe_json_loads(content)
    if not payload:
        return _heuristic_plan_update(question)

    op = str(payload.get("operation", "unknown")).lower()
    if op not in {"add", "delete", "change", "unknown"}:
        op = "unknown"
    normalized = str(payload.get("normalized_question") or question).strip() or question
    points = _coerce_points(payload.get("new_points", []))
    steps = _coerce_steps(payload.get("steps", []))
    return {
        "operation": op,
        "normalized_question": normalized,
        "steps": steps,
        "new_points": points,
        "llm_success": True,
        "llm_raw": content,
    }


def _heuristic_plan_update(question: str) -> dict[str, Any]:
    normalized = (question or "").strip()
    op = "unknown"
    lowered = normalized.lower()
    if re.search(r"\b(delete|remove|drop)\b", lowered):
        op = "delete"
    elif re.search(r"\b(change|update|modify|set)\b", lowered):
        op = "change"
    elif re.search(r"\b(add|append|insert)\b", lowered):
        op = "add"
    return {
        "operation": op,
        "normalized_question": normalized,
        "steps": [{"operation": op, "question": normalized, "new_points": []}] if normalized else [],
        "new_points": [],
        "llm_success": False,
    }


def _llm_split_update_and_qa(question: str, llm: Any) -> dict[str, Any]:
    prompt = (
        "You split mixed chart requests into update instruction and QA question.\n"
        "Return JSON only with keys:\n"
        "- update_question: the chart edit command only (imperative)\n"
        "- qa_question: only the pure QA query to answer after chart is updated\n"
        "- llm_success: true\n"
        "Rules:\n"
        "- Output English only.\n"
        "- qa_question must NOT contain update preconditions like 'after deleting ...'.\n"
        "- Remove leading operation clauses from qa_question, keep only the actual question part.\n"
        "- If one part is missing, return empty string for that key.\n"
        "- Do not add explanations.\n"
        "User input:\n"
        f"{question}"
    )
    try:
        response = llm.invoke(prompt)
    except Exception:
        return {"update_question": "", "qa_question": "", "llm_success": False}
    content = getattr(response, "content", "")
    payload = _safe_json_loads(content)
    if not payload:
        return {"update_question": "", "qa_question": "", "llm_success": False}
    update_q = str(payload.get("update_question") or "").strip()
    qa_q = str(payload.get("qa_question") or "").strip()
    return {
        "update_question": update_q or question,
        "qa_question": qa_q or question,
        "llm_success": True,
        "llm_raw": content,
    }


def _choose_planned_question(original: str, normalized: str) -> str:
    o = (original or "").strip()
    n = (normalized or "").strip()
    if not n:
        return o
    # Preserve explicit numeric series payload for add operations.
    if "[" in o and "]" in o and not ("[" in n and "]" in n):
        return o
    # Preserve multi-op intent when normalization collapses operations.
    if _count_ops(o) > _count_ops(n):
        return o
    return n


def _count_ops(text: str) -> int:
    if not text:
        return 0
    patterns = [
        r"(add|append|insert)",
        r"(remove|delete|drop)",
        r"(change|update|modify|set\\s+to)",
    ]
    count = 0
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            count += 1
    return count


def _safe_json_loads(content: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _coerce_points(raw: Any) -> list[dict[str, float]]:
    if not isinstance(raw, list):
        return []
    points: list[dict[str, float]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            points.append({"x": float(item.get("x")), "y": float(item.get("y"))})
        except Exception:
            continue
    return points


def _coerce_steps(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        op = str(item.get("operation") or "unknown").lower()
        if op not in {"add", "delete", "change", "unknown"}:
            op = "unknown"
        question = str(item.get("question") or "").strip()
        question_hint = str(item.get("question_hint") or question).strip()
        points = _coerce_points(item.get("new_points", []))
        if not question_hint and not points:
            continue
        out.append(
            {
                "operation": op,
                "question": question,
                "question_hint": question_hint,
                "new_points": points,
            }
        )
    return out


def _operation_steps_from_plan(
    operation_plan: dict[str, Any],
    planned_question: str,
    structured_context: dict[str, Any],
) -> list[dict[str, Any]]:
    structured_steps = _build_structured_steps(structured_context, operation_plan)
    if structured_steps:
        return structured_steps
    steps = operation_plan.get("steps", [])
    if isinstance(steps, list) and steps:
        return steps
    return [
        {
            "operation": str(operation_plan.get("operation") or "unknown"),
            "question": planned_question,
            "question_hint": planned_question,
            "new_points": operation_plan.get("new_points", []),
        }
    ]


def _build_structured_steps(
    structured_context: dict[str, Any],
    operation_plan: dict[str, Any],
) -> list[dict[str, Any]]:
    if not structured_context:
        return []

    operation = str(structured_context.get("operation") or "").strip().lower()
    operation_target = structured_context.get("operation_target")
    data_change = structured_context.get("data_change")
    if not isinstance(operation_target, dict):
        operation_target = {}
    if not isinstance(data_change, dict):
        data_change = {}

    plan_steps = operation_plan.get("steps", [])
    op_to_hints: dict[str, list[str]] = {}
    if isinstance(plan_steps, list):
        for step in plan_steps:
            if not isinstance(step, dict):
                continue
            step_op = str(step.get("operation") or "").strip().lower()
            if step_op not in {"add", "delete", "change"}:
                continue
            hint = str(step.get("question_hint") or step.get("question") or "").strip()
            if hint:
                op_to_hints.setdefault(step_op, []).append(hint)

    steps: list[dict[str, Any]] = []
    ordered_ops = [part.strip() for part in operation.split("+") if part.strip()] or [operation]
    for op in ordered_ops:
        if op in {"del", "delete"}:
            labels = _structured_delete_labels(operation_target)
            if labels:
                for idx, label in enumerate(labels):
                    steps.append(
                        {
                            "operation": "delete",
                            "operation_target": {"category_name": label},
                            "data_change": {},
                            "question_hint": _pick_hint(op_to_hints, "delete", idx),
                            "new_points": [],
                        }
                    )
                continue
        if op == "add":
            add_target, add_change = _structured_add_payload(operation_target, data_change)
            if add_target or add_change:
                steps.append(
                    {
                        "operation": "add",
                        "operation_target": add_target,
                        "data_change": add_change,
                        "question_hint": _pick_hint(op_to_hints, "add", len(steps)),
                        "new_points": _points_from_data_change(add_change),
                    }
                )
                continue
        if op == "change":
            change_steps = _structured_change_steps(operation_target, data_change, op_to_hints.get("change", []))
            if change_steps:
                steps.extend(change_steps)
                continue
    return steps


def _pick_hint(op_to_hints: dict[str, list[str]], op: str, idx: int) -> str:
    hints = op_to_hints.get(op, [])
    if not hints:
        return ""
    if idx < len(hints):
        return hints[idx]
    return hints[-1]


def _structured_delete_labels(operation_target: dict[str, Any]) -> list[str]:
    candidates = (
        operation_target.get("category_name"),
        operation_target.get("category_names"),
    )
    labels: list[str] = []
    for value in candidates:
        if isinstance(value, str) and value.strip():
            labels.append(value.strip())
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    labels.append(item.strip())
    return labels


def _structured_add_payload(
    operation_target: dict[str, Any],
    data_change: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    add_change = data_change.get("add") if isinstance(data_change.get("add"), dict) else data_change
    add_target: dict[str, Any] = {}
    for key in ("add_category", "category_name"):
        value = operation_target.get(key)
        if isinstance(value, str) and value.strip():
            add_target["category_name"] = value.strip()
            break
    return add_target, add_change if isinstance(add_change, dict) else {}


def _structured_change_steps(
    operation_target: dict[str, Any],
    data_change: dict[str, Any],
    hints: list[str],
) -> list[dict[str, Any]]:
    change_root = data_change.get("change") if isinstance(data_change.get("change"), dict) else data_change
    if not isinstance(change_root, dict):
        return []
    changes = change_root.get("changes")
    if not isinstance(changes, list) or not changes:
        return []

    steps: list[dict[str, Any]] = []
    for idx, change in enumerate(changes):
        if not isinstance(change, dict):
            continue
        label = str(change.get("category_name") or "").strip()
        step_target = {"category_name": label} if label else {}
        step_change = {
            "mode": "multi_step",
            "changes": [change],
        }
        steps.append(
            {
                "operation": "change",
                "operation_target": step_target,
                "data_change": step_change,
                "question_hint": hints[idx] if idx < len(hints) else "",
                "new_points": [],
            }
        )
    return steps


def _points_from_data_change(data_change: dict[str, Any]) -> list[dict[str, float]]:
    points = data_change.get("points") if isinstance(data_change, dict) else None
    return _coerce_points(points if isinstance(points, list) else [])


def _render_structured_step_question(step: dict[str, Any]) -> str:
    operation = str(step.get("operation") or "unknown").strip().lower()
    operation_target = step.get("operation_target")
    data_change = step.get("data_change")
    if not isinstance(operation_target, dict):
        operation_target = {}
    if not isinstance(data_change, dict):
        data_change = {}

    if operation == "add":
        label = str(operation_target.get("category_name") or "").strip()
        values = data_change.get("values")
        if label and isinstance(values, list) and values:
            values_text = ", ".join(str(v) for v in values)
            return f'Add the category/series "{label}" with values [{values_text}]'
        if label:
            return f'Add the category/series "{label}"'

    if operation == "delete":
        label = str(operation_target.get("category_name") or "").strip()
        if label:
            return f'Delete the category/series "{label}"'

    if operation == "change":
        clauses: list[str] = []
        changes = data_change.get("changes")
        if isinstance(changes, list):
            for change in changes:
                if not isinstance(change, dict):
                    continue
                label = str(change.get("category_name") or "").strip()
                years = change.get("years")
                values = change.get("values")
                if not isinstance(years, list) or not isinstance(values, list):
                    continue
                for year, value in zip(years, values):
                    clauses.append(f'Change "{label}" in {year} to {value}')
        if clauses:
            return "; ".join(clauses)

    hint = str(step.get("question_hint") or "").strip()
    if hint:
        return hint

    parts = [f"Operation: {operation}"]
    if operation_target:
        parts.append(f"Operation target: {json.dumps(operation_target, ensure_ascii=False)}")
    if data_change:
        parts.append(f"Data change: {json.dumps(data_change, ensure_ascii=False)}")
    return "\n".join(parts)


def _step_paths(svg_path: str, chart_type: str, idx: int, total: int) -> tuple[str | None, str | None]:
    final_svg, final_png = default_output_paths(svg_path, chart_type)
    if idx == total - 1:
        return final_svg, final_png
    stem_svg = Path(final_svg).stem
    stem_png = Path(final_png).stem
    step_svg = str(Path(final_svg).with_name(f"{stem_svg}_step{idx+1}.svg"))
    step_png = str(Path(final_png).with_name(f"{stem_png}_step{idx+1}.png"))
    return step_svg, step_png


def _execute_planned_steps(
    *,
    inputs: dict[str, Any],
    planned_question: str,
    operation_plan: dict[str, Any],
    structured_context: dict[str, Any],
    chart_type: str,
    llm: Any,
    used_scatter_points: list[dict[str, float]],
) -> tuple[str | None, list[dict[str, Any]], list[dict[str, Any]], Any, list[dict[str, float]]]:
    steps = _operation_steps_from_plan(operation_plan, planned_question, structured_context)
    step_logs: list[dict[str, Any]] = []
    perception_steps: list[dict[str, Any]] = []
    current_svg = str(inputs["svg_path"])
    last_state = None
    output_image = None
    scatter_points = list(used_scatter_points)

    for idx, step in enumerate(steps):
        step_q = _render_structured_step_question(step)
        step_inputs = dict(inputs)
        step_inputs["svg_path"] = current_svg
        step_inputs["question"] = step_q
        step_inputs["chart_type_hint"] = chart_type
        state = run_perception(step_inputs)
        last_state = state
        perception_steps.append(
            {
                "index": idx + 1,
                "operation": step.get("operation"),
                "question": step_q,
                "question_hint": step.get("question_hint"),
                "operation_target": step.get("operation_target"),
                "data_change": step.get("data_change"),
                "perception": _sanitize_perception(state.perception),
            }
        )
        mapping_info = state.perception.get("mapping_info", {})
        step_svg, step_png = _step_paths(str(inputs["svg_path"]), chart_type, idx, len(steps))

        if chart_type == "scatter":
            points = _coerce_points(step.get("new_points", []))
            if not points:
                points = _coerce_points(operation_plan.get("new_points", []))
            if not points:
                points = _points_from_data_change(step.get("data_change", {}))
            if not points:
                raise ValueError(f"step {idx+1}: scatter points missing")
            scatter_points = points
            output_image = update_scatter_svg(
                current_svg,
                points,
                mapping_info,
                output_path=step_png,
                svg_output_path=step_svg,
                chart_type=chart_type,
                question=step_q,
                llm=llm,
            )
        elif chart_type == "line":
            output_image = update_line_svg(
                current_svg,
                step_q,
                mapping_info,
                output_path=step_png,
                svg_output_path=step_svg,
                llm=llm,
            )
        elif chart_type == "area":
            output_image = update_area_svg(
                current_svg,
                step_q,
                mapping_info,
                output_path=step_png,
                svg_output_path=step_svg,
                llm=llm,
            )
        else:
            raise ValueError(f"unsupported chart type: {chart_type}")

        current_svg = step_svg or current_svg
        step_logs.append(
            {
                "index": idx + 1,
                "operation": step.get("operation"),
                "question": step_q,
                "output_svg_path": step_svg,
                "output_image_path": output_image,
            }
        )

    return output_image, step_logs, perception_steps, last_state, scatter_points


def _resolve_supported_chart_type(perception: dict[str, Any]) -> str:
    chart_type = str(perception.get("chart_type") or "").lower()
    if chart_type in SUPPORTED_SVG_CHART_TYPES:
        return chart_type
    summary = perception.get("primitives_summary", {}) or {}
    try:
        num_areas = int(summary.get("num_areas") or 0)
        num_lines = int(summary.get("num_lines") or 0)
        num_points = int(summary.get("num_points") or 0)
    except Exception:
        num_areas = 0
        num_lines = 0
        num_points = 0
    if num_areas > 0:
        return "area"
    if num_lines > 0:
        return "line"
    if num_points > 0:
        return "scatter"
    return chart_type or "unknown"


def _validate_render_with_programmatic(
    *,
    output_image: str | None,
    chart_type: str,
    update_spec: dict[str, Any],
    step_logs: list[dict[str, Any]],
    llm: Any,
) -> dict[str, Any]:
    basic_check = validate_render(output_image, chart_type, update_spec, llm=None)
    if not basic_check.get("ok"):
        return basic_check

    prog = _programmatic_validate(chart_type=chart_type, step_logs=step_logs)
    if prog is not None:
        if prog.get("ok"):
            return {
                "ok": True,
                "confidence": float(prog.get("confidence", 0.98)),
                "issues": list(prog.get("issues", [])),
                "programmatic": True,
            }
        return {
            "ok": False,
            "confidence": float(prog.get("confidence", 0.1)),
            "issues": list(prog.get("issues", [])),
            "programmatic": True,
        }

    return validate_render(output_image, chart_type, update_spec, llm=llm)


def _programmatic_validate(chart_type: str, step_logs: list[dict[str, Any]]) -> dict[str, Any] | None:
    if chart_type != "area":
        return None
    change_step = None
    for step in reversed(step_logs):
        step_op = str(step.get("operation", "")).lower()
        step_q = str(step.get("question", ""))
        if step_op == "change" or _looks_like_area_change_question(step_q):
            change_step = step
            break
    if not change_step:
        return None
    svg_path = str(change_step.get("output_svg_path") or "").strip()
    question = str(change_step.get("question") or "").strip()
    if not svg_path or not question:
        return None
    if not os.path.exists(svg_path):
        return {"ok": False, "confidence": 0.0, "issues": [f"programmatic: svg not found: {svg_path}"]}

    try:
        with open(svg_path, "r", encoding="utf-8") as handle:
            content = handle.read()
        parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
        root = ET.parse(svg_path, parser=parser).getroot()
        axes = root.find(f'.//{{{area_ops.SVG_NS}}}g[@id="axes_1"]')
        if axes is None:
            return {"ok": False, "confidence": 0.0, "issues": ["programmatic: axes_1 missing"]}

        perceived = perceive_svg(svg_path, question=question, llm=None)
        mapping = perceived.get("mapping_info", {}) if isinstance(perceived, dict) else {}
        x_ticks = mapping.get("x_ticks", []) if isinstance(mapping, dict) else []
        y_ticks = mapping.get("y_ticks", []) if isinstance(mapping, dict) else []
        if len(x_ticks) < 2 or len(y_ticks) < 2:
            return {"ok": False, "confidence": 0.0, "issues": ["programmatic: insufficient axis ticks"]}

        _, legend_items = area_ops._extract_legend_items(root, content)
        labels = [item["label"] for item in legend_items if item.get("label")]
        parsed = area_ops._parse_year_value_update(question, labels, llm=None)
        if parsed is None:
            return {"ok": False, "confidence": 0.0, "issues": ["programmatic: cannot parse change request"]}
        label, year_value, expected_value, _ = parsed

        target_fill = None
        for item in legend_items:
            if item.get("label") == label:
                target_fill = item.get("fill")
                break
        if not target_fill:
            return {"ok": False, "confidence": 0.0, "issues": [f"programmatic: legend label missing: {label}"]}

        areas = area_ops._extract_area_groups(axes)
        target_idx = area_ops._find_area_by_fill(areas, target_fill)
        if target_idx is None:
            return {"ok": False, "confidence": 0.0, "issues": [f"programmatic: area fill missing for {label}"]}

        x_values, series_values = area_ops._area_series_values(areas, y_ticks)
        target_x = area_ops._data_to_pixel(year_value, x_ticks)
        year_idx = min(range(len(x_values)), key=lambda i: abs(x_values[i] - target_x))
        actual_value = float(series_values[target_idx][year_idx])
        tolerance = max(1e-3, abs(expected_value) * 1e-3)
        if abs(actual_value - expected_value) <= tolerance:
            return {
                "ok": True,
                "confidence": 0.99,
                "issues": [],
            }
        return {
            "ok": False,
            "confidence": 0.05,
            "issues": [
                f"programmatic: expected {label}@{int(year_value)}={expected_value}, got {actual_value}",
            ],
        }
    except Exception as exc:
        return {"ok": False, "confidence": 0.0, "issues": [f"programmatic exception: {exc}"]}


def _looks_like_area_change_question(question: str) -> bool:
    if not question:
        return False
    has_year = bool(re.search(r"\b(19|20)\d{2}\b", question))
    if not has_year:
        return False
    return bool(
        re.search(
            r"(increase|decrease|reduce|raise|update|change|set\s+to|to\s+\d|modify)",
            question,
            re.IGNORECASE,
        )
    )


def _is_render_check_passed(render_check: dict[str, Any]) -> bool:
    if not bool(render_check.get("ok")):
        return False
    issues = [str(x) for x in render_check.get("issues", [])]
    hard_fail_markers = {"llm_check_failed", "invalid image size", "image appears empty"}
    return not any(marker in issues for marker in hard_fail_markers)


if __name__ == "__main__":
    main()
