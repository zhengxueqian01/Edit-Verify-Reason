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

from chart_agent.config import get_svg_update_mode, resolve_task_model_config
from chart_agent.core.perception_graph import run_perception
from chart_agent.core.trace import append_trace
from chart_agent.core.answerer import answer_question
from chart_agent.core.vision_tool_phase import run_visual_tool_phase
from chart_agent.core.clusterer import run_dbscan, run_dbscan_by_color, svg_points_to_data
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
TOOL_AUG_CONFIDENCE_THRESHOLD = 0.85


def run_main(
    inputs: dict[str, Any],
    *,
    event_callback: Any | None = None,
) -> dict[str, Any]:
    # Disable LangSmith tracing in this project runtime unless the user re-enables it explicitly.
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    model_overrides = _normalize_model_overrides(inputs.get("model_overrides"))
    splitter_config = resolve_task_model_config("splitter", model_overrides)
    planner_config = resolve_task_model_config("planner", model_overrides)
    executor_config = resolve_task_model_config("executor", model_overrides)
    answer_config = resolve_task_model_config("answer", model_overrides)
    tool_planner_config = resolve_task_model_config("tool_planner", model_overrides)
    splitter_llm = make_llm(splitter_config)
    planner_llm = make_llm(planner_config)
    executor_llm = make_llm(executor_config)
    answer_llm = make_llm(answer_config)
    tool_planner_llm = make_llm(tool_planner_config)
    svg_update_mode = get_svg_update_mode(inputs.get("svg_update_mode"))
    svg_perception_mode = inputs.get("svg_perception_mode")
    structured_context = _normalize_structured_context(inputs.get("structured_update_context"))
    update_question, qa_question, split_info, split_data_change = _resolve_questions(inputs, splitter_llm)
    structured_context = _merge_structured_operation_target(
        structured_context,
        split_info.get("operation_target"),
    )
    structured_context = _merge_structured_data_change(structured_context, split_data_change)
    _emit_event(
        event_callback,
        "question_split_done",
        {
            "result": {
                "question_split": split_info,
                "update_question": update_question,
                "qa_question": qa_question,
                "resolved_data_change": structured_context.get("data_change", {}),
                "operation_text": split_info.get("operation_text") or update_question,
            }
        },
    )
    original_image_path = _resolve_original_image_path(inputs)
    answer_original_input = {
        "question": qa_question,
        "output_image_path": original_image_path,
        "data_summary": {},
    }
    answer_original = answer_question(
        qa_question=qa_question,
        data_summary={},
        output_image_path=original_image_path,
        image_context_note="This is the original chart image before any requested update is applied.",
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
        _emit_event(
            event_callback,
            "plan_done",
            {
                "attempt": attempt,
                "chart_type": chart_type,
                "result": {
                    "question_split": split_info,
                    "operation_plan": operation_plan,
                    "update_question": update_question,
                    "qa_question": qa_question,
                    "resolved_data_change": structured_context.get("data_change", {}),
                    "operation_text": split_info.get("operation_text") or update_question,
                },
            },
        )
        normalized_question = str(operation_plan.get("normalized_question") or planned_question)
        planned_question = _choose_planned_question(planned_question, normalized_question)
        operation_plan = _maybe_apply_llm_intent_steps(
            operation_plan=operation_plan,
            operation_text=planned_question,
            chart_type=chart_type,
            perception=state.perception,
            structured_context=structured_context,
            llm=planner_llm,
            update_mode=svg_update_mode,
        )
        last_operation_plan = operation_plan
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
            _emit_event(
                event_callback,
                "render_check_done",
                {
                    "attempt": attempt,
                    "chart_type": chart_type,
                    "result": {
                        "question_split": split_info,
                        "operation_plan": operation_plan,
                        "render_check": render_check,
                        "attempt_logs": attempt_logs,
                        "update_question": update_question,
                        "qa_question": qa_question,
                        "resolved_data_change": structured_context.get("data_change", {}),
                        "operation_text": split_info.get("operation_text") or update_question,
                    },
                },
            )
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
                event_callback=event_callback,
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
            _emit_event(
                event_callback,
                "render_check_done",
                {
                    "attempt": attempt,
                    "chart_type": chart_type,
                    "result": {
                        "question_split": split_info,
                        "operation_plan": operation_plan,
                        "render_check": render_check,
                        "attempt_logs": attempt_logs,
                        "perception_steps": perception_steps,
                        "update_question": update_question,
                        "qa_question": qa_question,
                        "resolved_data_change": structured_context.get("data_change", {}),
                        "operation_text": split_info.get("operation_text") or update_question,
                    },
                },
            )
            retry_hint = str(exc)
            continue

        render_check = _validate_render_with_programmatic(
            output_image=output_image,
            chart_type=chart_type,
            update_spec=(last_state.perception if last_state else state.perception).get("update_spec", {}),
            step_logs=step_logs,
            llm=executor_llm,
            svg_perception_mode=svg_perception_mode,
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
        _emit_event(
            event_callback,
            "render_check_done",
            {
                "attempt": attempt,
                "chart_type": chart_type,
                "result": {
                    "question_split": split_info,
                    "operation_plan": operation_plan,
                    "render_check": render_check,
                    "attempt_logs": attempt_logs,
                    "perception_steps": perception_steps,
                    "output_image_path": output_image,
                    "update_question": update_question,
                    "qa_question": qa_question,
                    "resolved_data_change": structured_context.get("data_change", {}),
                    "operation_text": split_info.get("operation_text") or update_question,
                },
            },
        )
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
        "operation_text": split_info.get("operation_text") or update_question,
        "qa_question": qa_question,
        "update_question": update_question,
        "resolved_data_change": structured_context.get("data_change", {}),
        "question_split": split_info,
        "perception_steps": last_perception_steps,
        "svg_update_mode": svg_update_mode,
        "answer_original_input": answer_original_input,
        "answer_original": answer_original,
        "model_overrides": model_overrides,
        "resolved_task_models": {
            "splitter": splitter_config.model,
            "planner": planner_config.model,
            "executor": executor_config.model,
            "answer": answer_config.model,
            "tool_planner": tool_planner_config.model,
        },
    }

    allow_answer_after_failed_render = _should_answer_after_failed_render(
        chart_type=chart_type,
        structured_context=structured_context,
        output_image_path=last_output_image,
        attempt_logs=attempt_logs,
        max_render_retries=max_render_retries,
    )
    if not _is_render_check_passed(last_render_check) and not allow_answer_after_failed_render:
        output["answer"] = {
            "answer": "Render validation failed after retries; QA skipped.",
            "confidence": 0.0,
            "issues": ["render_validation_failed"] + list(last_render_check.get("issues", [])),
        }
        return output
    if not _is_render_check_passed(last_render_check):
        output["render_validation_warning"] = {
            "issues": ["render_validation_failed"] + list(last_render_check.get("issues", [])),
        }

    cluster_result = None
    cluster_params = _resolve_scatter_cluster_params(structured_context, qa_question)
    if chart_type == "scatter" and last_output_image:
        mapping_info = last_state.perception.get("mapping_info", {})
        svg_points = mapping_info.get("existing_points_svg", [])
        svg_colors = mapping_info.get("existing_point_colors", [])
        x_ticks = mapping_info.get("x_ticks", [])
        y_ticks = mapping_info.get("y_ticks", [])
        existing_data = svg_points_to_data(svg_points, x_ticks, y_ticks)
        points_by_color: dict[str, list[tuple[float, float]]] = {}
        for point, color in zip(existing_data, svg_colors):
            color_key = str(color or "").strip().lower()
            if not color_key:
                continue
            points_by_color.setdefault(color_key, []).append(point)
        for point in used_scatter_points:
            color_key = str(point.get("color") or point.get("point_color") or point.get("fill") or "").strip().lower()
            if not color_key:
                continue
            points_by_color.setdefault(color_key, []).append((float(point["x"]), float(point["y"])))
        if points_by_color:
            cluster_result = run_dbscan_by_color(
                points_by_color,
                qa_question,
                default_eps=float(cluster_params.get("eps") or 6.0),
                default_min_samples=int(cluster_params.get("min_samples") or 3),
            )
        else:
            new_data = [(p["x"], p["y"]) for p in used_scatter_points]
            all_points = existing_data + new_data
            if all_points:
                cluster_result = run_dbscan(
                    all_points,
                    qa_question,
                    default_eps=float(cluster_params.get("eps") or 6.0),
                    default_min_samples=int(cluster_params.get("min_samples") or 3),
                )
        if cluster_result:
            output["cluster_result"] = _sanitize_cluster_result(cluster_result)

    answer_data_summary: dict[str, Any] = {
        "update_spec": last_state.perception.get("update_spec", {}),
        "cluster_result": cluster_result,
        "cluster_params": cluster_params,
            "mapping_info_summary": {
                "num_points": last_state.perception.get("primitives_summary", {}).get("num_points"),
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
        data_summary=answer_data_summary,
        output_image_path=final_eval_image,
        image_context_note=(
            "The requested chart update has already been applied to this image. "
            "Answer the QA question only based on the updated chart."
        ),
        llm=answer_llm,
    )
    output["answer_initial"] = initial_answer
    output["answer"] = initial_answer
    _emit_event(
        event_callback,
        "answer_done",
        {
            "result": {
                **output,
            }
        },
    )

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

    force_tool_phase = _should_force_visual_tool_phase(qa_question, chart_type)
    if initial_confidence < TOOL_AUG_CONFIDENCE_THRESHOLD or force_tool_phase:
        tool_phase = run_visual_tool_phase(
            question=qa_question,
            chart_type=chart_type,
            data_summary=answer_data_summary,
            image_path=last_output_image,
            svg_path=final_step_svg_path or None,
            llm=tool_planner_llm,
            svg_perception_mode=svg_perception_mode,
        )
        if force_tool_phase and isinstance(tool_phase, dict):
            tool_phase["force_run"] = True
    else:
        tool_phase = {
            "ok": False,
            "reason": "skipped_high_confidence",
            "confidence_threshold": TOOL_AUG_CONFIDENCE_THRESHOLD,
            "initial_confidence": initial_confidence,
            "tool_calls": [],
            "augmented_svg_path": None,
            "augmented_image_path": None,
        }
    output["tool_phase"] = tool_phase
    _apply_tool_augmented_answer(
        output=output,
        qa_question=qa_question,
        answer_data_summary=answer_data_summary,
        tool_phase=tool_phase,
        answer_llm=answer_llm,
    )
    _emit_event(
        event_callback,
        "tool_phase_done",
        {
            "result": {
                **output,
            }
        },
    )
    _emit_event(
        event_callback,
        "completed",
        {
            "result": {
                **output,
            }
        },
    )
    return output


def _emit_event(event_callback: Any | None, event_type: str, payload: dict[str, Any]) -> None:
    if event_callback is None:
        return
    event_callback(event_type, payload)


def _resolve_questions(inputs: dict[str, Any], splitter_llm: Any) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
    raw_question = str(inputs.get("question") or "").strip()
    raw_update = str(inputs.get("update_question") or "").strip()
    raw_qa = str(inputs.get("qa_question") or "").strip()
    auto_split = bool(inputs.get("auto_split_question", True))
    inline_split_source, inline_structured_context, inline_structured_suffix = _extract_embedded_structured_context(raw_question)
    inline_structured_context = _normalize_structured_context(inline_structured_context)

    split_info: dict[str, Any] = {
        "used": False,
        "reason": "not_needed",
        "source_question": raw_question,
        "operation_text": "",
        "operation_target": {},
        "data_change": {},
    }

    if not raw_question and not raw_update and not raw_qa:
        raise ValueError("At least one of question/update_question/qa_question is required.")

    update_question = raw_update or raw_question
    qa_question = raw_qa or raw_question
    split_info["operation_text"] = update_question
    if not auto_split:
        split_info["reason"] = "disabled"
        return update_question, qa_question, split_info, {}

    # Prefer model-based split whenever a full question is available.
    split_source = inline_split_source or raw_question or f"{update_question}. {qa_question}".strip()
    if not split_source:
        split_info["reason"] = "empty_split_source"
        return update_question, qa_question, split_info, {}

    split_payload = _llm_split_request(split_source, splitter_llm)
    cand_update = str(split_payload.get("operation_text") or split_payload.get("update_question") or "").strip()
    cand_qa = str(split_payload.get("qa_question") or "").strip()
    cand_operation_target = split_payload.get("operation_target")
    if not isinstance(cand_operation_target, dict):
        cand_operation_target = {}
    cand_data_change = _normalize_data_change(split_payload.get("data_change"))
    llm_success = bool(split_payload.get("llm_success"))
    if not (cand_update and cand_qa):
        rule_payload = _rule_split_request(split_source)
        cand_update = cand_update or str(rule_payload.get("operation_text") or "").strip()
        cand_qa = cand_qa or str(rule_payload.get("qa_question") or "").strip()
        if not cand_operation_target:
            fallback_operation_target = rule_payload.get("operation_target")
            if isinstance(fallback_operation_target, dict):
                cand_operation_target = fallback_operation_target
        if not cand_data_change:
            cand_data_change = _normalize_data_change(rule_payload.get("data_change"))
        heuristic_update, heuristic_qa = _heuristic_split_update_and_qa(split_source)
        cand_update = cand_update or heuristic_update
        cand_qa = cand_qa or heuristic_qa
    cand_operation_target = _merge_dict_like(inline_structured_context.get("operation_target"), cand_operation_target)
    cand_data_change = _merge_dict_like(inline_structured_context.get("data_change"), cand_data_change)
    structured_context_note = ""
    if inline_structured_suffix:
        structured_context_note = _structured_context_suffix_text(
            operation_target=cand_operation_target,
            data_change=cand_data_change,
            raw_suffix=inline_structured_suffix,
        )
    elif llm_success and (cand_operation_target or cand_data_change):
        structured_context_note = _structured_context_suffix_text(
            operation_target=cand_operation_target,
            data_change=cand_data_change,
            raw_suffix="",
        )
    split_info = {
        "used": bool(cand_update or cand_qa),
        "reason": "llm_split" if llm_success else "llm_split_failed_fallback",
        "source_question": split_source,
        "llm_success": llm_success,
        "operation_text": _compose_operation_text(cand_update or update_question, structured_context_note),
        "operation_target": cand_operation_target,
        "data_change": cand_data_change,
    }
    if cand_update:
        update_question = cand_update
    if cand_qa:
        qa_question = cand_qa
    update_question = _compose_operation_text(update_question, structured_context_note)
    return update_question, qa_question, split_info, cand_data_change


def _normalize_model_overrides(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    normalized: dict[str, str] = {}
    for key, value in raw.items():
        task = str(key or "").strip()
        model_name = str(value or "").strip()
        if task and model_name:
            normalized[task] = model_name
    return normalized


def _merge_dict_like(base_value: Any, extra_value: Any) -> dict[str, Any]:
    base = dict(base_value) if isinstance(base_value, dict) else {}
    if isinstance(extra_value, dict):
        for key, value in extra_value.items():
            if key in base and isinstance(base.get(key), dict) and isinstance(value, dict):
                base[key] = _merge_dict_like(base.get(key), value)
                continue
            if key in base and isinstance(base.get(key), dict) and value == {}:
                continue
            base[key] = value
    return base


def _should_force_visual_tool_phase(question: str, chart_type: str) -> bool:
    text = str(question or "").strip().lower()
    chart = str(chart_type or "").strip().lower()
    if not text:
        return False
    if chart == "scatter":
        return any(token in text for token in ("cluster", "clusters", "聚类", "簇"))
    if chart == "line":
        return any(token in text for token in ("intersection", "intersections", "cross", "crossing", "交点", "相交", "穿过"))
    return False


def _heuristic_split_update_and_qa(question: str) -> tuple[str, str]:
    text = (question or "").strip()
    if not text:
        return "", ""
    raw_update, qa_question = _split_preface_and_question(text)
    if not raw_update or not qa_question:
        return "", ""
    update_question = _normalize_gerund_clause(raw_update)
    return update_question, qa_question


def _normalize_data_change(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    try:
        json.dumps(value, ensure_ascii=False)
    except TypeError:
        return {}
    return value


def _merge_structured_data_change(
    structured_context: dict[str, Any],
    split_data_change: dict[str, Any],
) -> dict[str, Any]:
    base = dict(structured_context or {})
    current = base.get("data_change")
    merged = _merge_dict_like(current, _normalize_data_change(split_data_change))
    base["data_change"] = merged
    return base


def _merge_structured_operation_target(
    structured_context: dict[str, Any],
    split_operation_target: Any,
) -> dict[str, Any]:
    base = dict(structured_context or {})
    current = base.get("operation_target")
    merged = dict(current) if isinstance(current, dict) else {}
    if isinstance(split_operation_target, dict):
        for key, value in split_operation_target.items():
            if value is None:
                continue
            merged[key] = value
    base["operation_target"] = merged
    return base


def _extract_embedded_structured_context(text: str) -> tuple[str, dict[str, Any], str]:
    raw = str(text or "").strip()
    if not raw:
        return "", {}, ""
    marker_positions = [
        idx
        for idx in (
            raw.find('"operation_target"'),
            raw.find('"data_change"'),
        )
        if idx >= 0
    ]
    if not marker_positions:
        return raw, {}, ""
    start = min(marker_positions)
    prefix = raw[:start].rstrip(" ,")
    suffix = raw[start:].strip().rstrip(",")
    payload = _parse_embedded_structured_context(suffix)
    return prefix, payload, suffix


def _parse_embedded_structured_context(suffix: str) -> dict[str, Any]:
    text = str(suffix or "").strip()
    if not text:
        return {}
    out: dict[str, Any] = {}
    for key in ("operation_target", "data_change"):
        pattern = f'"{key}"'
        idx = text.find(pattern)
        if idx < 0:
            continue
        brace_start = text.find("{", idx)
        if brace_start < 0:
            continue
        brace_end = _find_matching_brace(text, brace_start)
        if brace_end < 0:
            continue
        block = text[brace_start : brace_end + 1]
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            out[key] = parsed
    return out


def _find_matching_brace(text: str, start: int) -> int:
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return idx
    return -1


def _structured_context_suffix_text(
    *,
    operation_target: dict[str, Any],
    data_change: dict[str, Any],
    raw_suffix: str,
) -> str:
    if raw_suffix:
        return raw_suffix
    parts: list[str] = []
    if operation_target:
        parts.append(f'operation_target: {json.dumps(operation_target, ensure_ascii=False)}')
    if data_change:
        parts.append(f'data_change: {json.dumps(data_change, ensure_ascii=False)}')
    return ", ".join(parts)


def _compose_operation_text(operation_text: str, suffix_text: str) -> str:
    operation = str(operation_text or "").strip()
    suffix = str(suffix_text or "").strip()
    if not suffix:
        return operation
    if not operation:
        return suffix
    joiner = " " if operation.endswith((".", ")", "]", "}")) else ". "
    return f"{operation}{joiner}{suffix}"


def _rule_split_request(question: str) -> dict[str, Any]:
    text = str(question or "").strip()
    if not text:
        return {"operation_text": "", "qa_question": "", "operation_target": {}, "data_change": {}}

    prefix, qa_question = _split_preface_and_question(text)
    if not prefix or not qa_question:
        return {"operation_text": "", "qa_question": "", "operation_target": {}, "data_change": {}}

    operations = _extract_preface_operations(prefix)
    if not operations:
        return {"operation_text": "", "qa_question": "", "operation_target": {}, "data_change": {}}

    operation_lines: list[str] = []
    operation_target: dict[str, Any] = {}
    data_change: dict[str, Any] = {}

    for item in operations:
        op = item.get("operation")
        target = item.get("operation_target")
        change = item.get("data_change")
        sentence = str(item.get("operation_text") or "").strip()
        if sentence:
            operation_lines.append(sentence)
        if isinstance(target, dict):
            operation_target.update(target)
        if op == "add" and isinstance(change, dict):
            data_change["add"] = change
        elif op == "delete" and isinstance(change, dict):
            data_change["del"] = change
        elif op == "change" and isinstance(change, dict):
            data_change["change"] = change

    return {
        "operation_text": " ".join(operation_lines).strip(),
        "qa_question": qa_question,
        "operation_target": operation_target,
        "data_change": data_change,
    }


def _split_preface_and_question(text: str) -> tuple[str, str]:
    stripped = text.strip()
    lowered = stripped.lower()
    if lowered.startswith(("after ", "once ", "when ")):
        boundary_match = re.search(
            r"[,，]\s*[\"'“”]?\s*(in which|which|how|what|when|where|why|who|is|are|does|do|did|was|were|can|could|should|would)\b",
            stripped,
            flags=re.IGNORECASE,
        )
        if boundary_match:
            boundary = boundary_match.start()
            prefix = stripped[:boundary].strip()
            question = stripped[boundary + 1 :].strip()
            prefix = re.sub(r"^(?:after|once|when)\s+", "", prefix, flags=re.IGNORECASE).strip()
            prefix = prefix.rstrip(' \t\n\r"\'“”')
            question = question.lstrip(' \t\n\r"\'“”')
            return prefix, question
    return "", ""


def _extract_preface_operations(prefix: str) -> list[dict[str, Any]]:
    text = str(prefix or "").strip().rstrip(".")
    if not text:
        return []
    clauses = re.split(r"\s+(?:and|then)\s+", text, flags=re.IGNORECASE)
    operations: list[dict[str, Any]] = []
    for clause in clauses:
        parsed = _parse_operation_clause(clause)
        if parsed:
            operations.append(parsed)
    return operations


def _parse_operation_clause(clause: str) -> dict[str, Any] | None:
    text = str(clause or "").strip().rstrip(".")
    if not text:
        return None
    lowered = text.lower()
    if lowered.startswith("adding ") or lowered.startswith("add "):
        return _parse_add_clause(text)
    if lowered.startswith("deleting ") or lowered.startswith("delete ") or lowered.startswith("removing ") or lowered.startswith("remove "):
        return _parse_delete_clause(text)
    if lowered.startswith("changing ") or lowered.startswith("change ") or lowered.startswith("updating ") or lowered.startswith("update "):
        return _parse_change_clause(text)
    return None


def _parse_add_clause(text: str) -> dict[str, Any] | None:
    label = _extract_category_label(text)
    if not label:
        return None
    years, values = _extract_years_and_values(text)
    add_change: dict[str, Any] = {}
    if years and values:
        add_change = {"mode": "full_series", "years": years, "values": values}
    return {
        "operation": "add",
        "operation_text": _normalize_gerund_clause(text),
        "operation_target": {"add_category": label},
        "data_change": add_change,
    }


def _parse_delete_clause(text: str) -> dict[str, Any] | None:
    label = _extract_category_label(text)
    if not label:
        return None
    return {
        "operation": "delete",
        "operation_text": _normalize_gerund_clause(text),
        "operation_target": {"del_category": label},
        "data_change": {"category": label},
    }


def _parse_change_clause(text: str) -> dict[str, Any] | None:
    return {
        "operation": "change",
        "operation_text": _normalize_gerund_clause(text),
        "operation_target": {},
        "data_change": {},
    }


def _extract_category_label(text: str) -> str:
    quoted = re.findall(r'["“”]([^"“”]+)["“”]', text)
    if quoted:
        return quoted[0].strip()
    open_quoted = re.search(r'[“"]([^"“”]+)$', text)
    if open_quoted:
        return open_quoted.group(1).strip()
    match = re.search(
        r"(?:category|series)\s+([A-Za-z0-9][A-Za-z0-9&'()\/,\- ]*[A-Za-z0-9)])$",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    match = re.search(
        r"(?:category|series)\s+([A-Za-z0-9][A-Za-z0-9&'()\/,\- ]*[A-Za-z0-9])",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return ""


def _extract_years_and_values(text: str) -> tuple[list[str], list[float]]:
    year_match = re.search(r"years?\s+(\d{4})\s*[–-]\s*(\d{4})", text, flags=re.IGNORECASE)
    years: list[str] = []
    if year_match:
        start = int(year_match.group(1))
        end = int(year_match.group(2))
        if start <= end and end - start <= 50:
            years = [str(year) for year in range(start, end + 1)]

    values: list[float] = []
    values_match = re.search(r"\(([^()]*)\)", text)
    if values_match:
        values_text = values_match.group(1).strip()
        raw_values = re.split(r"\s*;\s*", values_text) if ";" in values_text else re.split(r"\s*,\s*", values_text)
        for raw in raw_values:
            cleaned = raw.replace(",", "").strip()
            if not cleaned:
                continue
            try:
                values.append(float(cleaned))
            except ValueError:
                return years, []
    if years and values and len(years) != len(values):
        return [], []
    return years, values


def _normalize_gerund_clause(text: str) -> str:
    cleaned = (text or "").strip().rstrip(".")
    cleaned = re.sub(r"\band\s+adding\b", "and add", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\band\s+deleting\b", "and delete", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\band\s+removing\b", "and remove", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\band\s+changing\b", "and change", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\band\s+updating\b", "and update", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\band\s+applying\b", "and apply", cleaned, flags=re.IGNORECASE)
    replacements = (
        ("adding ", "Add "),
        ("deleting ", "Delete "),
        ("removing ", "Remove "),
        ("changing ", "Change "),
        ("updating ", "Update "),
        ("applying ", "Apply "),
    )
    lowered = cleaned.lower()
    for prefix, replacement in replacements:
        if lowered.startswith(prefix):
            cleaned = replacement + cleaned[len(prefix):]
            break
    return cleaned if cleaned.endswith("?") else f"{cleaned}."


def _normalize_structured_context(structured_context: Any) -> dict[str, Any]:
    if not isinstance(structured_context, dict):
        return {}
    out: dict[str, Any] = {}
    chart_type = str(structured_context.get("chart_type") or "").strip().lower()
    task = str(structured_context.get("task") or "").strip().lower()
    operation_target = structured_context.get("operation_target")
    data_change = structured_context.get("data_change")
    cluster_params = structured_context.get("cluster_params")
    if chart_type:
        out["chart_type"] = chart_type
    if task:
        out["task"] = task
    out["operation_target"] = operation_target if isinstance(operation_target, dict) else {}
    out["data_change"] = data_change if isinstance(data_change, dict) else {}
    out["cluster_params"] = cluster_params if isinstance(cluster_params, dict) else {}
    return out


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


def _resolve_scatter_cluster_params(structured_context: dict[str, Any], qa_question: str) -> dict[str, Any]:
    params = structured_context.get("cluster_params")
    if isinstance(params, dict) and (params.get("eps") is not None or params.get("min_samples") is not None):
        return {
            "mode": "per_color",
            "algorithm": "DBSCAN",
            "eps": params.get("eps"),
            "min_samples": params.get("min_samples"),
            "source": str(params.get("source") or "structured_context"),
        }
    if str(structured_context.get("chart_type") or "").strip().lower() == "scatter" and (
        str(structured_context.get("task") or "").strip().lower() == "cluster"
        or any(token in (qa_question or "").lower() for token in ("cluster", "clusters"))
    ):
        eps_match = re.search(r"eps\s*=\s*([\d.]+)", qa_question or "", re.IGNORECASE)
        min_samples_match = re.search(r"min_samples?\s*=\s*(\d+)", qa_question or "", re.IGNORECASE)
        eps = float(eps_match.group(1)) if eps_match else None
        min_samples = int(min_samples_match.group(1)) if min_samples_match else None
        return {
            "mode": "per_color",
            "algorithm": "DBSCAN",
            "eps": eps,
            "min_samples": min_samples,
            "source": "qa_question_suffix",
        }
    return {}


def _should_answer_after_failed_render(
    *,
    chart_type: str,
    structured_context: dict[str, Any],
    output_image_path: str | None,
    attempt_logs: list[dict[str, Any]] | None = None,
    max_render_retries: int = 2,
) -> bool:
    output_path = str(output_image_path or "").strip()
    if chart_type == "scatter" and str(structured_context.get("task") or "").strip().lower() == "cluster":
        return bool(output_path)
    return _allow_answer_after_exhausted_render_validation(
        output_image_path=output_path,
        attempt_logs=attempt_logs,
        max_render_retries=max_render_retries,
    )


def _allow_answer_after_exhausted_render_validation(
    *,
    output_image_path: str,
    attempt_logs: list[dict[str, Any]] | None,
    max_render_retries: int,
) -> bool:
    if not output_image_path or not os.path.exists(output_image_path):
        return False
    if not isinstance(attempt_logs, list) or not attempt_logs:
        return False
    required_attempts = max(1, int(max_render_retries) + 1)
    if len(attempt_logs) < required_attempts:
        return False
    recent_attempts = attempt_logs[-required_attempts:]
    blocked = 0
    for attempt in recent_attempts:
        if not isinstance(attempt, dict):
            return False
        attempt_output = str(attempt.get("output_image_path") or "").strip()
        render_check = attempt.get("render_check")
        if not attempt_output or not isinstance(render_check, dict):
            return False
        if _is_render_check_passed(render_check):
            return False
        blocked += 1
    return blocked == required_attempts


def _llm_plan_update(question: str, chart_type: str, llm: Any, retry_hint: str = "") -> dict[str, Any]:
    new_points_schema = (
        "list of {x:number,y:number,color?:string} (required for scatter add; preserve per-point color when provided)"
        if chart_type == "scatter"
        else "list of {x:number,y:number} (required for scatter add, else empty list)"
    )
    prompt = (
        "You are planning chart-edit operations.\n"
        f"Chart type: {chart_type}\n"
        "Return JSON only with keys:\n"
        "- operation: one of add|delete|change|unknown\n"
        "- normalized_question: concise imperative update instruction in English\n"
        "- steps: array of step objects in execution order; each step has operation and optional question_hint, operation_target, data_change, and new_points fields\n"
        f"- new_points: {new_points_schema}\n"
        f"- retry_hint: {retry_hint or 'none'}\n"
        "Rules:\n"
        "- Do not rewrite or summarize structured data payloads.\n"
        "- question_hint is only a short execution hint, not the source of truth for values.\n"
        "- If the input includes structured operation target or data change, preserve them at step level instead of collapsing them into prose.\n"
        "- For scatter add, if point colors are provided in the question/data payload, copy them through to each new_points item.\n"
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


def _maybe_apply_llm_intent_steps(
    *,
    operation_plan: dict[str, Any],
    operation_text: str,
    chart_type: str,
    perception: dict[str, Any],
    structured_context: dict[str, Any],
    llm: Any,
    update_mode: str,
) -> dict[str, Any]:
    if update_mode != "llm_intent":
        return operation_plan
    intent_steps = _llm_plan_svg_intent(
        operation_text=operation_text,
        chart_type=chart_type,
        perception=perception,
        structured_context=structured_context,
        llm=llm,
    )
    if not intent_steps:
        return operation_plan
    merged = dict(operation_plan)
    merged["steps"] = intent_steps
    merged["llm_intent_success"] = True
    return merged


def _llm_plan_svg_intent(
    *,
    operation_text: str,
    chart_type: str,
    perception: dict[str, Any],
    structured_context: dict[str, Any],
    llm: Any,
) -> list[dict[str, Any]]:
    prompt = (
        "You are planning SVG edit intent for a chart update.\n"
        "Return JSON only with key: steps.\n"
        "- steps: array of step objects in execution order.\n"
        "- each step may contain: operation, question_hint, operation_target, data_change, new_points.\n"
        "- operation must be one of add|delete|change.\n"
        "- Decide the concrete edit target from the operation text and SVG summary.\n"
        "- Keep structured payloads in operation_target/data_change instead of prose when possible.\n"
        "- Do not output explanations.\n"
        f"Chart type: {chart_type}\n"
        f"Operation text: {operation_text}\n"
        f"Structured context: {json.dumps(_intent_structured_context_summary(structured_context), ensure_ascii=False)}\n"
        f"SVG summary: {json.dumps(_intent_perception_summary(perception), ensure_ascii=False)}"
    )
    try:
        response = llm.invoke(prompt)
    except Exception:
        return []
    content = getattr(response, "content", "")
    payload = _safe_json_loads(content)
    if not payload:
        return []
    return _coerce_steps(payload.get("steps", []))


def _intent_structured_context_summary(structured_context: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if not isinstance(structured_context, dict):
        return summary
    operation_target = structured_context.get("operation_target")
    data_change = structured_context.get("data_change")
    if isinstance(operation_target, dict) and operation_target:
        summary["operation_target"] = operation_target
    if isinstance(data_change, dict) and data_change:
        summary["data_change"] = data_change
    if str(structured_context.get("task") or "").strip():
        summary["task"] = structured_context.get("task")
    return summary


def _intent_perception_summary(perception: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(perception, dict):
        return {}
    mapping_info = perception.get("mapping_info")
    if not isinstance(mapping_info, dict):
        mapping_info = {}
    return {
        "chart_type": perception.get("chart_type"),
        "primitives_summary": perception.get("primitives_summary", {}),
        "axes_bounds": mapping_info.get("axes_bounds"),
        "x_ticks_count": len(mapping_info.get("x_ticks", [])) if isinstance(mapping_info.get("x_ticks"), list) else 0,
        "y_ticks_count": len(mapping_info.get("y_ticks", [])) if isinstance(mapping_info.get("y_ticks"), list) else 0,
        "x_labels_sample": _compact_value_list(mapping_info.get("x_labels"), limit=8),
        "point_color_sample": _compact_value_list(mapping_info.get("existing_point_colors"), limit=8),
        "area_fill_sample": _compact_value_list(mapping_info.get("area_fills"), limit=8),
    }


def _compact_value_list(values: Any, *, limit: int) -> list[Any]:
    if not isinstance(values, list):
        return []
    if len(values) <= limit:
        return values
    head = max(1, limit // 2)
    tail = max(1, limit - head)
    return values[:head] + ["..."] + values[-tail:]


def _llm_split_request(question: str, llm: Any) -> dict[str, Any]:
    prompt = (
        "You split chart requests into operation text, QA question, and structured data change.\n"
        "Return JSON only with keys:\n"
        "- operation_text: the chart edit command only (imperative, explicit, step-by-step)\n"
        "- qa_question: only the pure QA query to answer after chart is updated\n"
        "- operation_target: structured JSON object for concrete operation targets such as add_category, del_category, category_name, or category_names; use {} if unavailable\n"
        "- data_change: structured JSON object for concrete payloads such as points, values, years, or per-step changes; use {} if unavailable\n"
        "- llm_success: true\n"
        "Rules:\n"
        "- Output English only.\n"
        "- qa_question must NOT contain update preconditions like 'after deleting ...'.\n"
        "- Remove leading operation clauses from qa_question, keep only the actual question part.\n"
        "- If there are multiple edit actions, rewrite operation_text as an explicit ordered sequence.\n"
        "- Use concise imperative verbs such as 'Delete', 'Add', 'Change', 'Apply'. Never use gerunds like 'adding', 'deleting', 'applying'.\n"
        "- Preferred format for multiple actions: '1. Delete ...; 2. Apply ...; 3. Add ...'.\n"
        "- Extract operation targets into operation_target whenever possible instead of leaving them only in prose.\n"
        "- Move numeric values, points, year-value series, and concrete revision payloads into data_change whenever possible.\n"
        "- If one part is missing, return empty string for that key.\n"
        "- Do not add explanations.\n"
        "Example:\n"
        "{\"operation_text\":\"1. Delete the category CrimsonLink; 2. Apply the listed value revisions.\",\"qa_question\":\"How many times do the lines for Starburst and AetherNet intersect?\",\"operation_target\":{\"del_category\":\"CrimsonLink\"},\"data_change\":{\"change\":{\"changes\":[{\"category_name\":\"Starburst\",\"years\":[2020],\"values\":[12]}]}},\"llm_success\":true}\n"
        "User input:\n"
        f"{question}"
    )
    try:
        response = llm.invoke(prompt)
    except Exception:
        return {"operation_text": "", "qa_question": "", "operation_target": {}, "data_change": {}, "llm_success": False}
    content = getattr(response, "content", "")
    payload = _safe_json_loads(content)
    if not payload:
        return {"operation_text": "", "qa_question": "", "operation_target": {}, "data_change": {}, "llm_success": False}
    llm_success = bool(payload.get("llm_success"))
    if not llm_success:
        return {
            "operation_text": "",
            "qa_question": "",
            "operation_target": {},
            "data_change": {},
            "llm_success": False,
            "llm_raw": content,
        }
    update_q = str(payload.get("operation_text") or payload.get("update_question") or "").strip()
    qa_q = str(payload.get("qa_question") or "").strip()
    operation_target = payload.get("operation_target")
    if not isinstance(operation_target, dict):
        operation_target = {}
    data_change = _normalize_data_change(payload.get("data_change"))
    return {
        "operation_text": update_q or question,
        "update_question": update_q or question,
        "qa_question": qa_q or question,
        "operation_target": operation_target,
        "data_change": data_change,
        "llm_success": True,
        "llm_raw": content,
    }


def _llm_split_update_and_qa(question: str, llm: Any) -> dict[str, Any]:
    return _llm_split_request(question, llm)


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


def _coerce_points(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    points: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            point: dict[str, Any] = {"x": float(item.get("x")), "y": float(item.get("y"))}
        except Exception:
            continue
        for key in ("color", "point_color", "fill"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                point[key] = value.strip()
        points.append(point)
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
        operation_target = item.get("operation_target")
        data_change = _normalize_data_change(item.get("data_change"))
        if not isinstance(operation_target, dict):
            operation_target = {}
        if not question_hint and not points and not operation_target and not data_change:
            continue
        out.append(
            {
                "operation": op,
                "question": question,
                "question_hint": question_hint,
                "operation_target": operation_target,
                "data_change": data_change,
                "new_points": points,
            }
        )
    return out


def _operation_steps_from_plan(
    operation_plan: dict[str, Any],
    planned_question: str,
    structured_context: dict[str, Any],
) -> list[dict[str, Any]]:
    steps = operation_plan.get("steps", [])
    if isinstance(steps, list) and steps:
        enriched_steps = _enrich_llm_steps_with_structured_data(steps, structured_context, planned_question)
        if _llm_steps_cover_structured_ops(enriched_steps, structured_context, planned_question):
            return enriched_steps
    structured_steps = _build_structured_steps(structured_context, operation_plan, planned_question)
    if structured_steps:
        return structured_steps
    if isinstance(steps, list) and steps:
        return _enrich_llm_steps_with_structured_data(steps, structured_context, planned_question)
    return [
        {
            "operation": str(operation_plan.get("operation") or "unknown"),
            "question": planned_question,
            "question_hint": planned_question,
            "new_points": operation_plan.get("new_points", []),
        }
    ]


def _enrich_llm_steps_with_structured_data(
    steps: list[Any],
    structured_context: dict[str, Any],
    operation_text: str,
) -> list[dict[str, Any]]:
    if not isinstance(steps, list):
        return []
    operation_target = structured_context.get("operation_target")
    data_change = structured_context.get("data_change")
    if not isinstance(operation_target, dict):
        operation_target = {}
    if not isinstance(data_change, dict):
        data_change = {}

    add_target, add_change = _structured_add_payload(operation_target, data_change)
    delete_labels = _structured_delete_labels(operation_target)
    change_steps = _structured_change_steps(operation_target, data_change, [])
    fallback_delete_idx = 0
    fallback_change_idx = 0

    enriched: list[dict[str, Any]] = []
    for raw_step in steps:
        if not isinstance(raw_step, dict):
            continue
        step = dict(raw_step)
        op = _normalize_operation_token(str(step.get("operation") or ""))
        step["operation"] = op or str(step.get("operation") or "unknown").strip().lower() or "unknown"

        normalized_target = _normalize_step_operation_target(step.get("operation_target"))
        step_change = _normalize_data_change(step.get("data_change"))
        step["operation_target"] = normalized_target
        step["data_change"] = step_change
        step["question_hint"] = str(step.get("question_hint") or step.get("question") or "").strip()
        step["new_points"] = _coerce_points(step.get("new_points", []))

        if op == "add":
            merged_target = _merge_dict_like(add_target, normalized_target)
            step["operation_target"] = merged_target
            if not step_change and add_change:
                step["data_change"] = dict(add_change)
            elif add_change:
                step["data_change"] = _merge_dict_like(add_change, step_change)
            if not step["new_points"]:
                step["new_points"] = _points_from_data_change(step["data_change"])
        elif op == "delete":
            if not step["operation_target"]:
                label = delete_labels[fallback_delete_idx] if fallback_delete_idx < len(delete_labels) else ""
                if label:
                    step["operation_target"] = {"category_name": label}
            fallback_delete_idx += 1
        elif op == "change":
            if fallback_change_idx < len(change_steps):
                fallback_change = change_steps[fallback_change_idx]
                step["operation_target"] = _merge_dict_like(
                    fallback_change.get("operation_target"),
                    step["operation_target"],
                )
                if fallback_change.get("data_change"):
                    step["data_change"] = _merge_dict_like(
                        fallback_change.get("data_change"),
                        step["data_change"],
                    )
            fallback_change_idx += 1

        enriched.append(step)

    return enriched


def _normalize_step_operation_target(raw_target: Any) -> dict[str, Any]:
    if not isinstance(raw_target, dict):
        return {}
    normalized = dict(raw_target)
    label = ""
    for key in ("category_name", "category", "add_category", "del_category"):
        value = raw_target.get(key)
        if isinstance(value, str) and value.strip():
            label = value.strip()
            break
    if label:
        normalized["category_name"] = label
    return normalized


def _llm_steps_cover_structured_ops(
    steps: list[Any],
    structured_context: dict[str, Any],
    operation_text: str,
) -> bool:
    if not steps:
        return False
    expected_ops = _ordered_structured_ops("", operation_text)
    if not expected_ops:
        return True
    actual_ops: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        token = _normalize_operation_token(str(step.get("operation") or ""))
        if token:
            actual_ops.append(token)
    if not actual_ops:
        return False
    idx = 0
    for token in actual_ops:
        if idx < len(expected_ops) and token == expected_ops[idx]:
            idx += 1
    return idx == len(expected_ops)


def _build_structured_steps(
    structured_context: dict[str, Any],
    operation_plan: dict[str, Any],
    operation_text: str,
) -> list[dict[str, Any]]:
    if not structured_context:
        return []

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
    ordered_ops = _ordered_structured_ops("", operation_text)
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


def _ordered_structured_ops(operation: str, operation_text: str) -> list[str]:
    explicit = [part.strip() for part in operation.split("+") if part.strip()] if operation else []
    normalized_explicit = [_normalize_operation_token(part) for part in explicit]
    normalized_explicit = [part for part in normalized_explicit if part]
    if len(normalized_explicit) >= 2:
        return normalized_explicit

    ordered: list[str] = []
    text, _, _ = _extract_embedded_structured_context(str(operation_text or "").strip())
    for sentence in re.split(r"[.;]\s+", text):
        lowered = sentence.strip().lower()
        if not lowered:
            continue
        token = ""
        if re.search(r"\b(add|append|insert|adding)\b", lowered):
            token = "add"
        elif re.search(r"\b(delete|remove|drop|deleting|removing)\b", lowered):
            token = "delete"
        elif re.search(r"\b(change|update|modify|set|changing|updating)\b", lowered):
            token = "change"
        if token and (not ordered or ordered[-1] != token):
            ordered.append(token)
    if ordered:
        return ordered
    if normalized_explicit:
        return normalized_explicit
    fallback = _normalize_operation_token(operation)
    return [fallback] if fallback else []


def _normalize_operation_token(token: str) -> str:
    lowered = str(token or "").strip().lower()
    if lowered in {"add", "append", "insert"}:
        return "add"
    if lowered in {"del", "delete", "remove", "drop"}:
        return "delete"
    if lowered in {"change", "update", "modify", "set"}:
        return "change"
    return ""


def _pick_hint(op_to_hints: dict[str, list[str]], op: str, idx: int) -> str:
    hints = op_to_hints.get(op, [])
    if not hints:
        return ""
    if idx < len(hints):
        return hints[idx]
    return hints[-1]


def _structured_delete_labels(operation_target: dict[str, Any]) -> list[str]:
    candidates = (
        operation_target.get("del_category"),
        operation_target.get("del_categories"),
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


def _extract_scatter_requested_color(
    *,
    step: dict[str, Any],
    update_spec: dict[str, Any],
) -> str:
    if not isinstance(step, dict):
        step = {}
    if not isinstance(update_spec, dict):
        update_spec = {}

    operation_target = step.get("operation_target")
    data_change = step.get("data_change")
    if not isinstance(operation_target, dict):
        operation_target = {}
    if not isinstance(data_change, dict):
        data_change = {}

    for key in ("point_color", "color", "fill", "rgb"):
        value = data_change.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    points = data_change.get("points")
    if isinstance(points, list):
        for item in points:
            if not isinstance(item, dict):
                continue
            for key in ("point_color", "color", "fill", "rgb"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

    for key in ("point_color", "color", "fill"):
        value = operation_target.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    value = update_spec.get("point_color")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return ""


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
    event_callback: Any | None = None,
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
        _emit_event(
            event_callback,
            "step_started",
            {
                "step": {
                    "index": idx + 1,
                    "operation": step.get("operation"),
                    "question": step_q,
                    "question_hint": step.get("question_hint"),
                    "operation_target": step.get("operation_target"),
                    "data_change": step.get("data_change"),
                }
            },
        )
        step_inputs = dict(inputs)
        step_inputs["svg_path"] = current_svg
        step_inputs["question"] = step_q
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
        update_spec = state.perception.get("update_spec", {})
        if isinstance(mapping_info, dict) and isinstance(update_spec, dict):
            mapping_info = dict(mapping_info)
            mapping_info["requested_point_color"] = _extract_scatter_requested_color(
                step=step,
                update_spec=update_spec,
            )
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
        _emit_event(
            event_callback,
            "step_finished",
            {
                "step": perception_steps[-1],
                "step_log": step_logs[-1],
                "perception_steps": perception_steps,
                "step_logs": step_logs,
                "output_image_path": output_image,
            },
        )

    return output_image, step_logs, perception_steps, last_state, scatter_points


def _resolve_supported_chart_type(perception: dict[str, Any], chart_type_hint: str = "") -> str:
    hinted = str(chart_type_hint or "").strip().lower()
    if hinted in SUPPORTED_SVG_CHART_TYPES:
        return hinted
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


def _apply_tool_augmented_answer(
    *,
    output: dict[str, Any],
    qa_question: str,
    answer_data_summary: dict[str, Any],
    tool_phase: dict[str, Any],
    answer_llm: Any,
) -> None:
    augmented_path = str(tool_phase.get("augmented_image_path") or "").strip()
    output["tool_augmented_image_path"] = augmented_path or None
    if tool_phase.get("ok") and augmented_path:
        output["answer_input_tool_augmented"] = {
            "question": qa_question,
            "output_image_path": augmented_path,
            "data_summary": answer_data_summary,
        }
        output["answer_tool_augmented"] = answer_question(
            qa_question=qa_question,
            data_summary=answer_data_summary,
            output_image_path=augmented_path,
            image_context_note=(
                "The requested chart update has already been applied, and visual augmentation "
                "has also been added to help reasoning. Answer the QA question only based on "
                "this updated and enhanced chart."
            ),
            llm=answer_llm,
        )
        output["answer"] = output["answer_tool_augmented"]
    else:
        output["answer_tool_augmented"] = None


def _validate_render_with_programmatic(
    *,
    output_image: str | None,
    chart_type: str,
    update_spec: dict[str, Any],
    step_logs: list[dict[str, Any]],
    llm: Any,
    svg_perception_mode: str | None = None,
) -> dict[str, Any]:
    basic_check = validate_render(output_image, chart_type, update_spec, llm=None)
    if not basic_check.get("ok"):
        return basic_check

    prog = _programmatic_validate(
        chart_type=chart_type,
        step_logs=step_logs,
        llm=llm,
        svg_perception_mode=svg_perception_mode,
    )
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


def _programmatic_validate(
    chart_type: str,
    step_logs: list[dict[str, Any]],
    llm: Any,
    svg_perception_mode: str | None = None,
) -> dict[str, Any] | None:
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

        perceived = perceive_svg(
            svg_path,
            question=question,
            llm=llm,
            perception_mode=svg_perception_mode,
        )
        mapping = perceived.get("mapping_info", {}) if isinstance(perceived, dict) else {}
        x_ticks = mapping.get("x_ticks", []) if isinstance(mapping, dict) else []
        y_ticks = mapping.get("y_ticks", []) if isinstance(mapping, dict) else []
        if len(x_ticks) < 2 or len(y_ticks) < 2:
            return {"ok": False, "confidence": 0.0, "issues": ["programmatic: insufficient axis ticks"]}

        _, legend_items = area_ops._extract_legend_items(root, content)
        labels = [item["label"] for item in legend_items if item.get("label")]
        parsed = area_ops._parse_year_value_update(question, labels, llm=llm)
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
