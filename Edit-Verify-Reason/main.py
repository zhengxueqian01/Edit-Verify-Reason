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
from chart_agent.core.htn_planner import HtnMethod, HtnTask, decompose_tasks
from chart_agent.core.trace import append_trace
from chart_agent.core.answerer import answer_question
from chart_agent.core.vision_tool_phase import run_visual_tool_phase
from chart_agent.llm_factory import make_llm
from chart_agent.prompts.prompt import (
    ANSWER_TOOL_AUGMENTED_IMAGE_CONTEXT_PROMPT,
    ANSWER_UPDATED_IMAGE_CONTEXT_PROMPT,
    build_svg_intent_plan_prompt,
    build_update_plan_prompt,
)
from chart_agent.perception.scatter_svg_updater import update_scatter_svg
from chart_agent.perception.area_svg_updater import update_area_svg
from chart_agent.perception.line_svg_updater import update_line_svg
from chart_agent.perception.render_validator import validate_render
from chart_agent.perception.svg_renderer import default_output_paths
from chart_agent.perception.svg_perceiver import perceive_svg
from chart_agent.perception import area_svg_updater as area_ops
from chart_agent.perception import line_svg_updater as line_ops
from chart_agent.perception import scatter_svg_updater as scatter_ops


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chart agent perception CLI")
    parser.add_argument("--question", required=True, help="User question")
    parser.add_argument("--text_spec", default="", help="Text specification")
    parser.add_argument("--image", dest="image_path", default="", help="Path to image")
    parser.add_argument("--svg", dest="svg_path", default="", help="Path to SVG")
    parser.add_argument(
        "--experiment-mode",
        default="full",
        choices=("full", "wo_question_decomposition", "wo_svg_update"),
        help="Experiment mode.",
    )
    return parser.parse_args()


SUPPORTED_SVG_CHART_TYPES = {"scatter", "line", "area"}
EXPERIMENT_MODES = {"full", "wo_question_decomposition", "wo_svg_update"}
TOOL_AUG_CONFIDENCE_THRESHOLD = 0.85
STEP_REPLAN_RETRIES = 1


class _StepExecutionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        step_logs: list[dict[str, Any]],
        perception_steps: list[dict[str, Any]],
        last_state: Any,
        scatter_points: list[dict[str, float]],
        failure_info: dict[str, Any],
        failed_step: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.step_logs = list(step_logs)
        self.perception_steps = list(perception_steps)
        self.last_state = last_state
        self.scatter_points = list(scatter_points)
        self.failure_info = dict(failure_info)
        self.failed_step = dict(failed_step) if isinstance(failed_step, dict) else None


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
    experiment_mode = _normalize_experiment_mode(inputs.get("experiment_mode"))
    splitter_llm = make_llm(splitter_config)
    planner_llm = make_llm(planner_config)
    executor_llm = make_llm(executor_config)
    answer_llm = make_llm(answer_config)
    tool_planner_llm = make_llm(tool_planner_config)
    svg_update_mode = get_svg_update_mode(inputs.get("svg_update_mode"))
    svg_perception_mode = inputs.get("svg_perception_mode")
    structured_context = _normalize_structured_context(inputs.get("structured_update_context"))
    original_question = str(inputs.get("question") or "").strip()
    resolve_inputs = dict(inputs)
    if experiment_mode == "wo_question_decomposition":
        resolve_inputs["auto_split_question"] = False
    update_question, qa_question, split_info, split_data_change = _resolve_questions(resolve_inputs, splitter_llm)
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
        "question": original_question,
        "output_image_path": original_image_path,
        "data_summary": {},
    }
    answer_original = answer_question(
        qa_question=original_question,
        data_summary={},
        output_image_path=original_image_path,
        answer_stage="original",
        image_context_note="This is the original chart image before any requested update is applied.",
        llm=answer_llm,
    )
    if not inputs.get("svg_path"):
        raise ValueError("This entry only supports SVG-based updates for line/area/scatter.")

    max_render_retries = int(inputs.get("max_render_retries", 2))
    retries = max(0, max_render_retries)
    render_output_dir = str(inputs.get("render_output_dir") or "").strip() or None
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
        plan_steps = _operation_steps_from_plan(
            operation_plan,
            planned_question,
            structured_context,
            chart_type=str(chart_type),
            update_mode=svg_update_mode,
        )
        if experiment_mode == "wo_svg_update":
            answer_data_summary = {
                "update_spec": state.perception.get("update_spec", {}),
                "mapping_info_summary": {
                    "num_points": state.perception.get("primitives_summary", {}).get("num_points"),
                    "num_areas": state.perception.get("primitives_summary", {}).get("num_areas"),
                    "num_lines": state.perception.get("primitives_summary", {}).get("num_lines"),
                },
                "operation_plan": last_operation_plan,
                "perception_steps": [],
                "latest_step_logs": [],
                "ablation_mode": experiment_mode,
                "unexecuted_update_reasoning": {
                    "normalized_question": str(operation_plan.get("normalized_question") or planned_question),
                    "llm_success": bool(operation_plan.get("llm_success")),
                    "steps": [dict(step) for step in plan_steps if isinstance(step, dict)],
                },
            }
            render_check = {
                "ok": True,
                "confidence": 1.0,
                "issues": ["skipped:wo_svg_update"],
            }
            output = {
                "trace": [record.__dict__ for record in state.trace],
                "output_image_path": original_image_path,
                "operation_plan": last_operation_plan,
                "render_check": render_check,
                "attempt_logs": [
                    {
                        "attempt": attempt,
                        "chart_type": chart_type,
                        "operation_plan": operation_plan,
                        "planned_question": planned_question,
                        "output_image_path": original_image_path,
                        "step_logs": [],
                        "render_check": render_check,
                        "planned_steps": [dict(step) for step in plan_steps if isinstance(step, dict)],
                    }
                ],
                "operation_text": split_info.get("operation_text") or update_question,
                "qa_question": qa_question,
                "update_question": update_question,
                "resolved_data_change": structured_context.get("data_change", {}),
                "question_split": split_info,
                "perception_steps": [],
                "svg_update_mode": svg_update_mode,
                "experiment_mode": experiment_mode,
                "answer_original_input": answer_original_input,
                "answer_original": answer_original,
                "answer_input": {
                    "question": original_question,
                    "chart_type": chart_type,
                    "output_image_path": original_image_path,
                    "data_summary": answer_data_summary,
                },
                "model_overrides": model_overrides,
                "resolved_task_models": {
                    "splitter": splitter_config.model,
                    "planner": planner_config.model,
                    "executor": executor_config.model,
                    "answer": answer_config.model,
                    "tool_planner": tool_planner_config.model,
                },
            }
            initial_answer = answer_question(
                qa_question=original_question,
                data_summary=answer_data_summary,
                chart_type=chart_type,
                output_image_path=original_image_path,
                answer_stage="updated",
                image_context_note=(
                    "The chart image is the original chart. Planned update reasoning and operation steps are "
                    "provided in the prompt context, but the SVG update was not executed."
                ),
                llm=answer_llm,
            )
            output["answer_initial"] = initial_answer
            output["answer"] = initial_answer
            output["tool_phase"] = {"ok": False, "issues": ["skipped:wo_svg_update"]}
            output["tool_augmented_image_path"] = None
            output["answer_tool_augmented"] = None
            return output
        has_executable_plan = bool(plan_steps)
        if not operation_plan.get("llm_success") and not has_executable_plan:
            failure_info = _classify_plan_failure("llm planning failed")
            render_check = {
                "ok": False,
                "confidence": 0.0,
                "issues": ["llm_plan_failed"],
                "failure_info": failure_info,
            }
            attempt_logs.append(
                {
                    "attempt": attempt,
                    "chart_type": chart_type,
                    "operation_plan": operation_plan,
                    "planned_question": planned_question,
                    "output_image_path": None,
                    "render_check": render_check,
                    "failure_info": failure_info,
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
                render_output_dir=render_output_dir,
                update_mode=svg_update_mode,
                llm=executor_llm,
                planner_llm=planner_llm,
                used_scatter_points=used_scatter_points,
                event_callback=event_callback,
            )
        except _StepExecutionError as exc:
            step_logs = exc.step_logs
            perception_steps = exc.perception_steps
            if exc.last_state is not None:
                last_state = exc.last_state
            used_scatter_points = exc.scatter_points
            last_perception_steps = perception_steps
            render_check = {
                "ok": False,
                "confidence": 0.0,
                "issues": [f"operation_step_failed: {exc}"],
                "failure_info": exc.failure_info,
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
                    "failure_info": exc.failure_info,
                    "failed_step": exc.failed_step,
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
        except Exception as exc:
            failure_info = _build_failure_info(
                failure_type="execution_error",
                failure_stage="step_execute",
                retryable=True,
                retry_hint=str(exc),
                replan_strategy="restart_attempt",
                message=str(exc),
                exception_type=type(exc).__name__,
                chart_type=chart_type,
            )
            render_check = {
                "ok": False,
                "confidence": 0.0,
                "issues": [f"operation_step_failed: {exc}"],
                "failure_info": failure_info,
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
                    "failure_info": failure_info,
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
        if not render_check.get("ok"):
            render_check["failure_info"] = _classify_render_failure(render_check)
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
        "experiment_mode": experiment_mode,
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
        qa_question=qa_question,
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

    answer_data_summary: dict[str, Any] = {
        "update_spec": last_state.perception.get("update_spec", {}),
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
        "question": original_question if experiment_mode == "wo_question_decomposition" else qa_question,
        "chart_type": chart_type,
        "output_image_path": last_output_image,
        "data_summary": answer_data_summary,
    }

    final_eval_image = str(inputs.get("answer_image_path") or "").strip() or last_output_image

    initial_answer = answer_question(
        qa_question=(original_question if experiment_mode == "wo_question_decomposition" else qa_question),
        data_summary=answer_data_summary,
        chart_type=chart_type,
        output_image_path=final_eval_image,
        answer_stage="updated",
        image_context_note=ANSWER_UPDATED_IMAGE_CONTEXT_PROMPT,
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
    tool_phase = run_visual_tool_phase(
        question=qa_question,
        chart_type=chart_type,
        data_summary=answer_data_summary,
        image_path=last_output_image,
        svg_path=final_step_svg_path or None,
        llm=tool_planner_llm,
        svg_perception_mode=svg_perception_mode,
    )
    if isinstance(tool_phase, dict):
        tool_phase["initial_confidence"] = initial_confidence
        tool_phase["confidence_threshold"] = TOOL_AUG_CONFIDENCE_THRESHOLD
        tool_phase["force_tool_phase_candidate"] = force_tool_phase
    output["tool_phase"] = tool_phase
    _apply_tool_augmented_answer(
        output=output,
        qa_question=(original_question if experiment_mode == "wo_question_decomposition" else qa_question),
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
    embedded_update_text, embedded_update_context, embedded_update_suffix = _extract_embedded_structured_context(cand_update)
    embedded_update_context = _normalize_structured_context(embedded_update_context)
    cand_operation_target = _merge_dict_like(
        embedded_update_context.get("operation_target"),
        cand_operation_target,
    )
    cand_data_change = _merge_dict_like(
        embedded_update_context.get("data_change"),
        cand_data_change,
    )
    if embedded_update_text:
        cand_update = embedded_update_text
    structured_context_note = ""
    if inline_structured_suffix or embedded_update_suffix:
        structured_context_note = _structured_context_suffix_text(
            operation_target=cand_operation_target,
            data_change=cand_data_change,
            raw_suffix=inline_structured_suffix or embedded_update_suffix,
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
    qa_question = _preserve_cluster_param_suffix(split_source, qa_question)
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


def _normalize_experiment_mode(raw: Any) -> str:
    mode = str(raw or "").strip()
    if mode in EXPERIMENT_MODES:
        return mode
    return "full"


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
            raw.find('operation_target:'),
            raw.find('data_change:'),
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
        idx = -1
        for pattern in (f'"{key}"', f'{key}:'):
            idx = text.find(pattern)
            if idx >= 0:
                break
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


def _preserve_cluster_param_suffix(source_question: str, qa_question: str) -> str:
    source = str(source_question or "").strip()
    qa = str(qa_question or "").strip()
    if not source or not qa:
        return qa
    if "eps" in qa.lower() and "min_samples" in qa.lower():
        return qa
    match = re.search(
        r"(\(\s*eps\s*[:=]\s*[\d.]+\s*,\s*min_samples?\s*[:=]\s*\d+\s*\)\s*[?.!]?)\s*$",
        source,
        flags=re.IGNORECASE,
    )
    if not match:
        return qa
    suffix = match.group(1).strip()
    qa_core = qa.rstrip()
    if qa_core.endswith("?"):
        question_stem = qa_core[:-1].rstrip()
        return f"{question_stem}? {suffix}".strip()
    if qa_core.endswith((".", "!")):
        qa_core = qa_core[:-1].rstrip()
    if suffix and not qa_core.endswith(suffix):
        qa_core = f"{qa_core} {suffix}".strip()
    return qa_core.strip()


def _extract_preface_operations(prefix: str) -> list[dict[str, Any]]:
    text = str(prefix or "").strip().rstrip(".")
    if not text:
        return []
    clauses = re.split(
        r"\s+(?:and|then)\s+(?=(?:adding|add|deleting|delete|removing|remove|changing|change|updating|update|applying|apply)\b)",
        text,
        flags=re.IGNORECASE,
    )
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
        add_change = {"category_name": label, "years": years, "values": values}
    else:
        add_change = {"category_name": label}
    return {
        "operation": "add",
        "operation_text": _normalize_gerund_clause(text),
        "operation_target": {"add_category": label},
        "data_change": add_change,
    }


def _parse_delete_clause(text: str) -> dict[str, Any] | None:
    labels = _extract_category_labels(text)
    if not labels:
        return None
    if len(labels) == 1:
        label = labels[0]
        operation_target: dict[str, Any] = {"del_category": label}
        data_change: dict[str, Any] = {"category_name": label}
    else:
        operation_target = {"del_categories": labels}
        data_change = {"category_names": labels}
    return {
        "operation": "delete",
        "operation_text": _normalize_gerund_clause(text),
        "operation_target": operation_target,
        "data_change": data_change,
    }


def _parse_change_clause(text: str) -> dict[str, Any] | None:
    return {
        "operation": "change",
        "operation_text": _normalize_gerund_clause(text),
        "operation_target": {},
        "data_change": {},
    }


def _extract_category_label(text: str) -> str:
    labels = _extract_category_labels(text)
    return labels[0] if labels else ""


def _extract_category_labels(text: str) -> list[str]:
    quoted = re.findall(r'["“”]([^"“”]+)["“”]', text)
    if quoted:
        return [item.strip() for item in quoted if item.strip()]
    open_quoted = re.search(r'[“"]([^"“”]+)$', text)
    if open_quoted:
        label = open_quoted.group(1).strip()
        return [label] if label else []
    plural_match = re.search(
        r"(?:categories|series)\s+([A-Za-z0-9][A-Za-z0-9&'()\/,\- ]*[A-Za-z0-9)])$",
        text,
        flags=re.IGNORECASE,
    )
    if plural_match:
        labels = _split_category_label_segment(plural_match.group(1))
        if labels:
            return labels
    plural_match = re.search(
        r"(?:categories|series)\s+([A-Za-z0-9][A-Za-z0-9&'()\/,\- ]*[A-Za-z0-9])",
        text,
        flags=re.IGNORECASE,
    )
    if plural_match:
        labels = _split_category_label_segment(plural_match.group(1))
        if labels:
            return labels
    match = re.search(
        r"(?:category|series)\s+([A-Za-z0-9][A-Za-z0-9&'()\/,\- ]*[A-Za-z0-9])",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        label = match.group(1).strip()
        return [label] if label else []
    return []


def _split_category_label_segment(segment: str) -> list[str]:
    raw = str(segment or "").strip().strip(".")
    if not raw:
        return []
    parts = re.split(r"\s+and\s+|,\s*", raw)
    labels: list[str] = []
    for part in parts:
        label = part.strip().strip('"\''"“”")
        if label and label not in labels:
            labels.append(label)
    return labels


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
    operation_target = structured_context.get("operation_target")
    data_change = structured_context.get("data_change")
    out["operation_target"] = operation_target if isinstance(operation_target, dict) else {}
    out["data_change"] = data_change if isinstance(data_change, dict) else {}
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
        "experiment_mode": args.experiment_mode,
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


def _should_answer_after_failed_render(
    *,
    chart_type: str,
    structured_context: dict[str, Any],
    qa_question: str,
    output_image_path: str | None,
    attempt_logs: list[dict[str, Any]] | None = None,
    max_render_retries: int = 2,
) -> bool:
    del structured_context
    output_path = str(output_image_path or "").strip()
    if chart_type == "scatter" and any(token in (qa_question or "").lower() for token in ("cluster", "clusters")):
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
    # 更新规划阶段的 prompt。
    prompt = build_update_plan_prompt(
        question=question,
        chart_type=chart_type,
        retry_hint=retry_hint,
        new_points_schema=new_points_schema,
    )
    try:
        response = llm.invoke(prompt)
    except Exception:
        return _heuristic_plan_update(question)
    content = getattr(response, "content", "")
    payload = _safe_json_loads(content)
    if not payload:
        return _heuristic_plan_update(question)

    normalized = str(payload.get("normalized_question") or question).strip() or question
    points = _coerce_points(payload.get("new_points", []))
    steps = _coerce_steps(payload.get("steps", []))
    return {
        "normalized_question": normalized,
        "steps": steps,
        "new_points": points,
        "llm_success": True,
        "llm_raw": content,
    }


def _heuristic_plan_update(question: str) -> dict[str, Any]:
    normalized = (question or "").strip()
    op = _infer_single_operation_from_text(normalized)
    return {
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
    # SVG 意图规划阶段的 prompt。
    prompt = build_svg_intent_plan_prompt(
        operation_text=operation_text,
        chart_type=chart_type,
        structured_context_summary=_intent_structured_context_summary(structured_context),
        perception_summary=_intent_perception_summary(perception),
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
        "- Preserve the same operation order as the original input text when rewriting operation_text.\n"
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


def _operation_step_candidates_from_plan(
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
        if isinstance(steps, list) and steps:
            enriched_steps = _enrich_llm_steps_with_structured_data(steps, structured_context, planned_question)
            return _merge_structured_with_existing_steps(structured_steps, enriched_steps)
        return structured_steps
    if isinstance(steps, list) and steps:
        return _enrich_llm_steps_with_structured_data(steps, structured_context, planned_question)
    return [
        {
            "operation": _infer_single_operation_from_text(planned_question),
            "question": planned_question,
            "question_hint": planned_question,
            "new_points": operation_plan.get("new_points", []),
        }
    ]


def _operation_steps_from_rules_plan(
    operation_plan: dict[str, Any],
    planned_question: str,
    structured_context: dict[str, Any],
) -> list[dict[str, Any]]:
    candidate_steps = _operation_step_candidates_from_plan(
        operation_plan,
        planned_question,
        structured_context,
    )
    return _prune_redundant_component_steps(_expand_composite_steps(candidate_steps))


def _operation_steps_from_htn_plan(
    operation_plan: dict[str, Any],
    planned_question: str,
    structured_context: dict[str, Any],
    chart_type: str = "",
) -> list[dict[str, Any]]:
    candidate_steps = _operation_step_candidates_from_plan(
        operation_plan,
        planned_question,
        structured_context,
    )
    root_task = HtnTask(
        "handle_chart_update",
        {
            "chart_type": str(chart_type or "").strip().lower(),
            "candidate_steps": candidate_steps,
            "planned_question": planned_question,
            "structured_context": structured_context,
        },
    )
    methods = _build_domain_htn_methods()
    htn_result = decompose_tasks(root_task, methods=methods, operator_names={"emit_step"})
    emitted_steps = [dict(item.get("step") or {}) for item in htn_result.operators]
    resolved_steps = _prune_redundant_component_steps(emitted_steps)
    operation_plan["planning_form"] = "htn"
    operation_plan["htn_chart_type"] = str(chart_type or "").strip().lower() or "generic"
    operation_plan["htn_trace"] = htn_result.trace
    operation_plan["candidate_steps"] = [dict(step) for step in candidate_steps]
    operation_plan["resolved_steps"] = [dict(step) for step in resolved_steps]
    return resolved_steps


def _operation_steps_from_plan(
    operation_plan: dict[str, Any],
    planned_question: str,
    structured_context: dict[str, Any],
    *,
    chart_type: str = "",
    update_mode: str = "rules",
) -> list[dict[str, Any]]:
    if update_mode == "htn":
        return _operation_steps_from_htn_plan(
            operation_plan,
            planned_question,
            structured_context,
            chart_type=chart_type,
        )
    operation_plan["planning_form"] = "rules"
    return _operation_steps_from_rules_plan(operation_plan, planned_question, structured_context)


def _build_domain_htn_methods() -> list[HtnMethod]:
    return [
        HtnMethod(
            task_name="handle_chart_update",
            name="select_line_update_method",
            condition=lambda task: _htn_chart_type(task) == "line",
            expand=lambda task: [HtnTask("plan_line_update", dict(task.payload))],
        ),
        HtnMethod(
            task_name="handle_chart_update",
            name="select_area_update_method",
            condition=lambda task: _htn_chart_type(task) == "area",
            expand=lambda task: [HtnTask("plan_area_update", dict(task.payload))],
        ),
        HtnMethod(
            task_name="handle_chart_update",
            name="select_scatter_update_method",
            condition=lambda task: _htn_chart_type(task) == "scatter",
            expand=lambda task: [HtnTask("plan_scatter_update", dict(task.payload))],
        ),
        HtnMethod(
            task_name="handle_chart_update",
            name="select_generic_update_method",
            condition=lambda task: True,
            expand=lambda task: [HtnTask("plan_generic_update", dict(task.payload))],
        ),
        HtnMethod(
            task_name="plan_line_update",
            name="decompose_line_update",
            condition=_htn_has_candidate_steps,
            expand=_htn_expand_chart_update,
        ),
        HtnMethod(
            task_name="plan_area_update",
            name="decompose_area_update",
            condition=_htn_has_candidate_steps,
            expand=_htn_expand_chart_update,
        ),
        HtnMethod(
            task_name="plan_scatter_update",
            name="decompose_scatter_update",
            condition=_htn_has_candidate_steps,
            expand=_htn_expand_chart_update,
        ),
        HtnMethod(
            task_name="plan_generic_update",
            name="decompose_generic_update",
            condition=_htn_has_candidate_steps,
            expand=_htn_expand_chart_update,
        ),
        HtnMethod(
            task_name="plan_delete_family",
            name="expand_delete_family",
            condition=lambda task: bool(_htn_family_steps(task)),
            expand=lambda task: _htn_expand_family_steps(task, "plan_delete_step"),
        ),
        HtnMethod(
            task_name="plan_add_family",
            name="expand_add_family",
            condition=lambda task: bool(_htn_family_steps(task)),
            expand=lambda task: _htn_expand_family_steps(task, "plan_add_step"),
        ),
        HtnMethod(
            task_name="plan_change_family",
            name="expand_change_family",
            condition=lambda task: bool(_htn_family_steps(task)),
            expand=lambda task: _htn_expand_family_steps(task, "plan_change_step"),
        ),
        HtnMethod(
            task_name="plan_unknown_family",
            name="expand_unknown_family",
            condition=lambda task: bool(_htn_family_steps(task)),
            expand=lambda task: _htn_expand_family_steps(task, "plan_unknown_step"),
        ),
        HtnMethod(
            task_name="plan_delete_step",
            name="split_delete_targets",
            condition=_htn_step_is_composite_delete,
            expand=lambda task: _htn_expand_delete_step(task, child_task_name="plan_delete_step"),
        ),
        HtnMethod(
            task_name="plan_delete_step",
            name="emit_delete_step",
            condition=lambda task: isinstance(task.payload.get("step"), dict),
            expand=_htn_emit_typed_step,
        ),
        HtnMethod(
            task_name="plan_add_step",
            name="emit_add_step",
            condition=lambda task: isinstance(task.payload.get("step"), dict),
            expand=_htn_emit_typed_step,
        ),
        HtnMethod(
            task_name="plan_change_step",
            name="split_change_targets",
            condition=_htn_step_is_composite_change,
            expand=lambda task: _htn_expand_change_step(task, child_task_name="plan_change_step"),
        ),
        HtnMethod(
            task_name="plan_change_step",
            name="emit_change_step",
            condition=lambda task: isinstance(task.payload.get("step"), dict),
            expand=_htn_emit_typed_step,
        ),
        HtnMethod(
            task_name="plan_unknown_step",
            name="emit_unknown_step",
            condition=lambda task: isinstance(task.payload.get("step"), dict),
            expand=_htn_emit_typed_step,
        ),
    ]


def _htn_chart_type(task: HtnTask) -> str:
    return str(task.payload.get("chart_type") or "").strip().lower()


def _htn_has_candidate_steps(task: HtnTask) -> bool:
    candidate_steps = task.payload.get("candidate_steps")
    return isinstance(candidate_steps, list) and any(isinstance(step, dict) for step in candidate_steps)


def _htn_expand_chart_update(task: HtnTask) -> list[HtnTask]:
    candidate_steps = [
        dict(step)
        for step in task.payload.get("candidate_steps", [])
        if isinstance(step, dict)
    ]
    sequence = _htn_operation_family_sequence(
        candidate_steps,
        task.payload.get("structured_context"),
        str(task.payload.get("planned_question") or ""),
    )
    grouped_steps = _group_candidate_steps_by_operation(candidate_steps)
    children: list[HtnTask] = []
    for operation in sequence:
        family_steps = grouped_steps.get(operation, [])
        if not family_steps:
            continue
        family_task_name = {
            "delete": "plan_delete_family",
            "add": "plan_add_family",
            "change": "plan_change_family",
        }.get(operation, "plan_unknown_family")
        children.append(
            HtnTask(
                family_task_name,
                {
                    "chart_type": _htn_chart_type(task),
                    "operation": operation,
                    "steps": [dict(step) for step in family_steps],
                },
            )
        )
    return children


def _htn_operation_family_sequence(
    candidate_steps: list[dict[str, Any]],
    structured_context: Any,
    planned_question: str,
) -> list[str]:
    ordered: list[str] = []

    def append(token: str) -> None:
        normalized = _normalize_operation_token(token) or ("unknown" if token == "unknown" else "")
        if normalized and normalized not in ordered:
            ordered.append(normalized)

    if isinstance(structured_context, dict):
        for token in _expected_structured_ops(structured_context, planned_question):
            append(token)

    has_unknown = False
    for step in candidate_steps:
        token = _normalize_operation_token(str(step.get("operation") or ""))
        if token:
            append(token)
            continue
        has_unknown = True

    if has_unknown or not ordered:
        append("unknown")
    return ordered


def _group_candidate_steps_by_operation(candidate_steps: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for step in candidate_steps:
        operation = _normalize_operation_token(str(step.get("operation") or "")) or "unknown"
        grouped.setdefault(operation, []).append(dict(step))
    return grouped


def _htn_family_steps(task: HtnTask) -> list[dict[str, Any]]:
    steps = task.payload.get("steps")
    if not isinstance(steps, list):
        return []
    return [dict(step) for step in steps if isinstance(step, dict)]


def _htn_expand_family_steps(task: HtnTask, child_task_name: str) -> list[HtnTask]:
    children: list[HtnTask] = []
    for step in _htn_family_steps(task):
        children.append(
            HtnTask(
                child_task_name,
                {
                    "chart_type": _htn_chart_type(task),
                    "step": dict(step),
                },
            )
        )
    return children


def _htn_emit_typed_step(task: HtnTask) -> list[HtnTask]:
    step = task.payload.get("step")
    if not isinstance(step, dict):
        return []
    return [HtnTask("emit_step", {"step": dict(step)})]


def _htn_step_is_composite_delete(task: HtnTask) -> bool:
    step = task.payload.get("step")
    if not isinstance(step, dict):
        return False
    if str(step.get("operation") or "").strip().lower() != "delete":
        return False
    operation_target = step.get("operation_target") if isinstance(step.get("operation_target"), dict) else {}
    explicit_label = str(operation_target.get("category_name") or "").strip()
    explicit_labels = operation_target.get("category_names")
    if explicit_label and not explicit_labels:
        return False
    labels = _structured_delete_labels(
        operation_target,
        step.get("data_change") if isinstance(step.get("data_change"), dict) else {},
    )
    return len(labels) > 1


def _htn_expand_delete_step(task: HtnTask, *, child_task_name: str) -> list[HtnTask]:
    step = task.payload.get("step")
    if not isinstance(step, dict):
        return []
    operation_target = step.get("operation_target") if isinstance(step.get("operation_target"), dict) else {}
    data_change = step.get("data_change") if isinstance(step.get("data_change"), dict) else {}
    labels = _structured_delete_labels(operation_target, data_change)
    children: list[HtnTask] = []
    for label in labels:
        split_step = dict(step)
        split_step["operation_target"] = {"category_name": label}
        split_step["data_change"] = {"del": {"category_name": label}}
        children.append(
            HtnTask(
                child_task_name,
                {
                    "chart_type": _htn_chart_type(task),
                    "step": split_step,
                },
            )
        )
    return children


def _htn_step_is_composite_change(task: HtnTask) -> bool:
    step = task.payload.get("step")
    if not isinstance(step, dict):
        return False
    if str(step.get("operation") or "").strip().lower() != "change":
        return False
    atomic_changes = _atomic_changes_from_step(step.get("operation_target"), step.get("data_change"))
    return len(atomic_changes) > 1


def _htn_expand_change_step(task: HtnTask, *, child_task_name: str) -> list[HtnTask]:
    step = task.payload.get("step")
    if not isinstance(step, dict):
        return []
    atomic_changes = _atomic_changes_from_step(step.get("operation_target"), step.get("data_change"))
    children: list[HtnTask] = []
    for change in atomic_changes:
        if not isinstance(change, dict):
            continue
        split_step = dict(step)
        label = str(change.get("category_name") or "").strip()
        split_step["operation_target"] = {"category_name": label} if label else {}
        split_step["data_change"] = {"changes": [change]}
        children.append(
            HtnTask(
                child_task_name,
                {
                    "chart_type": _htn_chart_type(task),
                    "step": split_step,
                },
            )
        )
    return children


def _step_match_key(step: dict[str, Any]) -> tuple[str, str, str]:
    operation = str(step.get("operation") or "").strip().lower()
    target = step.get("operation_target")
    data_change = step.get("data_change")
    label = ""
    if isinstance(target, dict):
        label = _first_non_empty_string(target.get("category_name"), target.get("category_names"))
    marker = ""
    if operation == "change":
        atomic_changes = _atomic_changes_from_step(target, data_change)
        if atomic_changes:
            marker = str((atomic_changes[0].get("years") or [""])[0])
    return operation, label, marker


def _merge_structured_with_existing_steps(
    structured_steps: list[dict[str, Any]],
    existing_steps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    remaining: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for step in existing_steps:
        remaining.setdefault(_step_match_key(step), []).append(step)

    merged: list[dict[str, Any]] = []
    for structured_step in structured_steps:
        key = _step_match_key(structured_step)
        matches = remaining.get(key) or []
        if matches:
            existing = matches.pop(0)
            merged_step = dict(structured_step)
            op = str(structured_step.get("operation") or "").strip().lower()
            if op in {"add", "change"}:
                # Keep structured atomic change payloads authoritative so composite LLM
                # payloads do not leak stale sibling updates into this step.
                merged_step["operation_target"] = _merge_dict_like(
                    existing.get("operation_target"),
                    structured_step.get("operation_target"),
                )
                merged_step["data_change"] = dict(
                    structured_step.get("data_change")
                    if isinstance(structured_step.get("data_change"), dict)
                    else {}
                )
            else:
                merged_step["operation_target"] = _merge_dict_like(
                    structured_step.get("operation_target"),
                    existing.get("operation_target"),
                )
                merged_step["data_change"] = _merge_dict_like(
                    structured_step.get("data_change"),
                    existing.get("data_change"),
                )
            merged_step["new_points"] = _coerce_points(existing.get("new_points", structured_step.get("new_points", [])))
            merged_step["question_hint"] = str(
                existing.get("question_hint") or structured_step.get("question_hint") or ""
            ).strip()
            merged.append(merged_step)
        else:
            merged.append(structured_step)

    for leftovers in remaining.values():
        for leftover in leftovers:
            op = _normalize_operation_token(str(leftover.get("operation") or ""))
            if structured_steps and not op:
                continue
            merged.append(leftover)
    return merged


def _infer_single_operation_from_text(text: str) -> str:
    lowered = str(text or "").strip().lower()
    if re.search(r"\b(delete|remove|drop)\b", lowered):
        return "delete"
    if re.search(r"\b(change|update|modify|set)\b", lowered):
        return "change"
    if re.search(r"\b(add|append|insert)\b", lowered):
        return "add"
    return "unknown"


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
    delete_labels = _structured_delete_labels(operation_target, data_change)
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
            step_atomic_changes = _atomic_changes_from_step(step["operation_target"], step["data_change"])
            if step_atomic_changes:
                first_label = str(step_atomic_changes[0].get("category_name") or "").strip()
                if first_label:
                    step["operation_target"] = _merge_dict_like({"category_name": first_label}, step["operation_target"])
                step["data_change"] = {"changes": step_atomic_changes}
                fallback_change_idx += len(step_atomic_changes)
            elif fallback_change_idx < len(change_steps):
                remaining_changes: list[dict[str, Any]] = []
                for change_step in change_steps[fallback_change_idx:]:
                    change_payload = change_step.get("data_change")
                    if not isinstance(change_payload, dict):
                        continue
                    changes = change_payload.get("changes")
                    if isinstance(changes, list):
                        remaining_changes.extend(change for change in changes if isinstance(change, dict))
                if remaining_changes:
                    first_label = str(remaining_changes[0].get("category_name") or "").strip()
                    if first_label:
                        step["operation_target"] = _merge_dict_like(
                            {"category_name": first_label},
                            step["operation_target"],
                        )
                    step["data_change"] = {"changes": remaining_changes}
                fallback_change_idx = len(change_steps)

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
    expected_ops = _expected_structured_ops(structured_context, operation_text)
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
    ordered_ops = _expected_structured_ops(structured_context, operation_text)
    for op in ordered_ops:
        if op in {"del", "delete"}:
            labels = _structured_delete_labels(operation_target, data_change)
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


def _expected_structured_ops(
    structured_context: dict[str, Any],
    operation_text: str,
) -> list[str]:
    if isinstance(structured_context, dict):
        structured_ops = _ordered_structured_ops_from_context(structured_context)
        if structured_ops:
            return structured_ops
    return _ordered_structured_ops("", operation_text)


def _ordered_structured_ops_from_context(structured_context: dict[str, Any]) -> list[str]:
    if not isinstance(structured_context, dict):
        return []
    data_change = structured_context.get("data_change")
    if not isinstance(data_change, dict):
        return []

    ordered: list[str] = []

    def append(token: str) -> None:
        normalized = _normalize_operation_token(token)
        if normalized and (not ordered or ordered[-1] != normalized):
            ordered.append(normalized)

    for key, value in data_change.items():
        if value in (None, "", [], {}):
            continue
        if key in {"del", "delete", "remove", "drop"}:
            append("delete")
        elif key in {"change", "changes", "update", "modify", "set"}:
            append("change")
        elif key in {"add", "insert", "append"}:
            append("add")

    if ordered:
        return ordered

    operation_target = structured_context.get("operation_target")
    if isinstance(operation_target, dict):
        if any(operation_target.get(key) for key in ("add_category", "add_categories")):
            append("add")
        if any(operation_target.get(key) for key in ("del_category", "del_categories")):
            append("delete")

    return ordered


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


def _first_non_empty_string(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    return item.strip()
    return ""


def _append_unique_labels(labels: list[str], value: Any) -> None:
    candidates: list[str] = []
    if isinstance(value, str) and value.strip():
        candidates = [value.strip()]
    elif isinstance(value, list):
        candidates = [str(item).strip() for item in value if isinstance(item, str) and item.strip()]
    for item in candidates:
        if item not in labels:
            labels.append(item)


def _structured_delete_labels(
    operation_target: dict[str, Any],
    data_change: dict[str, Any] | None = None,
) -> list[str]:
    labels: list[str] = []
    del_change = data_change.get("del") if isinstance(data_change, dict) else None
    candidates = [
        operation_target.get("del_category"),
        operation_target.get("del_categories"),
    ]
    if isinstance(del_change, dict):
        candidates.extend(
            [
                del_change.get("category_name"),
                del_change.get("category_names"),
                del_change.get("category"),
            ]
        )
    else:
        candidates.extend(
            [
                operation_target.get("category_name"),
                operation_target.get("category_names"),
            ]
        )
    for value in candidates:
        _append_unique_labels(labels, value)
    if isinstance(del_change, dict):
        for key in ("category_name", "category_names", "category"):
            _append_unique_labels(labels, del_change.get(key))
    return labels


def _structured_add_payload(
    operation_target: dict[str, Any],
    data_change: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    add_change = data_change.get("add") if isinstance(data_change.get("add"), dict) else data_change
    add_target: dict[str, Any] = {}
    label = _first_non_empty_string(
        add_change.get("category_name") if isinstance(add_change, dict) else None,
        operation_target.get("add_category"),
        operation_target.get("category_name"),
    )
    if label:
        add_target["category_name"] = label
    return add_target, add_change if isinstance(add_change, dict) else {}


def _structured_change_steps(
    operation_target: dict[str, Any],
    data_change: dict[str, Any],
    hints: list[str],
) -> list[dict[str, Any]]:
    change_root = data_change.get("change") if isinstance(data_change.get("change"), dict) else data_change
    if not isinstance(change_root, dict):
        return []
    atomic_changes = _flatten_atomic_changes(change_root.get("changes"))
    if not atomic_changes:
        return []

    steps: list[dict[str, Any]] = []
    for idx, change in enumerate(atomic_changes):
        label = str(change.get("category_name") or "").strip()
        step_target = {"category_name": label} if label else {}
        step_change = {
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


def _flatten_atomic_changes(changes: Any) -> list[dict[str, Any]]:
    if not isinstance(changes, list):
        return []
    flattened: list[dict[str, Any]] = []
    for change in changes:
        if not isinstance(change, dict):
            continue
        label = str(change.get("category_name") or change.get("category") or "").strip()
        updates = change.get("updates")
        if isinstance(updates, list) and updates:
            for update in updates:
                if not isinstance(update, dict):
                    continue
                year = update.get("year")
                value = update.get("value")
                if year in (None, "") or value is None:
                    continue
                flattened.append(
                    {
                        "category_name": label,
                        "years": [str(year)],
                        "values": [value],
                    }
                )
            continue
        points = change.get("points")
        if isinstance(points, list) and points:
            for point in points:
                if not isinstance(point, dict):
                    continue
                year = point.get("year")
                value = point.get("value")
                if year in (None, "") or value is None:
                    continue
                flattened.append(
                    {
                        "category_name": label,
                        "years": [str(year)],
                        "values": [value],
                    }
                )
            continue
        years = change.get("years")
        values = change.get("values")
        if isinstance(years, list) and isinstance(values, list) and years and values:
            for year, value in zip(years, values):
                flattened.append(
                    {
                        "category_name": label,
                        "years": [str(year)],
                        "values": [value],
                    }
                )
            continue
        year = change.get("year")
        value = change.get("value")
        if year not in (None, "") and value is not None:
            flattened.append(
                {
                    "category_name": label,
                    "years": [str(year)],
                    "values": [value],
                }
            )
    return flattened


def _atomic_changes_from_payload(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    change_root = payload.get("change") if isinstance(payload.get("change"), dict) else payload
    if not isinstance(change_root, dict):
        return []

    changes = change_root.get("changes")
    if isinstance(changes, list):
        return _flatten_atomic_changes(changes)

    if any(key in change_root for key in ("updates", "points", "years", "values", "year", "value")):
        return _flatten_atomic_changes([change_root])
    return []


def _atomic_changes_from_step(operation_target: Any, data_change: Any) -> list[dict[str, Any]]:
    target = operation_target if isinstance(operation_target, dict) else {}
    change = data_change if isinstance(data_change, dict) else {}

    atomic_changes = _atomic_changes_from_payload(change)
    if atomic_changes:
        fallback_label = _first_non_empty_string(
            target.get("category_name"),
            target.get("category"),
        )
        if fallback_label:
            normalized: list[dict[str, Any]] = []
            for item in atomic_changes:
                if not isinstance(item, dict):
                    continue
                candidate = dict(item)
                if not _first_non_empty_string(candidate.get("category_name"), candidate.get("category")):
                    candidate["category_name"] = fallback_label
                normalized.append(candidate)
            return normalized
        return atomic_changes

    label = _first_non_empty_string(
        change.get("category_name"),
        change.get("category"),
        target.get("category_name"),
        target.get("category"),
    )
    years = change.get("years")
    values = change.get("values")
    year = change.get("year")
    value = change.get("value")

    if not isinstance(years, list):
        years = target.get("years")
    if not isinstance(values, list):
        values = target.get("values")
    if year in (None, ""):
        year = target.get("year")
    if value is None:
        value = target.get("value")

    candidate: dict[str, Any] = {}
    if label:
        candidate["category_name"] = label
    if isinstance(years, list) and isinstance(values, list) and years and values:
        candidate["years"] = years
        candidate["values"] = values
        return _flatten_atomic_changes([candidate])
    if year not in (None, "") and value is not None:
        candidate["year"] = year
        candidate["value"] = value
        return _flatten_atomic_changes([candidate])
    return []


def _expand_composite_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for raw_step in steps:
        if not isinstance(raw_step, dict):
            continue
        step = dict(raw_step)
        operation = str(step.get("operation") or "").strip().lower()
        operation_target = step.get("operation_target")
        data_change = step.get("data_change")
        if not isinstance(operation_target, dict):
            operation_target = {}
        if not isinstance(data_change, dict):
            data_change = {}

        if operation == "delete":
            labels = _structured_delete_labels(operation_target, data_change)
            if len(labels) > 1:
                for label in labels:
                    split_step = dict(step)
                    split_step["operation_target"] = {"category_name": label}
                    expanded.append(split_step)
                continue

        if operation == "change":
            atomic_changes = _atomic_changes_from_step(operation_target, data_change)
            if len(atomic_changes) > 1:
                for change in atomic_changes:
                    split_step = dict(step)
                    label = str(change.get("category_name") or "").strip()
                    split_step["operation_target"] = {"category_name": label} if label else {}
                    split_step["data_change"] = {"changes": [change]}
                    expanded.append(split_step)
                continue
            if len(atomic_changes) == 1:
                split_step = dict(step)
                label = str(atomic_changes[0].get("category_name") or "").strip()
                split_step["operation_target"] = {"category_name": label} if label else operation_target
                split_step["data_change"] = {"changes": atomic_changes}
                expanded.append(split_step)
                continue

        expanded.append(step)
    return expanded


def _step_component_type(step: dict[str, Any]) -> str:
    target = step.get("operation_target")
    if not isinstance(target, dict):
        return ""
    value = target.get("element_type") or target.get("type")
    return str(value or "").strip().lower()


def _prune_redundant_component_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    series_add_labels: set[str] = set()

    for step in steps:
        if not isinstance(step, dict):
            continue
        op = str(step.get("operation") or "").strip().lower()
        label = _first_non_empty_string((step.get("operation_target") or {}).get("category_name"))
        component_type = _step_component_type(step)

        if op == "add":
            is_legend_only = component_type == "legend_item"
            is_series_add = component_type in {"", "area", "line", "series", "line_and_legend"}

            if is_legend_only and label and label in series_add_labels:
                continue
            if is_series_add and label:
                series_add_labels.add(label)

        kept.append(step)

    return kept


def _points_from_data_change(data_change: dict[str, Any]) -> list[dict[str, float]]:
    points = None
    if isinstance(data_change, dict):
        add_change = data_change.get("add")
        if isinstance(add_change, dict):
            points = add_change.get("points")
        if points is None:
            points = data_change.get("points")
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

    color_sources: list[dict[str, Any]] = []
    add_change = data_change.get("add")
    if isinstance(add_change, dict):
        color_sources.append(add_change)
    color_sources.append(data_change)

    for source in color_sources:
        for key in ("point_color", "color", "fill", "rgb"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    points = None
    if isinstance(add_change, dict) and isinstance(add_change.get("points"), list):
        points = add_change.get("points")
    if points is None:
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
        label_value = operation_target.get("category_name")
        if isinstance(label_value, list):
            labels = [str(item).strip() for item in label_value if str(item).strip()]
            if len(labels) > 1:
                return "; ".join(f'Delete the category/series "{label}"' for label in labels)
            label = labels[0] if labels else ""
        else:
            label = str(label_value or "").strip()
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
                points = change.get("points")
                if isinstance(points, list):
                    for point in points:
                        if not isinstance(point, dict):
                            continue
                        year = point.get("year")
                        value = point.get("value")
                        if label and year not in (None, "") and value is not None:
                            clauses.append(f'Change "{label}" in {year} to {value}')
                    continue
                year = change.get("year")
                value = change.get("value")
                if label and year not in (None, "") and value is not None:
                    clauses.append(f'Change "{label}" in {year} to {value}')
                    continue
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


def _step_paths(
    svg_path: str,
    chart_type: str,
    idx: int,
    total: int,
    *,
    render_output_dir: str | None = None,
) -> tuple[str | None, str | None]:
    final_svg, final_png = default_output_paths(svg_path, chart_type)
    if render_output_dir:
        base_dir = Path(render_output_dir)
        final_svg = str(base_dir / Path(final_svg).name)
        final_png = str(base_dir / Path(final_png).name)
    if idx == total - 1:
        return final_svg, final_png
    stem_svg = Path(final_svg).stem
    stem_png = Path(final_png).stem
    step_svg = str(Path(final_svg).with_name(f"{stem_svg}_step{idx+1}.svg"))
    step_png = str(Path(final_png).with_name(f"{stem_png}_step{idx+1}.png"))
    return step_svg, step_png


def _validation_context_from_mapping(mapping_info: Any) -> dict[str, Any]:
    if not isinstance(mapping_info, dict):
        return {}
    context: dict[str, Any] = {}
    for key in ("x_ticks", "y_ticks"):
        value = mapping_info.get(key)
        if isinstance(value, list):
            context[key] = value
    return context


def _build_failure_info(
    *,
    failure_type: str,
    failure_stage: str,
    retryable: bool,
    retry_hint: str,
    replan_strategy: str,
    message: str,
    exception_type: str = "",
    chart_type: str = "",
    operation: str = "",
) -> dict[str, Any]:
    return {
        "failure_type": str(failure_type or "").strip() or "unknown_failure",
        "failure_stage": str(failure_stage or "").strip() or "unknown_stage",
        "retryable": bool(retryable),
        "retry_hint": str(retry_hint or message or "").strip(),
        "replan_strategy": str(replan_strategy or "").strip() or "stop_attempt",
        "message": str(message or "").strip(),
        "exception_type": str(exception_type or "").strip(),
        "chart_type": str(chart_type or "").strip(),
        "operation": str(operation or "").strip(),
    }


def _classify_step_failure(
    *,
    exc: Exception,
    chart_type: str,
    step: dict[str, Any],
) -> dict[str, Any]:
    message = str(exc).strip() or exc.__class__.__name__
    lowered = message.lower()
    failure_type = "execution_error"
    retryable = True
    replan_strategy = "replan_current_step"

    if "unsupported chart type" in lowered:
        failure_type = "unsupported_chart_type"
        retryable = False
        replan_strategy = "stop_attempt"
    elif "cannot remove the only remaining area series" in lowered:
        failure_type = "operation_conflict"
        retryable = False
        replan_strategy = "stop_attempt"
    elif "svg missing" in lowered or "file does not exist" in lowered:
        failure_type = "input_missing"
        retryable = False
        replan_strategy = "restart_attempt"
    elif (
        "axes group not found" in lowered
        or "axes_1 missing" in lowered
        or "stacked area collections found" in lowered
        or "update group missing" in lowered
    ):
        failure_type = "chart_structure_missing"
        retryable = False
        replan_strategy = "restart_attempt"
    elif "insufficient" in lowered and ("ticks" in lowered or "mapping" in lowered or "x grid" in lowered):
        failure_type = "mapping_incomplete"
        retryable = False
        replan_strategy = "restart_attempt"
    elif (
        "no matching" in lowered
        or "target missing" in lowered
        or "series missing" in lowered
        or "legend color" in lowered
        or "path not found" in lowered
        or "matches selected" in lowered
    ):
        failure_type = "target_not_found"
        retryable = True
        replan_strategy = "replan_current_step"
    elif (
        "scatter points missing" in lowered
        or "no valid" in lowered
        or "cannot parse" in lowered
        or "no new points" in lowered
        or "no new series values" in lowered
    ):
        failure_type = "step_data_invalid"
        retryable = True
        replan_strategy = "replan_current_step"

    return _build_failure_info(
        failure_type=failure_type,
        failure_stage="step_execute",
        retryable=retryable,
        retry_hint=message,
        replan_strategy=replan_strategy,
        message=message,
        exception_type=type(exc).__name__,
        chart_type=chart_type,
        operation=str(step.get("operation") or ""),
    )


def _classify_render_failure(render_check: dict[str, Any]) -> dict[str, Any]:
    issues = [str(issue).strip() for issue in render_check.get("issues", []) if str(issue).strip()]
    retry_hint = "; ".join(issues[:4]) or "render validation failed"
    failure_type = "render_validation_failed"
    if any("output image not found" in issue.lower() or "no output image path" in issue.lower() for issue in issues):
        failure_type = "render_output_missing"
    elif any("image appears empty" in issue.lower() for issue in issues):
        failure_type = "render_output_empty"
    elif any("programmatic:" in issue.lower() for issue in issues):
        failure_type = "render_programmatic_mismatch"
    return _build_failure_info(
        failure_type=failure_type,
        failure_stage="render_validate",
        retryable=True,
        retry_hint=retry_hint,
        replan_strategy="restart_attempt",
        message=retry_hint,
    )


def _classify_plan_failure(reason: str) -> dict[str, Any]:
    message = str(reason or "").strip() or "llm planning failed"
    return _build_failure_info(
        failure_type="planning_failed",
        failure_stage="plan",
        retryable=True,
        retry_hint=message,
        replan_strategy="restart_attempt",
        message=message,
    )


def _replan_current_step(
    *,
    step: dict[str, Any],
    chart_type: str,
    update_mode: str = "rules",
    llm: Any,
    retry_hint: str,
) -> dict[str, Any]:
    step_question = _render_structured_step_question(step)
    step_context: dict[str, Any] = {}
    operation_target = step.get("operation_target")
    data_change = step.get("data_change")
    if isinstance(operation_target, dict) and operation_target:
        step_context["operation_target"] = dict(operation_target)
    if isinstance(data_change, dict) and data_change:
        step_context["data_change"] = dict(data_change)

    operation_plan = _llm_plan_update(step_question, chart_type, llm, retry_hint=retry_hint)
    replanned_steps = _operation_steps_from_plan(
        operation_plan,
        step_question,
        step_context,
        chart_type=chart_type,
        update_mode=update_mode,
    )
    if not replanned_steps:
        return dict(step)

    candidate = replanned_steps[0]
    replanned = dict(step)
    op = str(candidate.get("operation") or "").strip().lower()
    if op and op != "unknown":
        replanned["operation"] = op

    question = str(candidate.get("question") or "").strip()
    if question:
        replanned["question"] = question
    question_hint = str(candidate.get("question_hint") or question).strip()
    if question_hint:
        replanned["question_hint"] = question_hint

    replanned["operation_target"] = _merge_dict_like(step.get("operation_target"), candidate.get("operation_target"))
    replanned["data_change"] = _merge_dict_like(step.get("data_change"), candidate.get("data_change"))
    replanned["new_points"] = _coerce_points(candidate.get("new_points", step.get("new_points", [])))
    return replanned


def _execute_planned_steps(
    *,
    inputs: dict[str, Any],
    planned_question: str,
    operation_plan: dict[str, Any],
    structured_context: dict[str, Any],
    chart_type: str,
    render_output_dir: str | None = None,
    update_mode: str = "rules",
    llm: Any,
    planner_llm: Any,
    used_scatter_points: list[dict[str, float]],
    event_callback: Any | None = None,
) -> tuple[str | None, list[dict[str, Any]], list[dict[str, Any]], Any, list[dict[str, float]]]:
    steps = _operation_steps_from_plan(
        operation_plan,
        planned_question,
        structured_context,
        chart_type=chart_type,
        update_mode=update_mode,
    )
    step_logs: list[dict[str, Any]] = []
    perception_steps: list[dict[str, Any]] = []
    current_svg = str(inputs["svg_path"])
    last_state = None
    output_image = None
    scatter_points = list(used_scatter_points)

    for idx, original_step in enumerate(steps):
        step = dict(original_step)
        step_retry_history: list[dict[str, Any]] = []
        for step_attempt in range(1, STEP_REPLAN_RETRIES + 2):
            step_q = _render_structured_step_question(step)
            _emit_event(
                event_callback,
                "step_started",
                {
                    "step": {
                        "index": idx + 1,
                        "step_attempt": step_attempt,
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
            current_perception_step = {
                "index": idx + 1,
                "step_attempt": step_attempt,
                "operation": step.get("operation"),
                "question": step_q,
                "question_hint": step.get("question_hint"),
                "operation_target": step.get("operation_target"),
                "data_change": step.get("data_change"),
                "perception": _sanitize_perception(state.perception),
            }
            mapping_info = state.perception.get("mapping_info", {})
            update_spec = state.perception.get("update_spec", {})
            if isinstance(mapping_info, dict) and isinstance(update_spec, dict):
                mapping_info = dict(mapping_info)
                mapping_info["requested_point_color"] = _extract_scatter_requested_color(
                    step=step,
                    update_spec=update_spec,
                )
            step_svg, step_png = _step_paths(
                str(inputs["svg_path"]),
                chart_type,
                idx,
                len(steps),
                render_output_dir=render_output_dir,
            )
            next_scatter_points = list(scatter_points)

            try:
                if chart_type == "scatter":
                    points = _coerce_points(step.get("new_points", []))
                    if not points:
                        points = _coerce_points(operation_plan.get("new_points", []))
                    if not points:
                        points = _points_from_data_change(step.get("data_change", {}))
                    if not points:
                        raise ValueError(f"step {idx+1}: scatter points missing")
                    next_scatter_points = points
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
                        operation_target=step.get("operation_target"),
                        data_change=step.get("data_change"),
                        operation=str(step.get("operation") or ""),
                    )
                elif chart_type == "area":
                    output_image = update_area_svg(
                        current_svg,
                        step_q,
                        mapping_info,
                        output_path=step_png,
                        svg_output_path=step_svg,
                        llm=llm,
                        operation_target=step.get("operation_target"),
                        data_change=step.get("data_change"),
                    )
                else:
                    raise ValueError(f"unsupported chart type: {chart_type}")
            except Exception as exc:
                failure_info = _classify_step_failure(exc=exc, chart_type=chart_type, step=step)
                failed_step = dict(current_perception_step)
                failed_step["input_svg_path"] = step_inputs["svg_path"]
                failed_step["failure_info"] = failure_info
                if step_retry_history:
                    failed_step["retry_history"] = list(step_retry_history)
                if step_attempt >= STEP_REPLAN_RETRIES + 1 or not failure_info.get("retryable"):
                    raise _StepExecutionError(
                        f"step {idx+1}: {exc}",
                        step_logs=step_logs,
                        perception_steps=perception_steps,
                        last_state=last_state,
                        scatter_points=scatter_points,
                        failure_info=failure_info,
                        failed_step=failed_step,
                    ) from exc
                step_retry_history.append(
                    {
                        "step_attempt": step_attempt,
                        "question": step_q,
                        "failure_info": failure_info,
                    }
                )
                step = _replan_current_step(
                    step=step,
                    chart_type=chart_type,
                    update_mode=update_mode,
                    llm=planner_llm,
                    retry_hint=str(failure_info.get("retry_hint") or exc),
                )
                continue

            current_svg = step_svg or current_svg
            scatter_points = next_scatter_points
            step_logs.append(
                {
                    "index": idx + 1,
                    "step_attempt": step_attempt,
                    "operation": step.get("operation"),
                    "question": step_q,
                    "input_svg_path": step_inputs["svg_path"],
                    "operation_target": step.get("operation_target"),
                    "data_change": step.get("data_change"),
                    "new_points": _coerce_points(
                        scatter_points if chart_type == "scatter" else step.get("new_points", [])
                    ),
                    "validation_context": _validation_context_from_mapping(mapping_info),
                    "output_svg_path": step_svg,
                    "output_image_path": output_image,
                    "retry_history": list(step_retry_history),
                }
            )
            if step_retry_history:
                current_perception_step["retry_history"] = list(step_retry_history)
            perception_steps.append(current_perception_step)
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
            break

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
            chart_type=str(output.get("answer_input", {}).get("chart_type") or ""),
            output_image_path=augmented_path,
            answer_stage="tool_augmented",
            image_context_note=ANSWER_TOOL_AUGMENTED_IMAGE_CONTEXT_PROMPT,
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
    effective_update_spec = _effective_update_spec_for_render(
        chart_type=chart_type,
        update_spec=update_spec,
        step_logs=step_logs,
    )
    basic_check = validate_render(output_image, chart_type, effective_update_spec, llm=None)
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

    return validate_render(output_image, chart_type, effective_update_spec, llm=llm)


def _effective_update_spec_for_render(
    *,
    chart_type: str,
    update_spec: dict[str, Any],
    step_logs: list[dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(update_spec, dict):
        update_spec = {}
    if str(chart_type or "").strip().lower() != "scatter":
        return update_spec

    for step in reversed(step_logs):
        if not isinstance(step, dict):
            continue
        points = _coerce_points(step.get("new_points", []))
        if points:
            merged = dict(update_spec)
            merged["new_points"] = points
            return merged
    return update_spec


def _programmatic_validate(
    chart_type: str,
    step_logs: list[dict[str, Any]],
    llm: Any,
    svg_perception_mode: str | None = None,
) -> dict[str, Any] | None:
    del llm, svg_perception_mode
    chart = str(chart_type or "").strip().lower()
    if chart == "area":
        return _programmatic_validate_area_steps(step_logs)
    if chart == "line":
        return _programmatic_validate_line_steps(step_logs)
    if chart == "scatter":
        return _programmatic_validate_scatter_steps(step_logs)
    return None


def _programmatic_validate_area_steps(step_logs: list[dict[str, Any]]) -> dict[str, Any] | None:
    results: list[dict[str, Any]] = []
    for step in step_logs:
        if not isinstance(step, dict):
            continue
        step_op = str(step.get("operation") or "").strip().lower()
        question = str(step.get("question") or "")
        if step_op == "change" or _looks_like_area_change_question(question):
            results.append(_validate_area_change_step(step))
        elif step_op == "delete":
            results.append(_validate_area_delete_step(step))
    return _combine_programmatic_results(results)


def _programmatic_validate_line_steps(step_logs: list[dict[str, Any]]) -> dict[str, Any] | None:
    results: list[dict[str, Any]] = []
    for step in step_logs:
        if not isinstance(step, dict):
            continue
        step_op = str(step.get("operation") or "").strip().lower()
        question = str(step.get("question") or "")
        if step_op == "change" or _looks_like_line_change_question(question):
            results.append(_validate_line_change_step(step))
        elif step_op == "delete":
            results.append(_validate_line_delete_step(step))
    return _combine_programmatic_results(results)


def _programmatic_validate_scatter_steps(step_logs: list[dict[str, Any]]) -> dict[str, Any] | None:
    results: list[dict[str, Any]] = []
    for step in step_logs:
        if not isinstance(step, dict):
            continue
        step_op = str(step.get("operation") or "").strip().lower()
        points = _coerce_points(step.get("new_points", []))
        if step_op == "add" and points:
            results.append(_validate_scatter_add_step(step))
    return _combine_programmatic_results(results)


def _combine_programmatic_results(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    filtered = [result for result in results if isinstance(result, dict)]
    if not filtered:
        return None
    for result in filtered:
        if not result.get("ok"):
            return result
    confidence = min(float(result.get("confidence", 0.99)) for result in filtered)
    return {"ok": True, "confidence": confidence, "issues": []}


def _load_svg_document(svg_path: str) -> tuple[ET.Element, str]:
    with open(svg_path, "r", encoding="utf-8") as handle:
        content = handle.read()
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    root = ET.parse(svg_path, parser=parser).getroot()
    return root, content


def _validation_ticks(step: dict[str, Any]) -> tuple[list[Any], list[Any]]:
    context = step.get("validation_context")
    if not isinstance(context, dict):
        return [], []
    x_ticks = context.get("x_ticks") if isinstance(context.get("x_ticks"), list) else []
    y_ticks = context.get("y_ticks") if isinstance(context.get("y_ticks"), list) else []
    return x_ticks, y_ticks


def _step_atomic_changes(step: dict[str, Any]) -> list[tuple[str, float, float]]:
    operation_target = step.get("operation_target")
    data_change = step.get("data_change")
    atomic = _atomic_changes_from_step(operation_target, data_change)
    out: list[tuple[str, float, float]] = []
    for change in atomic:
        if not isinstance(change, dict):
            continue
        label = str(change.get("category_name") or "").strip()
        years = change.get("years")
        values = change.get("values")
        if not label or not isinstance(years, list) or not isinstance(values, list):
            continue
        for year, value in zip(years, values):
            try:
                out.append((label, float(year), float(value)))
            except (TypeError, ValueError):
                continue
    return out


def _validation_tolerance(expected_value: float) -> float:
    return max(1e-3, abs(expected_value) * 1e-3)


def _validate_area_change_step(step: dict[str, Any]) -> dict[str, Any]:
    svg_path = str(step.get("output_svg_path") or "").strip()
    question = str(step.get("question") or "").strip()
    if not svg_path or not os.path.exists(svg_path):
        return {"ok": False, "confidence": 0.0, "issues": [f"programmatic: svg not found: {svg_path}"]}
    x_ticks, y_ticks = _validation_ticks(step)
    if len(x_ticks) < 2 or len(y_ticks) < 2:
        return {"ok": False, "confidence": 0.0, "issues": ["programmatic: insufficient axis ticks"]}

    try:
        root, content = _load_svg_document(svg_path)
        axes = root.find(f'.//{{{area_ops.SVG_NS}}}g[@id="axes_1"]')
        if axes is None:
            return {"ok": False, "confidence": 0.0, "issues": ["programmatic: axes_1 missing"]}
        _, legend_items = area_ops._extract_legend_items(root, content)
        labels = [item["label"] for item in legend_items if item.get("label")]
        preserved = _validate_change_step_preserved_labels(
            step=step,
            output_labels=labels,
            chart_name="area",
            extract_legend_items=area_ops._extract_legend_items,
        )
        if preserved is not None:
            return preserved
        parsed_changes = _step_atomic_changes(step)
        if not parsed_changes and question:
            parsed = area_ops._parse_year_value_update(question, labels, llm=None)
            if parsed is not None:
                parsed_changes = [(parsed[0], parsed[1], parsed[2])]
        if not parsed_changes:
            return {"ok": False, "confidence": 0.0, "issues": ["programmatic: cannot parse change request"]}

        areas = area_ops._extract_area_groups(axes)
        x_values, series_values = area_ops._area_series_values(areas, y_ticks)
        for label, year_value, expected_value in parsed_changes:
            resolved_label = area_ops._resolve_matching_label(label, labels) or label
            target_idx = area_ops._find_area_index(resolved_label, legend_items, areas)
            if target_idx is None:
                return {
                    "ok": False,
                    "confidence": 0.0,
                    "issues": [f"programmatic: area series missing for {resolved_label}"],
                }
            target_x = area_ops._data_to_pixel(year_value, x_ticks)
            year_idx = min(range(len(x_values)), key=lambda i: abs(x_values[i] - target_x))
            actual_value = float(series_values[target_idx][year_idx])
            if abs(actual_value - expected_value) > _validation_tolerance(expected_value):
                return {
                    "ok": False,
                    "confidence": 0.05,
                    "issues": [
                        f"programmatic: expected {resolved_label}@{int(year_value)}={expected_value}, got {actual_value}",
                    ],
                }
        return {"ok": True, "confidence": 0.99, "issues": []}
    except Exception as exc:
        return {"ok": False, "confidence": 0.0, "issues": [f"programmatic exception: {exc}"]}


def _validate_line_change_step(step: dict[str, Any]) -> dict[str, Any]:
    svg_path = str(step.get("output_svg_path") or "").strip()
    question = str(step.get("question") or "").strip()
    if not svg_path or not os.path.exists(svg_path):
        return {"ok": False, "confidence": 0.0, "issues": [f"programmatic: svg not found: {svg_path}"]}
    x_ticks, y_ticks = _validation_ticks(step)
    if len(x_ticks) < 2 or len(y_ticks) < 2:
        return {"ok": False, "confidence": 0.0, "issues": ["programmatic: insufficient axis ticks"]}

    try:
        root, content = _load_svg_document(svg_path)
        axes = root.find(f'.//{{{line_ops.SVG_NS}}}g[@id="axes_1"]')
        if axes is None:
            return {"ok": False, "confidence": 0.0, "issues": ["programmatic: axes_1 missing"]}
        _, legend_items = line_ops._extract_legend_items(root, content)
        labels = [item["label"] for item in legend_items if item.get("label")]
        preserved = _validate_change_step_preserved_labels(
            step=step,
            output_labels=labels,
            chart_name="line",
            extract_legend_items=line_ops._extract_legend_items,
        )
        if preserved is not None:
            return preserved
        parsed_changes = _step_atomic_changes(step)
        if not parsed_changes and question:
            parsed = line_ops._parse_year_value_update(question, labels, llm=None)
            if parsed is not None:
                parsed_changes = [(parsed[0], parsed[1], parsed[2])]
        if not parsed_changes:
            return {"ok": False, "confidence": 0.0, "issues": ["programmatic: cannot parse change request"]}

        for label, year_value, expected_value in parsed_changes:
            resolved_label = line_ops._resolve_matching_label(label, labels) or label
            target_stroke = None
            for item in legend_items:
                if line_ops._labels_match(str(item.get("label") or ""), resolved_label):
                    target_stroke = item.get("stroke")
                    break
            line_group = line_ops._find_line_by_stroke(axes, target_stroke) if target_stroke else None
            if line_group is None:
                line_group = line_ops._find_line_by_legend_index(axes, legend_items, resolved_label)
            if line_group is None:
                return {
                    "ok": False,
                    "confidence": 0.0,
                    "issues": [f"programmatic: line series missing for {resolved_label}"],
                }
            line_path = line_group.find(f'./{{{line_ops.SVG_NS}}}path')
            if line_path is None:
                return {"ok": False, "confidence": 0.0, "issues": ["programmatic: line path missing"]}
            points = line_ops._extract_path_points(line_path.get("d", ""))
            if not points:
                return {"ok": False, "confidence": 0.0, "issues": ["programmatic: line path empty"]}
            target_x = line_ops._data_to_pixel(year_value, x_ticks)
            point_idx = min(range(len(points)), key=lambda i: abs(points[i][0] - target_x))
            actual_value = float(line_ops._pixel_to_data(points[point_idx][1], y_ticks))
            if abs(actual_value - expected_value) > _validation_tolerance(expected_value):
                return {
                    "ok": False,
                    "confidence": 0.05,
                    "issues": [
                        f"programmatic: expected {resolved_label}@{int(year_value)}={expected_value}, got {actual_value}",
                    ],
                }
        return {"ok": True, "confidence": 0.99, "issues": []}
    except Exception as exc:
        return {"ok": False, "confidence": 0.0, "issues": [f"programmatic exception: {exc}"]}


def _validate_area_delete_step(step: dict[str, Any]) -> dict[str, Any]:
    return _validate_delete_step(
        step=step,
        svg_ns=area_ops.SVG_NS,
        extract_labels=area_ops._extract_structured_delete_labels,
        match_labels=area_ops._match_labels,
        resolve_label=area_ops._resolve_matching_label,
        extract_legend_items=area_ops._extract_legend_items,
        chart_name="area",
    )


def _validate_line_delete_step(step: dict[str, Any]) -> dict[str, Any]:
    return _validate_delete_step(
        step=step,
        svg_ns=line_ops.SVG_NS,
        extract_labels=line_ops._extract_structured_line_delete_labels,
        match_labels=line_ops._match_labels,
        resolve_label=line_ops._resolve_matching_label,
        extract_legend_items=line_ops._extract_legend_items,
        chart_name="line",
    )


def _validate_delete_step(
    *,
    step: dict[str, Any],
    svg_ns: str,
    extract_labels: Any,
    match_labels: Any,
    resolve_label: Any,
    extract_legend_items: Any,
    chart_name: str,
) -> dict[str, Any]:
    input_svg = str(step.get("input_svg_path") or "").strip()
    output_svg = str(step.get("output_svg_path") or "").strip()
    question = str(step.get("question") or "").strip()
    if not input_svg or not os.path.exists(input_svg):
        return {"ok": False, "confidence": 0.0, "issues": [f"programmatic: input svg not found: {input_svg}"]}
    if not output_svg or not os.path.exists(output_svg):
        return {"ok": False, "confidence": 0.0, "issues": [f"programmatic: svg not found: {output_svg}"]}

    try:
        input_root, input_content = _load_svg_document(input_svg)
        output_root, output_content = _load_svg_document(output_svg)
        _ = input_root.find(f'.//{{{svg_ns}}}g[@id="axes_1"]')
        _ = output_root.find(f'.//{{{svg_ns}}}g[@id="axes_1"]')
        _, input_items = extract_legend_items(input_root, input_content)
        _, output_items = extract_legend_items(output_root, output_content)
        input_labels = [item["label"] for item in input_items if item.get("label")]
        output_labels = [item["label"] for item in output_items if item.get("label")]

        labels = extract_labels(step.get("operation_target"), step.get("data_change"))
        if not labels and question:
            labels = match_labels(question, input_labels)
        if not labels:
            return {"ok": False, "confidence": 0.0, "issues": [f"programmatic: cannot parse {chart_name} delete request"]}

        resolved_deleted: list[str] = []
        for label in labels:
            resolved = resolve_label(label, input_labels) or label
            if resolved not in input_labels:
                return {
                    "ok": False,
                    "confidence": 0.0,
                    "issues": [f"programmatic: delete target missing in original legend: {resolved}"],
                }
            if resolved in output_labels:
                return {
                    "ok": False,
                    "confidence": 0.05,
                    "issues": [f"programmatic: delete target still present after update: {resolved}"],
                }
            resolved_deleted.append(resolved)
        for label in input_labels:
            if label in resolved_deleted:
                continue
            if label not in output_labels:
                return {
                    "ok": False,
                    "confidence": 0.05,
                    "issues": [f"programmatic: non-target legend item missing after update: {label}"],
                }
        return {"ok": True, "confidence": 0.99, "issues": []}
    except Exception as exc:
        return {"ok": False, "confidence": 0.0, "issues": [f"programmatic exception: {exc}"]}


def _validate_change_step_preserved_labels(
    *,
    step: dict[str, Any],
    output_labels: list[str],
    chart_name: str,
    extract_legend_items: Any,
) -> dict[str, Any] | None:
    input_svg = str(step.get("input_svg_path") or "").strip()
    if not input_svg or not os.path.exists(input_svg):
        return None
    input_root, input_content = _load_svg_document(input_svg)
    _, input_items = extract_legend_items(input_root, input_content)
    input_labels = [item["label"] for item in input_items if item.get("label")]
    extra_labels = [label for label in output_labels if label not in input_labels]
    if extra_labels:
        return {
            "ok": False,
            "confidence": 0.05,
            "issues": [f"programmatic: unexpected {chart_name} legend item added: {extra_labels[0]}"],
        }
    missing_labels = [label for label in input_labels if label not in output_labels]
    if missing_labels:
        return {
            "ok": False,
            "confidence": 0.05,
            "issues": [f"programmatic: expected {chart_name} legend item missing after change: {missing_labels[0]}"],
        }
    return None


def _validate_scatter_add_step(step: dict[str, Any]) -> dict[str, Any]:
    svg_path = str(step.get("output_svg_path") or "").strip()
    if not svg_path or not os.path.exists(svg_path):
        return {"ok": False, "confidence": 0.0, "issues": [f"programmatic: svg not found: {svg_path}"]}
    expected_points = _coerce_points(step.get("new_points", []))
    if not expected_points:
        return {"ok": False, "confidence": 0.0, "issues": ["programmatic: scatter points missing"]}
    x_ticks, y_ticks = _validation_ticks(step)
    if len(x_ticks) < 2 or len(y_ticks) < 2:
        return {"ok": False, "confidence": 0.0, "issues": ["programmatic: insufficient axis ticks"]}

    try:
        root, _content = _load_svg_document(svg_path)
        axes = root.find(f'.//{{{scatter_ops.SVG_NS}}}g[@id="axes_1"]')
        if axes is None:
            return {"ok": False, "confidence": 0.0, "issues": ["programmatic: axes_1 missing"]}
        update_group = axes.find(f'.//{{{scatter_ops.SVG_NS}}}g[@id="PathCollection_update"]')
        if update_group is None:
            return {"ok": False, "confidence": 0.0, "issues": ["programmatic: scatter update group missing"]}

        rendered_points: list[tuple[float, float]] = []
        for circle in update_group.findall(f'.//{{{scatter_ops.SVG_NS}}}circle'):
            try:
                rendered_points.append((float(circle.get("cx", "nan")), float(circle.get("cy", "nan"))))
            except ValueError:
                continue
        for use in update_group.findall(f'.//{{{scatter_ops.SVG_NS}}}use'):
            try:
                rendered_points.append((float(use.get("x", "nan")), float(use.get("y", "nan"))))
            except ValueError:
                continue
        if len(rendered_points) != len(expected_points):
            return {
                "ok": False,
                "confidence": 0.05,
                "issues": [f"programmatic: expected {len(expected_points)} scatter points, got {len(rendered_points)}"],
            }

        unmatched = list(rendered_points)
        tolerance = 1.5
        for point in expected_points:
            expected_x = float(scatter_ops._interpolate_axis(float(point["x"]), x_ticks))
            expected_y = float(scatter_ops._interpolate_axis(float(point["y"]), y_ticks))
            match_idx = None
            for idx, (actual_x, actual_y) in enumerate(unmatched):
                if abs(actual_x - expected_x) <= tolerance and abs(actual_y - expected_y) <= tolerance:
                    match_idx = idx
                    break
            if match_idx is None:
                return {
                    "ok": False,
                    "confidence": 0.05,
                    "issues": [f"programmatic: scatter point missing near ({point['x']}, {point['y']})"],
                }
            unmatched.pop(match_idx)
        return {"ok": True, "confidence": 0.99, "issues": []}
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


def _looks_like_line_change_question(question: str) -> bool:
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
