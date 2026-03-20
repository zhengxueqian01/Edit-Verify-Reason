from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from chart_agent.config import get_perception_max_retries, get_task_model_config
from chart_agent.core.perception_actions import (
    DETECT_INPUT_MODE,
    PARSE_QUESTION,
    PERCEIVE_IMAGE_SVG,
    PERCEIVE_TEXT,
    RETRY,
    SANITY_CHECK,
    STOP,
)
from chart_agent.core.perception_policy import perception_policy
from chart_agent.core.state import Budget, GraphState
from chart_agent.core.trace import append_trace
from chart_agent.llm_factory import make_llm
from chart_agent.perception.chart_type_selector import select_chart_type
from chart_agent.perception.question_parser import parse_question
from chart_agent.perception.sanity_check import run_sanity_check
from chart_agent.perception.svg_perceiver import perceive_svg
from chart_agent.perception.text_perceiver import perceive_text


def run_perception(inputs: dict[str, Any]) -> GraphState:
    state = GraphState(inputs=inputs, budget=Budget(max_retries=get_perception_max_retries()))
    llm = make_llm(get_task_model_config("router"))

    graph = _build_graph(llm)
    result = graph.invoke(state)
    final_state = _coerce_state(result)
    _finalize_perception(final_state)
    return final_state


def _build_graph(llm: Any) -> Any:
    graph = StateGraph(GraphState)

    def policy_node(state: GraphState) -> GraphState:
        action = perception_policy(state)
        state.current_action = action
        append_trace(
            state.trace,
            node="policy",
            action=action,
            rationale="Select next perception action based on state.",
            inputs_summary={"retry_count": state.budget.retry_count},
            outputs_summary={"next_action": action},
        )
        return state

    def execute_node(state: GraphState) -> GraphState:
        action = state.current_action
        if action is None:
            return state

        if action == DETECT_INPUT_MODE:
            _detect_input_mode(state)
        elif action == PARSE_QUESTION:
            _parse_question(state, llm)
        elif action == PERCEIVE_IMAGE_SVG:
            _perceive_image_svg(state, llm)
        elif action == PERCEIVE_TEXT:
            _perceive_text(state, llm)
        elif action == SANITY_CHECK:
            _sanity_check(state)
        elif action == RETRY:
            _apply_retry(state)
        elif action == STOP:
            pass
        else:
            state.errors.last_error = f"Unknown action: {action}"
            state.errors.error_history.append(state.errors.last_error)

        return state

    graph.add_node("policy", policy_node)
    graph.add_node("execute", execute_node)

    graph.set_entry_point("policy")
    graph.add_edge("policy", "execute")

    def route(state: GraphState) -> str:
        if state.current_action == STOP:
            return END
        return "policy"

    graph.add_conditional_edges("execute", route)

    return graph.compile()


def _detect_input_mode(state: GraphState) -> None:
    inputs = state.inputs
    image_path = inputs.get("image_path")
    svg_path = inputs.get("svg_path")
    text_spec = inputs.get("text_spec")
    if image_path or svg_path:
        input_mode = "image"
    elif text_spec:
        input_mode = "text"
    else:
        input_mode = "text"
    state.perception["input_mode"] = input_mode
    append_trace(
        state.trace,
        node="execute",
        action=DETECT_INPUT_MODE,
        rationale="Detect input mode from provided fields.",
        inputs_summary={"image_path": bool(image_path), "svg_path": bool(svg_path)},
        outputs_summary={"input_mode": input_mode},
    )


def _parse_question(state: GraphState, llm: Any) -> None:
    question = state.inputs.get("question", "")
    chart_type_hint = state.inputs.get("chart_type_hint")
    update_spec, issues = parse_question(question, llm, chart_type_hint=chart_type_hint)
    state.perception["update_spec"] = update_spec
    _extend_issues(state.perception, issues)
    append_trace(
        state.trace,
        node="execute",
        action=PARSE_QUESTION,
        rationale="Parse update intent from question.",
        inputs_summary={"question_len": len(question)},
        outputs_summary={"new_points": len(update_spec.get("new_points", []))},
    )


def _perceive_image_svg(state: GraphState, llm: Any) -> None:
    svg_path = state.inputs.get("svg_path")
    question = state.inputs.get("question", "")
    perception = perceive_svg(
        svg_path,
        question=question,
        llm=llm,
        perception_mode=state.inputs.get("svg_perception_mode"),
    )
    state.perception.update(perception)
    state.perception["svg_perceived"] = True
    append_trace(
        state.trace,
        node="execute",
        action=PERCEIVE_IMAGE_SVG,
        rationale="Perceive SVG primitives.",
        inputs_summary={"svg_path": svg_path},
        outputs_summary={"primitives_summary": perception.get("primitives_summary", {})},
        error=perception.get("error"),
    )


def _perceive_text(state: GraphState, llm: Any) -> None:
    text_spec = state.inputs.get("text_spec", "")
    question = state.inputs.get("question", "")
    update_spec = state.perception.get("update_spec", {})
    perception = perceive_text(text_spec, question, update_spec, llm)
    state.perception.update(perception)
    state.perception["text_perceived"] = True
    append_trace(
        state.trace,
        node="execute",
        action=PERCEIVE_TEXT,
        rationale="Perceive chart type from text.",
        inputs_summary={"text_len": len(text_spec)},
        outputs_summary={"chart_type": perception.get("chart_type")},
        error=perception.get("error"),
    )


def _sanity_check(state: GraphState) -> None:
    result = run_sanity_check(state)
    state.perception["sanity_checked"] = True
    state.perception["sanity_passed"] = result["passed"]
    _extend_issues(state.perception, result.get("issues", []))
    _extend_actions(state.perception, result.get("suggested_next_actions", []))
    if result.get("adjusted_confidence") is not None:
        state.perception["chart_type_confidence"] = result["adjusted_confidence"]
    append_trace(
        state.trace,
        node="execute",
        action=SANITY_CHECK,
        rationale="Validate perception outputs.",
        inputs_summary={"input_mode": state.perception.get("input_mode")},
        outputs_summary={"passed": result["passed"]},
    )


def _apply_retry(state: GraphState) -> None:
    state.budget.retry_count += 1
    state.perception["sanity_checked"] = False
    state.perception["sanity_passed"] = False
    append_trace(
        state.trace,
        node="execute",
        action=RETRY,
        rationale="Retry perception with relaxed mode.",
        inputs_summary={"retry_count": state.budget.retry_count},
        outputs_summary={},
    )


def _extend_issues(perception: dict[str, Any], issues: list[str]) -> None:
    if not issues:
        return
    perception.setdefault("issues", [])
    for issue in issues:
        if issue not in perception["issues"]:
            perception["issues"].append(issue)


def _extend_actions(perception: dict[str, Any], actions: list[str]) -> None:
    if not actions:
        return
    perception.setdefault("suggested_next_actions", [])
    for action in actions:
        if action not in perception["suggested_next_actions"]:
            perception["suggested_next_actions"].append(action)


def _finalize_perception(state: GraphState) -> None:
    perception = state.perception
    perception.setdefault("input_mode", "text")
    perception.setdefault("chart_type", "unknown")
    perception.setdefault("chart_type_confidence", 0.0)
    perception.setdefault("update_spec", {"new_points": [], "raw": ""})
    perception.setdefault("mapping_ok", False)
    perception.setdefault("mapping_confidence", 0.0)
    perception.setdefault("primitives_summary", {})
    perception.setdefault("issues", [])
    perception.setdefault("suggested_next_actions", [])


def _coerce_state(state: Any) -> GraphState:
    if isinstance(state, GraphState):
        return state
    if isinstance(state, dict):
        return GraphState(**state)
    raise TypeError(f"Unexpected state type: {type(state)}")
