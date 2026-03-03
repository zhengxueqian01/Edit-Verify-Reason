from __future__ import annotations

import os
from typing import Any

from chart_agent.core.state import GraphState


def run_sanity_check(state: GraphState) -> dict[str, Any]:
    issues: list[str] = []
    suggested_next_actions: list[str] = []
    adjusted_confidence: float | None = None

    input_mode = state.perception.get("input_mode")
    if input_mode not in ("image", "text"):
        issues.append("input_mode is invalid")
        suggested_next_actions.append("RETRY_PARSE_SVG")

    if input_mode == "image":
        svg_path = state.inputs.get("svg_path")
        if not svg_path or not os.path.exists(svg_path):
            issues.append("svg missing for image input")
            suggested_next_actions.append("ASK_FOR_SVG")

    if input_mode == "text" and state.perception.get("chart_type") in (None, "unknown"):
        issues.append("chart_type unresolved for text input")
        suggested_next_actions.append("USE_LLM_TEXT_PARSE")

    new_points = state.perception.get("update_spec", {}).get("new_points", [])
    chart_type = state.perception.get("chart_type")
    if new_points and chart_type and chart_type not in ("scatter", "graph"):
        issues.append("new_points present but chart_type is not scatter")
        suggested_next_actions.append("CONFIRM_CHART_TYPE")
        adjusted_confidence = min(0.4, state.perception.get("chart_type_confidence", 0.0))

    passed = len(issues) == 0

    return {
        "passed": passed,
        "issues": issues,
        "suggested_next_actions": suggested_next_actions,
        "adjusted_confidence": adjusted_confidence,
    }
