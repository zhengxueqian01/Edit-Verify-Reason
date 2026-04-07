from __future__ import annotations

from typing import Any

from chart_agent.perception.chart_type_selector import select_chart_type


def perceive_text(
    text_spec: str,
    question: str,
    update_spec: dict[str, Any],
    llm: Any,
) -> dict[str, Any]:
    selection = select_chart_type(text_spec, question, update_spec, llm)
    chart_type = selection["chart_type"]
    confidence = float(selection.get("confidence", 0.0))

    primitives_summary = {
        "parsed_points_count": len(update_spec.get("new_points", [])),
    }

    suggested_next_actions = ["RENDER_BASE_CHART"]

    return {
        "chart_type": chart_type,
        "chart_type_confidence": confidence,
        "mapping_ok": False,
        "mapping_confidence": 0.0,
        "primitives_summary": primitives_summary,
        "issues": [],
        "suggested_next_actions": suggested_next_actions,
        "selection_rationale": selection.get("rationale", ""),
    }
