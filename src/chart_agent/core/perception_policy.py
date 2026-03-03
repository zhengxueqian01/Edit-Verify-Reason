from __future__ import annotations

from chart_agent.core.perception_actions import (
    DETECT_INPUT_MODE,
    PARSE_QUESTION,
    PERCEIVE_IMAGE_SVG,
    PERCEIVE_TEXT,
    RETRY,
    SANITY_CHECK,
    STOP,
)
from chart_agent.core.state import GraphState


def perception_policy(state: GraphState) -> str:
    perception = state.perception

    if perception.get("input_mode") is None:
        return DETECT_INPUT_MODE

    if perception.get("update_spec") is None:
        return PARSE_QUESTION

    if perception.get("sanity_checked") is True:
        if perception.get("sanity_passed") is True:
            return STOP
        if state.budget.retry_count >= state.budget.max_retries:
            return STOP
        if _issue_blocks_retry(perception.get("issues", [])):
            return STOP
        return RETRY

    if perception.get("input_mode") == "image":
        if perception.get("svg_perceived") is not True:
            return PERCEIVE_IMAGE_SVG
    else:
        if perception.get("text_perceived") is not True:
            return PERCEIVE_TEXT

    return SANITY_CHECK


def _issue_blocks_retry(issues: list[str]) -> bool:
    for issue in issues:
        if "svg missing" in issue.lower():
            return True
    return False
