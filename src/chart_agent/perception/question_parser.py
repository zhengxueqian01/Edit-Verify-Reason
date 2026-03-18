from __future__ import annotations

import json
import re
from typing import Any

_POINT_PATTERN = re.compile(r"\((-?\d+(?:\.\d+)?)[\s,]+(-?\d+(?:\.\d+)?)\)")
_BRACKET_PAIR_PATTERN = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)
_HEX_COLOR_PATTERN = re.compile(r"#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?\b")
_COLOR_KEYWORDS = {
    "red": {"red", "reds", "红", "红色"},
    "blue": {"blue", "blues", "蓝", "蓝色"},
    "green": {"green", "greens", "绿", "绿色"},
    "orange": {"orange", "oranges", "橙", "橙色", "橘", "橘色"},
    "yellow": {"yellow", "yellows", "黄", "黄色"},
    "purple": {"purple", "purples", "violet", "violets", "紫", "紫色"},
    "pink": {"pink", "pinks", "粉", "粉色"},
    "black": {"black", "blacks", "黑", "黑色"},
    "gray": {"gray", "grey", "grays", "greys", "灰", "灰色"},
    "white": {"white", "whites", "白", "白色"},
    "cyan": {"cyan", "cyans", "teal", "青", "青色"},
    "brown": {"brown", "browns", "棕", "棕色", "褐", "褐色"},
}


def parse_question(
    question: str,
    llm: Any | None = None,
    chart_type_hint: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    llm_used = False
    llm_success = False

    scatter_like = _looks_like_scatter_request(question)
    hint = str(chart_type_hint or "").strip().lower()
    if hint and hint not in ("scatter", "graph"):
        scatter_like = False

    if llm is not None:
        llm_used = True
        llm_result = _parse_with_llm(question, llm)
        if llm_result is not None:
            llm_success = True
            llm_points = llm_result.get("new_points", [])
            if not scatter_like:
                llm_points = []
            update_spec = {
                "new_points": llm_points,
                "raw": question,
                "llm_used": llm_used,
                "llm_success": llm_success,
                "llm_fallback_used": False,
                "point_color": _extract_point_color(question, llm_result.get("point_color")),
            }
            llm_issues = llm_result.get("issues", [])
            if llm_issues:
                issues.extend(llm_issues)
            if update_spec["new_points"]:
                return update_spec, issues
            issues.append("LLM parsed no new points from question; falling back to regex.")
            update_spec = _parse_with_regex(question)
            if not scatter_like:
                update_spec["new_points"] = []
            update_spec["llm_used"] = llm_used
            update_spec["llm_success"] = llm_success
            update_spec["llm_fallback_used"] = True
            update_spec["point_color"] = _extract_point_color(question, update_spec.get("point_color"))
            return update_spec, issues

    update_spec = _parse_with_regex(question)
    if not scatter_like:
        update_spec["new_points"] = []
    update_spec["llm_used"] = llm_used
    update_spec["llm_success"] = llm_success
    update_spec["llm_fallback_used"] = True if llm_used else False
    update_spec["point_color"] = _extract_point_color(question, update_spec.get("point_color"))
    if not update_spec["new_points"]:
        issues.append("No new points parsed from question.")
    return update_spec, issues


def _parse_with_regex(question: str) -> dict[str, Any]:
    points = []
    for match in _POINT_PATTERN.findall(question):
        x_str, y_str = match
        try:
            x_val = float(x_str)
            y_val = float(y_str)
        except ValueError:
            continue
        points.append({"x": x_val, "y": y_val})

    for match in _BRACKET_PAIR_PATTERN.findall(question):
        x_str, y_str = match
        try:
            x_val = float(x_str)
            y_val = float(y_str)
        except ValueError:
            continue
        points.append({"x": x_val, "y": y_val})

    return {
        "new_points": points,
        "raw": question,
        "point_color": _extract_point_color(question),
    }


def _parse_with_llm(question: str, llm: Any) -> dict[str, Any] | None:
    prompt = (
        "You are parsing a chart-update request. "
        "Identify any explicit numeric updates in the question. "
        "Return JSON only with keys: new_points (list of {x,y}), point_color (string), issues (list). "
        "If the chart type is unclear or not scatter, leave new_points empty and add a note in issues."
        f"\nQuestion: {question}"
    )
    try:
        response = llm.invoke(prompt)
    except Exception:
        return None
    content = getattr(response, "content", "")
    payload = _safe_json_loads(content)
    if not payload:
        return None
    points = payload.get("new_points", [])
    if not isinstance(points, list):
        points = []
    cleaned = []
    for item in points:
        if not isinstance(item, dict):
            continue
        try:
            x_val = float(item.get("x"))
            y_val = float(item.get("y"))
        except (TypeError, ValueError):
            continue
        cleaned.append({"x": x_val, "y": y_val})
    payload["new_points"] = cleaned
    payload["point_color"] = _extract_point_color(question, payload.get("point_color"))
    return payload


def _safe_json_loads(content: str) -> dict[str, Any] | None:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def _looks_like_scatter_request(question: str) -> bool:
    text = (question or "").lower()
    if "scatter" in text or "point" in text or "points" in text:
        return True
    if "coordinate" in text or "x=" in text or "y=" in text:
        return True
    if _POINT_PATTERN.search(question):
        return True
    return False


def _extract_point_color(question: str, llm_color: Any | None = None) -> str:
    llm_normalized = _normalize_color_token(llm_color)
    if llm_normalized:
        return llm_normalized

    text = str(question or "")
    hex_match = _HEX_COLOR_PATTERN.search(text)
    if hex_match:
        return hex_match.group(0).lower()

    lowered = text.lower()
    for color_name, aliases in _COLOR_KEYWORDS.items():
        for alias in aliases:
            if _contains_color_alias(text, lowered, alias):
                return color_name
    return ""


def _normalize_color_token(value: Any | None) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    if _HEX_COLOR_PATTERN.fullmatch(text):
        return text
    for color_name, aliases in _COLOR_KEYWORDS.items():
        if text == color_name or text in aliases:
            return color_name
    return ""


def _contains_color_alias(raw_text: str, lowered_text: str, alias: str) -> bool:
    if not alias:
        return False
    if re.search(r"[a-z]", alias):
        return re.search(rf"\b{re.escape(alias.lower())}\b", lowered_text) is not None
    return alias in raw_text
