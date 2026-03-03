from __future__ import annotations

import json
import re
from typing import Any

_PAIR_PATTERN = re.compile(r"([A-Za-z0-9_\-]+)\s*[:=]\s*(-?\d+(?:\.\d+)?)")
_POINT_PATTERN = re.compile(r"\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)")
_EDGE_TUPLE_PATTERN = re.compile(r"\(([^()]+)\)")
_TIME_KEYWORDS = ("time", "date", "year", "month")
_GRAPH_KEYWORDS = ("graph", "node", "edge", "edges", "shortest path", "directed", "undirected")


def select_chart_type(
    text_spec: str,
    question: str,
    update_spec: dict[str, Any],
    llm: Any,
) -> dict[str, Any]:
    llm_result = _llm_fallback(text_spec, question, llm)
    if llm_result.get("chart_type") != "unknown":
        return llm_result

    points = _POINT_PATTERN.findall(text_spec) or _POINT_PATTERN.findall(question)
    pairs = _PAIR_PATTERN.findall(text_spec) or _PAIR_PATTERN.findall(question)
    lower_text = f"{text_spec} {question}".lower()

    if _looks_like_graph(text_spec, question, lower_text):
        return {
            "chart_type": "graph",
            "confidence": 0.8,
            "parsed_data": {},
            "rationale": "Detected graph keywords or edge tuples (rule fallback).",
        }

    if points:
        return {
            "chart_type": "scatter",
            "confidence": 0.85,
            "parsed_data": {"points": points},
            "rationale": "Detected coordinate pairs (rule fallback).",
        }

    if len(pairs) >= 2:
        return {
            "chart_type": "bar",
            "confidence": 0.75,
            "parsed_data": {"categories": pairs},
            "rationale": "Detected category:value pairs (rule fallback).",
        }

    if any(keyword in lower_text for keyword in _TIME_KEYWORDS):
        return {
            "chart_type": "line",
            "confidence": 0.6,
            "parsed_data": {},
            "rationale": "Detected time series keywords (rule fallback).",
        }

    return llm_result


def _llm_fallback(text_spec: str, question: str, llm: Any) -> dict[str, Any]:
    prompt = (
        "Classify chart type based on text specification and question. "
        "Return JSON with keys: chart_type, confidence, parsed_data, rationale. "
        "chart_type should be one of: scatter, bar, line, area, graph, unknown."
        f"\ntext_spec: {text_spec}\nquestion: {question}"
    )
    try:
        response = llm.invoke(prompt)
    except Exception:
        return {
            "chart_type": "unknown",
            "confidence": 0.0,
            "parsed_data": {},
            "rationale": "LLM call failed.",
        }
    content = getattr(response, "content", "")
    payload = _safe_json_loads(content)
    if not payload:
        return {
            "chart_type": "unknown",
            "confidence": 0.3,
            "parsed_data": {},
            "rationale": "LLM fallback failed to parse JSON.",
        }
    return {
        "chart_type": payload.get("chart_type", "unknown"),
        "confidence": float(payload.get("confidence", 0.3)),
        "parsed_data": payload.get("parsed_data", {}),
        "rationale": payload.get("rationale", ""),
    }


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


def _looks_like_graph(text_spec: str, question: str, lower_text: str) -> bool:
    if any(keyword in lower_text for keyword in _GRAPH_KEYWORDS):
        return True

    for match in _EDGE_TUPLE_PATTERN.findall(text_spec):
        parts = [part.strip() for part in match.split(",")]
        if len(parts) == 3:
            return True
    for match in _EDGE_TUPLE_PATTERN.findall(question):
        parts = [part.strip() for part in match.split(",")]
        if len(parts) == 3:
            return True
    return False
