from __future__ import annotations

import base64
import html
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from chart_agent.core.clusterer import resolve_dbscan_params
from chart_agent.perception.svg_perceiver import perceive_svg
from chart_agent.perception.svg_renderer import render_svg_to_png

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "add_point",
        "description": "Draw a point marker at SVG coordinate.",
        "args": {
            "x": "number",
            "y": "number",
            "radius": "number, optional, default 3",
            "color": "string hex color, optional, default #ff2d55",
            "label": "string, optional",
        },
    },
    {
        "name": "draw_line",
        "description": "Draw a line segment between two SVG coordinates.",
        "args": {
            "x1": "number",
            "y1": "number",
            "x2": "number",
            "y2": "number",
            "width": "number, optional, default 1.6",
            "color": "string hex color, optional, default #ff9500",
            "label": "string, optional",
        },
    },
    {
        "name": "highlight_rect",
        "description": "Highlight a rectangle region in SVG coordinates.",
        "args": {
            "x1": "number",
            "y1": "number",
            "x2": "number",
            "y2": "number",
            "width": "number, optional, default 1.2",
            "color": "string hex color, optional, default #007aff",
            "fill_opacity": "number 0-1, optional, default 0.08",
            "label": "string, optional",
        },
    },
    {
        "name": "isolate_color_topology",
        "description": (
            "For scatter charts, fade non-target point colors and draw convex hull polygons "
            "around DBSCAN clusters of the target-color points."
        ),
        "args": {
            "target_color": "string color name or hex chosen from the legend, such as red, blue, or #ff0000",
        },
    },
    {
        "name": "isolate_all_color_topologies",
        "description": (
            "For scatter charts, cluster points separately for every detected color and draw "
            "convex hull polygons around all color-specific clusters."
        ),
        "args": {},
    },
    {
        "name": "draw_global_peak_crosshairs",
        "description": (
            "For area charts, scan all area vertices, find the absolute highest point "
            "(minimum SVG y), and draw global horizontal and vertical crosshairs plus a peak marker."
        ),
        "args": {},
    },
    {
        "name": "zoom_and_highlight_intersection",
        "description": (
            "For line charts, extract two named line series, compute polyline intersections, "
            "and mark the intersections with highlighted symbols."
        ),
        "args": {
            "line_A": "string exact or near-exact line label from the legend",
            "line_B": "string exact or near-exact line label from the legend",
        },
    },
]


def run_visual_tool_phase(
    *,
    question: str,
    chart_type: str,
    data_summary: dict[str, Any],
    image_path: str | None,
    llm: Any,
    svg_path: str | None = None,
    max_tool_calls: int = 6,
) -> dict[str, Any]:
    svg = Path(svg_path or "").expanduser() if svg_path else None
    base_image = Path(image_path or "").expanduser() if image_path else None
    base_image_ok = bool(base_image and base_image.exists() and base_image.is_file())
    if svg is None or not svg.exists() or svg.suffix.lower() != ".svg":
        return {
            "ok": False,
            "reason": "svg_missing",
            "tool_calls": [],
            "augmented_svg_path": None,
            "augmented_image_path": (str(base_image) if base_image_ok else image_path),
        }

    width, height = _svg_canvas_size(svg)
    perception = perceive_svg(str(svg), question=question, llm=None)
    scatter_cluster_context = _build_scatter_cluster_context(perception, question)
    plan = _plan_tool_calls(
        question=question,
        chart_type=chart_type,
        data_summary=data_summary,
        image_path=svg,
        llm=llm,
        canvas_width=width,
        canvas_height=height,
    )
    tool_calls = plan.get("tool_calls", [])
    if not isinstance(tool_calls, list) or not tool_calls:
        return {
            "ok": True,
            "reason": "no_tool_needed",
            "tool_calls": [],
            "planner": plan,
            "augmented_svg_path": str(svg),
            "augmented_image_path": (str(base_image) if base_image_ok else str(svg)),
        }

    out_svg = svg.with_name(f"{svg.stem}_tool_aug.svg")
    exec_result = _execute_svg_tool_calls(
        svg,
        out_svg,
        tool_calls,
        max_tool_calls=max_tool_calls,
        scatter_cluster_context=scatter_cluster_context,
    )
    exec_result["planner"] = plan

    if not exec_result.get("ok"):
        exec_result["augmented_svg_path"] = None
        exec_result["augmented_image_path"] = (str(base_image) if base_image_ok else None)
        return exec_result

    out_png = out_svg.with_suffix(".png")
    try:
        render_svg_to_png(str(out_svg), str(out_png))
        exec_result["augmented_image_path"] = str(out_png)
    except Exception:
        exec_result["augmented_image_path"] = (str(base_image) if base_image_ok else str(out_svg))

    return exec_result


def _plan_tool_calls(
    *,
    question: str,
    chart_type: str,
    data_summary: dict[str, Any],
    image_path: Path,
    llm: Any,
    canvas_width: float,
    canvas_height: float,
) -> dict[str, Any]:
    prompt = (
        "You can use visual markup tools on a chart image to improve answer reliability.\n"
        "Important: chart updates have already been applied to this SVG.\n"
        "Do NOT re-apply edits and do NOT add factual conclusions onto the chart.\n"
        "Only add light visual guides for answering the QA question.\n"
        "You must choose tools according to the chart type and the QA task.\n"
        "First, explicitly state your understanding of the QA question.\n"
        "Return JSON only with schema:\n"
        "{\"qa_understanding\":string,\"tool_calls\":[{\"tool\":string,\"args\":object}],\"notes\":string}\n"
        "Rules:\n"
        "- Use only tools listed below.\n"
        "- Use SVG coordinates only.\n"
        f"- Canvas size is width={canvas_width:.2f}, height={canvas_height:.2f}.\n"
        "- Tool selection by chart type:\n"
        "  * scatter: default to isolate_all_color_topologies for cluster/topology reading across all colors; use isolate_color_topology only when a single target color is explicitly required; use add_point/draw_line/highlight_rect only as light local guides.\n"
        "  * line: use zoom_and_highlight_intersection for counting or locating crossings between two named lines, and use add_point/draw_line/highlight_rect only as light local guides.\n"
        "  * area: use draw_global_peak_crosshairs for absolute/global highest-peak questions, and use add_point/draw_line/highlight_rect only as light local guides.\n"
        "  * bar/unknown/other: do not use scatter-only, line-only, or area-only tools unless the visual structure truly matches that tool.\n"
        "- Cross-type restrictions:\n"
        "  * Do not use isolate_color_topology or isolate_all_color_topologies unless the chart is a scatter chart or scatter-like point cloud.\n"
        "  * Do not use zoom_and_highlight_intersection unless the chart contains line series and the task is about line crossings/intersections.\n"
        "  * Do not use draw_global_peak_crosshairs unless the chart is an area chart and the task is about the global top peak.\n"
        "- Keep overlays minimal and non-occluding.\n"
        "- Prefer 1-3 calls; use 4+ only if the question truly requires multiple marks.\n"
        "- Never add duplicate or near-duplicate marks.\n"
        "- Avoid full-chart boxes or long explanatory text.\n"
        "- If no visual enhancement is needed, return empty tool_calls.\n"
        "- Max 6 tool calls.\n"
        f"Question: {question}\n"
        f"Chart type: {chart_type}\n"
        f"Data summary: {json.dumps(data_summary, ensure_ascii=False)}\n"
        f"Tools: {json.dumps(TOOL_SPECS, ensure_ascii=False)}\n"
    )

    content = ""
    try:
        response = _invoke_multimodal_or_text(llm, prompt, image_path)
        content = _coerce_content_to_text(getattr(response, "content", ""))
    except Exception as exc:
        return {"tool_calls": [], "notes": f"planner_error: {exc}", "llm_success": False}

    payload = _safe_json_loads(content)
    if not payload:
        calls = _default_line_intersection_tool_calls(chart_type=chart_type, question=question)
        return {
            "tool_calls": calls,
            "notes": "planner_non_json",
            "llm_success": False,
            "llm_raw": content,
        }
    raw_calls = payload.get("tool_calls", [])
    calls, rejected = _coerce_tool_calls(raw_calls, canvas_width=canvas_width, canvas_height=canvas_height)
    calls = _prefer_multi_color_scatter_tools(chart_type=chart_type, question=question, tool_calls=calls)
    calls = _prefer_line_intersection_tools(chart_type=chart_type, question=question, tool_calls=calls)
    return {
        "qa_understanding": str(payload.get("qa_understanding") or ""),
        "tool_calls": calls,
        "rejected_tool_calls": rejected,
        "tool_call_count": len(calls),
        "notes": str(payload.get("notes") or ""),
        "llm_success": True,
        "llm_raw": content,
    }


def _prefer_multi_color_scatter_tools(
    *,
    chart_type: str,
    question: str,
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if str(chart_type or "").strip().lower() != "scatter":
        return tool_calls
    if not isinstance(tool_calls, list) or not tool_calls:
        return tool_calls
    keep_single_color = _question_explicitly_mentions_color(question)

    has_all_colors = any(str(call.get("tool") or "").strip() == "isolate_all_color_topologies" for call in tool_calls)
    if has_all_colors:
        return tool_calls if keep_single_color else _remove_single_color_scatter_calls(tool_calls)

    if keep_single_color:
        return tool_calls

    updated = _remove_single_color_scatter_calls(tool_calls)
    had_single_color = any(str(call.get("tool") or "").strip() == "isolate_color_topology" for call in tool_calls)
    if had_single_color:
        return [{"tool": "isolate_all_color_topologies", "args": {}}] + updated
    return updated


def _remove_single_color_scatter_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = []
    for call in tool_calls:
        tool = str(call.get("tool") or "").strip()
        if tool == "isolate_color_topology":
            continue
        updated.append(call)
    return updated


def _prefer_line_intersection_tools(
    *,
    chart_type: str,
    question: str,
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if str(chart_type or "").strip().lower() != "line":
        return tool_calls
    fallback = _default_line_intersection_tool_calls(chart_type=chart_type, question=question)
    if not fallback:
        return tool_calls
    for call in tool_calls:
        if str(call.get("tool") or "").strip() == "zoom_and_highlight_intersection":
            return tool_calls
    return fallback + tool_calls


def _default_line_intersection_tool_calls(*, chart_type: str, question: str) -> list[dict[str, Any]]:
    if str(chart_type or "").strip().lower() != "line":
        return []
    labels = _extract_line_labels_from_intersection_question(question)
    if not labels:
        return []
    return [
        {
            "tool": "zoom_and_highlight_intersection",
            "args": {"line_A": labels[0], "line_B": labels[1]},
        }
    ]


def _extract_line_labels_from_intersection_question(question: str) -> tuple[str, str] | None:
    text = str(question or "").strip().rstrip("?.")
    if not text:
        return None
    patterns = [
        r"lines?\s+for\s+(.+?)\s+and\s+(.+?)\s+(?:intersect|intersections|cross|crossing)\b",
        r"between\s+(.+?)\s+and\s+(.+?)\s*,?\s*how many times do (?:the )?lines?\s+(?:intersect|cross)\b",
        r"how many times do\s+(.+?)\s+and\s+(.+?)\s+(?:intersect|cross)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        left = _clean_line_question_label(match.group(1))
        right = _clean_line_question_label(match.group(2))
        if left and right and left.lower() != right.lower():
            return left, right
    return None


def _clean_line_question_label(label: str) -> str:
    cleaned = str(label or "").strip().strip("\"'` ")
    cleaned = re.sub(r"^(the|line|lines|category|series)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+(line|lines|category|series)$", "", cleaned, flags=re.IGNORECASE)
    return _short_text(cleaned.strip(), 64)


def _question_explicitly_mentions_color(question: str) -> bool:
    text = str(question or "").strip()
    if not text:
        return False
    lowered = text.lower()
    if re.search(r"#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?\b", text):
        return True
    for aliases in _question_color_aliases().values():
        for alias in aliases:
            if re.search(r"[a-z]", alias):
                if re.search(rf"\b{re.escape(alias.lower())}\b", lowered):
                    return True
            elif alias in text:
                return True
    return False


def _question_color_aliases() -> dict[str, set[str]]:
    return {
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


def _build_scatter_cluster_context(perception: dict[str, Any], question: str) -> dict[str, Any] | None:
    if not isinstance(perception, dict):
        return None
    mapping_info = perception.get("mapping_info")
    if not isinstance(mapping_info, dict):
        return None
    x_ticks = mapping_info.get("x_ticks")
    y_ticks = mapping_info.get("y_ticks")
    if not isinstance(x_ticks, list) or not isinstance(y_ticks, list) or len(x_ticks) < 2 or len(y_ticks) < 2:
        return None
    eps, min_samples = resolve_dbscan_params(
        question,
        default_eps=6.0,
        default_min_samples=3,
    )
    return {
        "x_ticks": x_ticks,
        "y_ticks": y_ticks,
        "eps": eps,
        "min_samples": min_samples,
    }


def _coerce_tool_calls(
    raw: Any,
    *,
    canvas_width: float,
    canvas_height: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(raw, list):
        return [], []
    out: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        tool = str(item.get("tool") or "").strip()
        args = item.get("args", {})
        if tool not in {
            "add_point",
            "draw_line",
            "highlight_rect",
            "isolate_color_topology",
            "isolate_all_color_topologies",
            "draw_global_peak_crosshairs",
            "zoom_and_highlight_intersection",
        }:
            rejected.append({"tool": tool, "args": args, "reason": "unknown_tool"})
            continue
        if not isinstance(args, dict):
            args = {}
        normalized, reason = _normalize_tool_call(
            tool,
            args,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
        )
        if normalized is None:
            rejected.append({"tool": tool, "args": args, "reason": reason or "invalid_args"})
            continue
        fingerprint = _tool_call_fingerprint(normalized)
        if fingerprint in seen:
            rejected.append({"tool": tool, "args": args, "reason": "duplicate"})
            continue
        seen.add(fingerprint)
        out.append(normalized)
    return out, rejected


def _normalize_tool_call(
    tool: str,
    args: dict[str, Any],
    *,
    canvas_width: float,
    canvas_height: float,
) -> tuple[dict[str, Any] | None, str | None]:
    if tool == "add_point":
        x = _clamp(_as_float(args.get("x"), 0.0), 0.0, canvas_width)
        y = _clamp(_as_float(args.get("y"), 0.0), 0.0, canvas_height)
        radius = _clamp(_as_float(args.get("radius"), 3.0), 0.8, 8.0)
        return (
            {
                "tool": tool,
                "args": {
                    "x": round(x, 3),
                    "y": round(y, 3),
                    "radius": round(radius, 3),
                    "color": _safe_color(args.get("color"), "#ff2d55"),
                    "label": _short_text(str(args.get("label") or "").strip(), 28),
                },
            },
            None,
        )
    if tool == "draw_line":
        x1 = _clamp(_as_float(args.get("x1"), 0.0), 0.0, canvas_width)
        y1 = _clamp(_as_float(args.get("y1"), 0.0), 0.0, canvas_height)
        x2 = _clamp(_as_float(args.get("x2"), 0.0), 0.0, canvas_width)
        y2 = _clamp(_as_float(args.get("y2"), 0.0), 0.0, canvas_height)
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if length < 2.0:
            return None, "line_too_short"
        return (
            {
                "tool": tool,
                "args": {
                    "x1": round(x1, 3),
                    "y1": round(y1, 3),
                    "x2": round(x2, 3),
                    "y2": round(y2, 3),
                    "width": round(_clamp(_as_float(args.get("width"), 1.6), 0.6, 4.0), 3),
                    "color": _safe_color(args.get("color"), "#ff9500"),
                    "label": _short_text(str(args.get("label") or "").strip(), 28),
                },
            },
            None,
        )
    if tool == "highlight_rect":
        x1 = _clamp(_as_float(args.get("x1"), 0.0), 0.0, canvas_width)
        y1 = _clamp(_as_float(args.get("y1"), 0.0), 0.0, canvas_height)
        x2 = _clamp(_as_float(args.get("x2"), 0.0), 0.0, canvas_width)
        y2 = _clamp(_as_float(args.get("y2"), 0.0), 0.0, canvas_height)
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        width = right - left
        height = bottom - top
        if width < 2.0 or height < 2.0:
            return None, "rect_too_small"
        area_ratio = (width * height) / max(canvas_width * canvas_height, 1.0)
        if area_ratio > 0.6:
            return None, "rect_too_large"
        return (
            {
                "tool": tool,
                "args": {
                    "x1": round(left, 3),
                    "y1": round(top, 3),
                    "x2": round(right, 3),
                    "y2": round(bottom, 3),
                    "width": round(_clamp(_as_float(args.get("width"), 1.2), 0.6, 4.0), 3),
                    "color": _safe_color(args.get("color"), "#007aff"),
                    "fill_opacity": round(_clamp(_as_float(args.get("fill_opacity"), 0.08), 0.0, 0.25), 4),
                    "label": _short_text(str(args.get("label") or "").strip(), 28),
                },
            },
            None,
        )
    if tool == "isolate_color_topology":
        target_color = _normalize_color_selector(args.get("target_color"))
        if not target_color:
            return None, "invalid_target_color"
        return (
            {
                "tool": tool,
                "args": {
                    "target_color": target_color,
                },
            },
            None,
        )
    if tool == "draw_global_peak_crosshairs":
        return (
            {
                "tool": tool,
                "args": {},
            },
            None,
        )
    if tool == "isolate_all_color_topologies":
        return (
            {
                "tool": tool,
                "args": {},
            },
            None,
        )
    if tool == "zoom_and_highlight_intersection":
        line_a = _short_text(str(args.get("line_A") or "").strip(), 64)
        line_b = _short_text(str(args.get("line_B") or "").strip(), 64)
        if not line_a or not line_b:
            return None, "missing_line_labels"
        if line_a.lower() == line_b.lower():
            return None, "same_line_labels"
        return (
            {
                "tool": tool,
                "args": {
                    "line_A": line_a,
                    "line_B": line_b,
                },
            },
            None,
        )
    return None, "unknown_tool"


def _tool_call_fingerprint(call: dict[str, Any]) -> str:
    tool = str(call.get("tool") or "")
    args = call.get("args", {}) if isinstance(call.get("args"), dict) else {}
    rounded: dict[str, Any] = {}
    for key, value in sorted(args.items()):
        if isinstance(value, float):
            rounded[key] = round(value, 1)
        else:
            rounded[key] = value
    return json.dumps({"tool": tool, "args": rounded}, ensure_ascii=False, sort_keys=True)


def _execute_svg_tool_calls(
    svg_path: Path,
    out_svg: Path,
    tool_calls: list[dict[str, Any]],
    *,
    max_tool_calls: int,
    scatter_cluster_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.parse(svg_path, parser=parser)
    root = tree.getroot()
    ns = _svg_ns(root)
    canvas_w, canvas_h = _svg_canvas_size(svg_path)
    parent = _find_overlay_parent(root, ns)
    overlay = _ensure_overlay_group(parent, ns)

    executed: list[dict[str, Any]] = []
    errors: list[str] = []

    for idx, call in enumerate(tool_calls[:max_tool_calls]):
        tool = str(call.get("tool") or "")
        args = call.get("args", {}) if isinstance(call.get("args"), dict) else {}
        try:
            if tool == "add_point":
                _svg_add_point(overlay, ns, args, canvas_w, canvas_h)
            elif tool == "draw_line":
                _svg_draw_line(overlay, ns, args, canvas_w, canvas_h)
            elif tool == "highlight_rect":
                _svg_highlight_rect(overlay, ns, args, canvas_w, canvas_h)
            elif tool == "isolate_color_topology":
                _svg_isolate_color_topology(
                    root,
                    overlay,
                    ns,
                    args,
                    canvas_w,
                    canvas_h,
                    scatter_cluster_context=scatter_cluster_context,
                )
            elif tool == "isolate_all_color_topologies":
                _svg_isolate_all_color_topologies(
                    root,
                    overlay,
                    ns,
                    canvas_w,
                    canvas_h,
                    scatter_cluster_context=scatter_cluster_context,
                )
            elif tool == "draw_global_peak_crosshairs":
                _svg_draw_global_peak_crosshairs(root, overlay, ns, canvas_w, canvas_h)
            elif tool == "zoom_and_highlight_intersection":
                _svg_zoom_and_highlight_intersection(root, overlay, ns, args, canvas_w, canvas_h)
            else:
                raise ValueError(f"unknown tool: {tool}")
            executed.append({"index": idx + 1, "tool": tool, "args": args})
        except Exception as exc:
            errors.append(f"tool[{idx + 1}] {tool} failed: {exc}")

    out_svg.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_svg, encoding="utf-8", xml_declaration=True)
    return {
        "ok": len(executed) > 0,
        "tool_calls": tool_calls[:max_tool_calls],
        "executed": executed,
        "errors": errors,
        "augmented_svg_path": str(out_svg),
        "augmented_image_path": str(out_svg),
    }


def _svg_add_point(overlay: ET.Element, ns: str, args: dict[str, Any], w: float, h: float) -> None:
    x = _clamp(_as_float(args.get("x"), 0.0), 0.0, w)
    y = _clamp(_as_float(args.get("y"), 0.0), 0.0, h)
    r = _clamp(_as_float(args.get("radius"), 3.0), 0.8, 8.0)
    color = _safe_color(args.get("color"), "#ff2d55")

    ET.SubElement(
        overlay,
        _nstag(ns, "circle"),
        {
            "cx": f"{x:.6f}",
            "cy": f"{y:.6f}",
            "r": f"{r:.6f}",
            "fill": color,
            "fill-opacity": "0.88",
            "stroke": "#000000",
            "stroke-width": "0.6",
        },
    )

    label = _short_text(str(args.get("label") or "").strip(), 28)
    if label:
        _svg_text(overlay, ns, x + r + 2, y - r - 1, label, "#111111", 9.5)


def _svg_draw_line(overlay: ET.Element, ns: str, args: dict[str, Any], w: float, h: float) -> None:
    x1 = _clamp(_as_float(args.get("x1"), 0.0), 0.0, w)
    y1 = _clamp(_as_float(args.get("y1"), 0.0), 0.0, h)
    x2 = _clamp(_as_float(args.get("x2"), 0.0), 0.0, w)
    y2 = _clamp(_as_float(args.get("y2"), 0.0), 0.0, h)
    width = _clamp(_as_float(args.get("width"), 1.6), 0.6, 4.0)
    color = _safe_color(args.get("color"), "#ff9500")

    ET.SubElement(
        overlay,
        _nstag(ns, "line"),
        {
            "x1": f"{x1:.6f}",
            "y1": f"{y1:.6f}",
            "x2": f"{x2:.6f}",
            "y2": f"{y2:.6f}",
            "stroke": color,
            "stroke-opacity": "0.92",
            "stroke-width": f"{width:.6f}",
            "stroke-linecap": "round",
        },
    )

    label = _short_text(str(args.get("label") or "").strip(), 28)
    if label:
        _svg_text(overlay, ns, (x1 + x2) / 2.0 + 2, (y1 + y2) / 2.0 - 2, label, "#111111", 9.5)


def _svg_highlight_rect(overlay: ET.Element, ns: str, args: dict[str, Any], w: float, h: float) -> None:
    x1 = _clamp(_as_float(args.get("x1"), 0.0), 0.0, w)
    y1 = _clamp(_as_float(args.get("y1"), 0.0), 0.0, h)
    x2 = _clamp(_as_float(args.get("x2"), 0.0), 0.0, w)
    y2 = _clamp(_as_float(args.get("y2"), 0.0), 0.0, h)
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    width = _clamp(_as_float(args.get("width"), 1.2), 0.6, 4.0)
    color = _safe_color(args.get("color"), "#007aff")
    fill_opacity = _clamp(_as_float(args.get("fill_opacity"), 0.08), 0.0, 0.25)

    area_ratio = ((right - left) * (bottom - top)) / max(1.0, (w * h))
    if area_ratio > 0.35:
        fill_opacity = 0.0
    elif area_ratio > 0.18:
        fill_opacity = min(fill_opacity, 0.05)

    ET.SubElement(
        overlay,
        _nstag(ns, "rect"),
        {
            "x": f"{left:.6f}",
            "y": f"{top:.6f}",
            "width": f"{(right - left):.6f}",
            "height": f"{(bottom - top):.6f}",
            "fill": color,
            "fill-opacity": f"{fill_opacity:.4f}",
            "stroke": color,
            "stroke-opacity": "0.92",
            "stroke-width": f"{width:.6f}",
        },
    )

    label = _short_text(str(args.get("label") or "").strip(), 28)
    if label:
        _svg_text(overlay, ns, left + 2, top + 11, label, "#111111", 9.5)


def _svg_text(
    overlay: ET.Element,
    ns: str,
    x: float,
    y: float,
    text: str,
    color: str,
    size: float,
) -> None:
    t = ET.SubElement(
        overlay,
        _nstag(ns, "text"),
        {
            "x": f"{x:.6f}",
            "y": f"{y:.6f}",
            "fill": color,
            "font-size": f"{size:.2f}",
            "font-family": "DejaVu Sans, Arial, sans-serif",
        },
    )
    t.text = text


def _svg_isolate_color_topology(
    root: ET.Element,
    overlay: ET.Element,
    ns: str,
    args: dict[str, Any],
    w: float,
    h: float,
    scatter_cluster_context: dict[str, Any] | None = None,
) -> None:
    target_selector = _normalize_color_selector(args.get("target_color"))
    if not target_selector:
        raise ValueError("target_color is required")

    points = _extract_colored_scatter_points(root, ns)
    if not points:
        raise ValueError("no scatter points found")

    resolved_color = _resolve_target_color(target_selector, [p["fill"] for p in points])
    if not resolved_color:
        raise ValueError(f"target color not found in svg: {target_selector}")

    target_points: list[tuple[float, float]] = []
    for point in points:
        fill = str(point.get("fill") or "").lower()
        elem = point.get("element")
        if not isinstance(elem, ET.Element):
            continue
        if fill == resolved_color:
            target_points.append((float(point["x"]), float(point["y"])))
            _set_element_opacity(elem, fill_opacity="0.98", stroke_opacity="0.98", opacity="1.0")
        else:
            _set_element_opacity(elem, fill_opacity="0.10", stroke_opacity="0.10", opacity="0.22")

    if not target_points:
        raise ValueError(f"no points found for target color: {target_selector}")

    labels = _cluster_svg_points(
        target_points,
        scatter_cluster_context=scatter_cluster_context,
    )
    clusters = _group_points_by_label(target_points, labels)
    hull_color = "#ff3b30"
    for cluster_points in clusters:
        _draw_cluster_outline(overlay, ns, cluster_points, hull_color, w, h)


def _svg_isolate_all_color_topologies(
    root: ET.Element,
    overlay: ET.Element,
    ns: str,
    w: float,
    h: float,
    scatter_cluster_context: dict[str, Any] | None = None,
) -> None:
    points = _extract_colored_scatter_points(root, ns)
    if not points:
        raise ValueError("no scatter points found")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for point in points:
        fill = str(point.get("fill") or "").lower()
        if not fill:
            continue
        grouped.setdefault(fill, []).append(point)

    if not grouped:
        raise ValueError("no colored scatter points found")

    active_colors = sorted(grouped)
    for point in points:
        elem = point.get("element")
        if not isinstance(elem, ET.Element):
            continue
        fill = str(point.get("fill") or "").lower()
        if fill in active_colors:
            _set_element_opacity(elem, fill_opacity="0.95", stroke_opacity="0.95", opacity="1.0")
        else:
            _set_element_opacity(elem, fill_opacity="0.10", stroke_opacity="0.10", opacity="0.22")

    drew_cluster = False
    for fill, color_points in grouped.items():
        cluster_points = [(float(item["x"]), float(item["y"])) for item in color_points]
        if not cluster_points:
            continue
        labels = _cluster_svg_points(
            cluster_points,
            scatter_cluster_context=scatter_cluster_context,
        )
        clusters = _group_points_by_label(cluster_points, labels)
        for cluster in clusters:
            if not cluster:
                continue
            _draw_cluster_outline(overlay, ns, cluster, fill, w, h)
            drew_cluster = True

    if not drew_cluster:
        raise ValueError("no clusters found")


def _svg_draw_global_peak_crosshairs(
    root: ET.Element,
    overlay: ET.Element,
    ns: str,
    w: float,
    h: float,
) -> None:
    area_points = _extract_area_geometry_points(root, ns)
    if not area_points:
        raise ValueError("no area geometry found")

    peak_x, peak_y = min(area_points, key=lambda point: (point[1], point[0]))
    color = "#ff3b30"
    dash = "5,5"

    ET.SubElement(
        overlay,
        _nstag(ns, "line"),
        {
            "x1": "0.000000",
            "y1": f"{peak_y:.6f}",
            "x2": f"{w:.6f}",
            "y2": f"{peak_y:.6f}",
            "stroke": color,
            "stroke-width": "2.000000",
            "stroke-dasharray": dash,
            "stroke-opacity": "0.96",
        },
    )
    ET.SubElement(
        overlay,
        _nstag(ns, "line"),
        {
            "x1": f"{peak_x:.6f}",
            "y1": f"{peak_y:.6f}",
            "x2": f"{peak_x:.6f}",
            "y2": f"{h:.6f}",
            "stroke": color,
            "stroke-width": "2.000000",
            "stroke-dasharray": dash,
            "stroke-opacity": "0.96",
        },
    )
    ET.SubElement(
        overlay,
        _nstag(ns, "circle"),
        {
            "cx": f"{peak_x:.6f}",
            "cy": f"{peak_y:.6f}",
            "r": "5.000000",
            "fill": color,
            "fill-opacity": "0.98",
            "stroke": "#ffffff",
            "stroke-width": "0.8",
        },
    )


def _svg_zoom_and_highlight_intersection(
    root: ET.Element,
    overlay: ET.Element,
    ns: str,
    args: dict[str, Any],
    w: float,
    h: float,
) -> None:
    line_a = _short_text(str(args.get("line_A") or "").strip(), 64)
    line_b = _short_text(str(args.get("line_B") or "").strip(), 64)
    if not line_a or not line_b:
        raise ValueError("line_A and line_B are required")

    content = ET.tostring(root, encoding="unicode")
    legend_map = _extract_line_legend_map(root, ns, content)
    stroke_a = _resolve_line_stroke(line_a, legend_map)
    stroke_b = _resolve_line_stroke(line_b, legend_map)
    if not stroke_a or not stroke_b:
        raise ValueError("line labels not found in legend")

    points_a = _extract_line_points_by_stroke(root, ns, stroke_a)
    points_b = _extract_line_points_by_stroke(root, ns, stroke_b)
    if len(points_a) < 2 or len(points_b) < 2:
        raise ValueError("line geometry not found")

    _svg_highlight_polyline(overlay, ns, points_a, "#ff2d55", width=3.2, label=line_a)
    _svg_highlight_polyline(overlay, ns, points_b, "#007aff", width=3.2, label=line_b)

    intersections = _polyline_intersections(points_a, points_b)
    if not intersections:
        raise ValueError("no intersections found")

    marker_color = "#ff3b30"
    for x, y in intersections:
        _svg_add_point(
            overlay,
            ns,
            {"x": x, "y": y, "radius": 4.8, "color": marker_color, "label": ""},
            w,
            h,
        )
        _svg_draw_line(
            overlay,
            ns,
            {"x1": x - 5.5, "y1": y, "x2": x + 5.5, "y2": y, "width": 1.6, "color": marker_color, "label": ""},
            w,
            h,
        )
        _svg_draw_line(
            overlay,
            ns,
            {"x1": x, "y1": y - 5.5, "x2": x, "y2": y + 5.5, "width": 1.6, "color": marker_color, "label": ""},
            w,
            h,
        )


def _svg_highlight_polyline(
    overlay: ET.Element,
    ns: str,
    points: list[tuple[float, float]],
    color: str,
    *,
    width: float,
    label: str,
) -> None:
    if len(points) < 2:
        return
    path_d = " ".join(
        [f"M {points[0][0]:.6f} {points[0][1]:.6f}"]
        + [f"L {x:.6f} {y:.6f}" for x, y in points[1:]]
    )
    ET.SubElement(
        overlay,
        _nstag(ns, "path"),
        {
            "d": path_d,
            "fill": "none",
            "stroke": color,
            "stroke-width": f"{width:.3f}",
            "stroke-linecap": "round",
            "stroke-linejoin": "round",
            "opacity": "0.92",
        },
    )
    if label:
        x, y = points[-1]
        _svg_text(overlay, ns, x + 4.0, y - 4.0, label, color, 9.5)


def _draw_cluster_outline(
    overlay: ET.Element,
    ns: str,
    cluster_points: list[tuple[float, float]],
    color: str,
    w: float,
    h: float,
) -> None:
    outline_color = _safe_color(color, "#ff3b30")
    if len(cluster_points) == 1:
        x, y = cluster_points[0]
        _svg_add_point(
            overlay,
            ns,
            {"x": x, "y": y, "radius": 5.0, "color": outline_color, "label": ""},
            w,
            h,
        )
        return
    if len(cluster_points) == 2:
        (x1, y1), (x2, y2) = cluster_points
        _svg_draw_line(
            overlay,
            ns,
            {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "width": 2.2, "color": outline_color, "label": ""},
            w,
            h,
        )
        return
    hull = _convex_hull(cluster_points)
    if len(hull) < 3:
        return
    ET.SubElement(
        overlay,
        _nstag(ns, "polygon"),
        {
            "points": " ".join(f"{x:.6f},{y:.6f}" for x, y in hull),
            "fill": outline_color,
            "fill-opacity": "0.02",
            "stroke": outline_color,
            "stroke-opacity": "0.96",
            "stroke-width": "2.2",
            "stroke-linejoin": "round",
        },
    )


def _find_overlay_parent(root: ET.Element, ns: str) -> ET.Element:
    axes = root.find(f".//{_nstag(ns, 'g')}[@id='axes_1']")
    if axes is not None:
        return axes
    return root


def _ensure_overlay_group(parent: ET.Element, ns: str) -> ET.Element:
    existing = parent.find(f"./{_nstag(ns, 'g')}[@id='tool_aug_overlay']")
    if existing is not None:
        return existing
    return ET.SubElement(parent, _nstag(ns, "g"), {"id": "tool_aug_overlay"})


def _svg_ns(root: ET.Element) -> str:
    if root.tag.startswith("{") and "}" in root.tag:
        return root.tag[1 : root.tag.find("}")]
    return "http://www.w3.org/2000/svg"


def _nstag(ns: str, name: str) -> str:
    return f"{{{ns}}}{name}" if ns else name


def _svg_canvas_size(svg_path: Path) -> tuple[float, float]:
    try:
        root = ET.parse(svg_path).getroot()
    except Exception:
        return 1000.0, 800.0

    vb = str(root.get("viewBox") or "").strip()
    if vb:
        parts = vb.replace(",", " ").split()
        if len(parts) == 4:
            try:
                w = float(parts[2])
                h = float(parts[3])
                if w > 0 and h > 0:
                    return w, h
            except Exception:
                pass

    w = _parse_svg_length(root.get("width"))
    h = _parse_svg_length(root.get("height"))
    if w > 0 and h > 0:
        return w, h
    return 1000.0, 800.0


def _parse_svg_length(value: Any) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    m = re.match(r"^(-?\d+(?:\.\d+)?)", text)
    if not m:
        return 0.0
    try:
        return float(m.group(1))
    except Exception:
        return 0.0


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _short_text(text: str, limit: int) -> str:
    value = (text or "").strip()
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


def _safe_color(raw: Any, default: str) -> str:
    text = str(raw or "").strip()
    if re.match(r"^#[0-9a-fA-F]{6}$", text):
        return text
    if re.match(r"^#[0-9a-fA-F]{3}$", text):
        return text
    return default


def _normalize_color_selector(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    if not text:
        return ""
    if re.match(r"^#[0-9a-f]{6}$", text):
        return text
    if re.match(r"^#[0-9a-f]{3}$", text):
        return text
    if re.match(r"^[a-z][a-z0-9_-]{1,20}$", text):
        return text
    return ""


def _extract_colored_scatter_points(root: ET.Element, ns: str) -> list[dict[str, Any]]:
    axes = root.find(f".//{_nstag(ns, 'g')}[@id='axes_1']")
    if axes is None:
        axes = root

    points: list[dict[str, Any]] = []
    for path_collection in _iter_scatter_path_collections(axes, ns):
        for use_elem in path_collection.findall(f".//{_nstag(ns, 'use')}"):
            x_attr = use_elem.get("x")
            y_attr = use_elem.get("y")
            fill = _extract_fill_from_element(use_elem)
            if not x_attr or not y_attr or not fill:
                continue
            try:
                points.append(
                    {
                        "element": use_elem,
                        "x": float(x_attr),
                        "y": float(y_attr),
                        "fill": fill,
                    }
                )
            except ValueError:
                continue
    for circle in axes.findall(f".//{_nstag(ns, 'circle')}"):
        fill = _extract_fill_from_element(circle)
        cx = circle.get("cx")
        cy = circle.get("cy")
        if not fill or not cx or not cy:
            continue
        try:
            points.append(
                {
                    "element": circle,
                    "x": float(cx),
                    "y": float(cy),
                    "fill": fill,
                }
            )
        except ValueError:
            continue
    return points


def _iter_scatter_path_collections(axes: ET.Element, ns: str) -> list[ET.Element]:
    collections: list[tuple[int, ET.Element]] = []
    for group in axes.findall(f".//{_nstag(ns, 'g')}"):
        gid = str(group.get("id") or "")
        match = re.fullmatch(r"PathCollection_(\d+|update)", gid)
        if not match:
            continue
        order = 10**9 if match.group(1) == "update" else int(match.group(1))
        collections.append((order, group))
    collections.sort(key=lambda item: item[0])
    return [group for _, group in collections]


def _extract_area_geometry_points(root: ET.Element, ns: str) -> list[tuple[float, float]]:
    axes = root.find(f".//{_nstag(ns, 'g')}[@id='axes_1']")
    if axes is None:
        axes = root

    points: list[tuple[float, float]] = []
    for group in axes.findall(f".//{_nstag(ns, 'g')}"):
        gid = str(group.get("id") or "")
        if not gid.startswith("FillBetweenPolyCollection_"):
            continue
        for path in group.findall(f".//{_nstag(ns, 'path')}"):
            points.extend(_parse_svg_path_points(path.get("d")))
        for polygon in group.findall(f".//{_nstag(ns, 'polygon')}"):
            points.extend(_parse_svg_polygon_points(polygon.get("points")))
    return points


def _extract_line_legend_map(root: ET.Element, ns: str, content: str) -> dict[str, str]:
    legend = root.find(f".//{_nstag(ns, 'g')}[@id='legend_1']")
    if legend is None:
        return {}

    mapping: dict[str, str] = {}
    pending_stroke: str | None = None
    for child in list(legend):
        stroke = _extract_stroke_from_group(child, ns)
        if stroke:
            pending_stroke = stroke
            continue
        label = _extract_text_label_from_group(child, content)
        if label and pending_stroke:
            mapping[label] = pending_stroke
            pending_stroke = None
    return mapping


def _extract_stroke_from_group(group: ET.Element, ns: str) -> str | None:
    path = group.find(f".//{_nstag(ns, 'path')}")
    if path is None:
        return None
    style = str(path.get("style") or "")
    match = re.search(r"stroke:\s*(#[0-9a-fA-F]{6})", style)
    if match:
        return match.group(1).lower()
    stroke = str(path.get("stroke") or "").strip().lower()
    if re.match(r"^#[0-9a-f]{6}$", stroke):
        return stroke
    return None


def _extract_text_label_from_group(group: ET.Element, content: str) -> str | None:
    gid = str(group.get("id") or "")
    if not gid.startswith("text_"):
        return None
    for node in group.iter():
        if node.tag is ET.Comment:
            comment_text = html.unescape(str(node.text or "")).strip()
            if comment_text:
                return comment_text
        text = html.unescape(str(node.text or "")).strip()
        if text:
            return text
    pattern = rf'<(?:\w+:)?g\s+id="{re.escape(gid)}"[^>]*>.*?<!--\s*(.*?)\s*-->'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return html.unescape(match.group(1)).strip()
    return None


def _resolve_line_stroke(label: str, legend_map: dict[str, str]) -> str | None:
    exact = {key.lower(): value for key, value in legend_map.items()}
    if label.lower() in exact:
        return exact[label.lower()]
    for key, value in legend_map.items():
        lowered = key.lower()
        target = label.lower()
        if target in lowered or lowered in target:
            return value
    return None


def _extract_line_points_by_stroke(root: ET.Element, ns: str, stroke: str) -> list[tuple[float, float]]:
    axes = root.find(f".//{_nstag(ns, 'g')}[@id='axes_1']")
    if axes is None:
        return []
    stroke_norm = stroke.lower()
    for group in axes.findall(f".//{_nstag(ns, 'g')}"):
        gid = str(group.get("id") or "")
        if not gid.startswith("line2d_"):
            continue
        path = group.find(f".//{_nstag(ns, 'path')}")
        if path is None:
            continue
        style = str(path.get("style") or "")
        match = re.search(r"stroke:\s*(#[0-9a-fA-F]{6})", style)
        path_stroke = match.group(1).lower() if match else str(path.get("stroke") or "").strip().lower()
        if path_stroke == stroke_norm:
            return _parse_svg_path_points(path.get("d"))
    return []


def _parse_svg_path_points(d_attr: Any) -> list[tuple[float, float]]:
    text = str(d_attr or "").strip()
    if not text:
        return []
    coords = re.findall(r"([ML])\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)", text)
    points: list[tuple[float, float]] = []
    for _, x_raw, y_raw in coords:
        try:
            points.append((float(x_raw), float(y_raw)))
        except ValueError:
            continue
    return points


def _parse_svg_polygon_points(points_attr: Any) -> list[tuple[float, float]]:
    text = str(points_attr or "").strip()
    if not text:
        return []
    tokens = re.split(r"[\s,]+", text)
    if len(tokens) < 4:
        return []
    points: list[tuple[float, float]] = []
    for idx in range(0, len(tokens) - 1, 2):
        try:
            points.append((float(tokens[idx]), float(tokens[idx + 1])))
        except ValueError:
            continue
    return points


def _polyline_intersections(
    points_a: list[tuple[float, float]],
    points_b: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    intersections: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for idx in range(len(points_a) - 1):
        a1 = points_a[idx]
        a2 = points_a[idx + 1]
        for jdx in range(len(points_b) - 1):
            b1 = points_b[jdx]
            b2 = points_b[jdx + 1]
            point = _segment_intersection(a1, a2, b1, b2)
            if point is None:
                continue
            key = (round(point[0], 3), round(point[1], 3))
            if key in seen:
                continue
            seen.add(key)
            intersections.append(point)
    return intersections


def _segment_intersection(
    a1: tuple[float, float],
    a2: tuple[float, float],
    b1: tuple[float, float],
    b2: tuple[float, float],
) -> tuple[float, float] | None:
    x1, y1 = a1
    x2, y2 = a2
    x3, y3 = b1
    x4, y4 = b2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-9:
        return None
    det_a = x1 * y2 - y1 * x2
    det_b = x3 * y4 - y3 * x4
    px = (det_a * (x3 - x4) - (x1 - x2) * det_b) / denom
    py = (det_a * (y3 - y4) - (y1 - y2) * det_b) / denom
    if not _point_on_segment(px, py, a1, a2):
        return None
    if not _point_on_segment(px, py, b1, b2):
        return None
    return (px, py)


def _point_on_segment(
    px: float,
    py: float,
    a: tuple[float, float],
    b: tuple[float, float],
) -> bool:
    min_x = min(a[0], b[0]) - 1e-6
    max_x = max(a[0], b[0]) + 1e-6
    min_y = min(a[1], b[1]) - 1e-6
    max_y = max(a[1], b[1]) + 1e-6
    return min_x <= px <= max_x and min_y <= py <= max_y


def _extract_fill_from_element(elem: ET.Element) -> str:
    fill = str(elem.get("fill") or "").strip().lower()
    if re.match(r"^#[0-9a-f]{6}$", fill):
        return fill
    if re.match(r"^#[0-9a-f]{3}$", fill):
        return fill
    style = str(elem.get("style") or "")
    match = re.search(r"fill:\s*(#[0-9a-fA-F]{3,6})", style)
    if match:
        return match.group(1).lower()
    return ""


def _set_element_opacity(
    elem: ET.Element,
    *,
    fill_opacity: str,
    stroke_opacity: str,
    opacity: str,
) -> None:
    style = str(elem.get("style") or "")
    style = _replace_style_attr(style, "fill-opacity", fill_opacity)
    style = _replace_style_attr(style, "stroke-opacity", stroke_opacity)
    style = _replace_style_attr(style, "opacity", opacity)
    if style:
        elem.set("style", style)
    else:
        elem.set("fill-opacity", fill_opacity)
        elem.set("stroke-opacity", stroke_opacity)
        elem.set("opacity", opacity)


def _replace_style_attr(style: str, name: str, value: str) -> str:
    pattern = rf"(^|;)\s*{re.escape(name)}\s*:\s*[^;]+"
    replacement = rf"\1{name}: {value}"
    if re.search(pattern, style):
        return re.sub(pattern, replacement, style)
    if style and not style.rstrip().endswith(";"):
        style = style.rstrip() + "; "
    return f"{style}{name}: {value}".strip()


def _resolve_target_color(target_selector: str, fills: list[str]) -> str | None:
    normalized_fills = sorted({str(fill or "").lower() for fill in fills if fill})
    if target_selector in normalized_fills:
        return target_selector
    selector_aliases = _color_aliases(target_selector)
    for fill in normalized_fills:
        if selector_aliases & _color_aliases(fill):
            return fill
    return None


def _color_aliases(color: str) -> set[str]:
    token = str(color or "").strip().lower()
    aliases = {token} if token else set()
    canonical = {
        "red": {"#ff0000", "#d62728", "#e41a1c", "#ff3b30"},
        "blue": {"#0000ff", "#1f77b4", "#4c78a8", "#007aff"},
        "green": {"#008000", "#2ca02c", "#4daf4a", "#34c759"},
        "orange": {"#ff7f0e", "#ff9500", "#ffa500"},
        "purple": {"#9467bd", "#800080", "#af52de"},
        "pink": {"#ff1493", "#e377c2"},
        "yellow": {"#bcbd22", "#ffff00", "#ffd60a"},
        "cyan": {"#17becf", "#00ffff"},
        "gray": {"#7f7f7f", "#808080"},
        "black": {"#000000"},
    }
    for name, values in canonical.items():
        if token == name or token in values:
            aliases.add(name)
            aliases.update(values)
    return aliases


def _dbscan_points(points: list[tuple[float, float]], eps: float, min_samples: int) -> list[int]:
    labels = [-1 for _ in points]
    visited = [False for _ in points]
    cluster_id = 0
    for idx in range(len(points)):
        if visited[idx]:
            continue
        visited[idx] = True
        neighbors = _region_query(points, idx, eps)
        if len(neighbors) < min_samples:
            labels[idx] = -1
            continue
        _expand_cluster(points, labels, visited, idx, neighbors, cluster_id, eps, min_samples)
        cluster_id += 1
    return labels


def _expand_cluster(
    points: list[tuple[float, float]],
    labels: list[int],
    visited: list[bool],
    point_idx: int,
    neighbors: list[int],
    cluster_id: int,
    eps: float,
    min_samples: int,
) -> None:
    labels[point_idx] = cluster_id
    i = 0
    while i < len(neighbors):
        neighbor_idx = neighbors[i]
        if not visited[neighbor_idx]:
            visited[neighbor_idx] = True
            neighbor_neighbors = _region_query(points, neighbor_idx, eps)
            if len(neighbor_neighbors) >= min_samples:
                for candidate in neighbor_neighbors:
                    if candidate not in neighbors:
                        neighbors.append(candidate)
        if labels[neighbor_idx] == -1:
            labels[neighbor_idx] = cluster_id
        i += 1


def _region_query(points: list[tuple[float, float]], idx: int, eps: float) -> list[int]:
    px, py = points[idx]
    neighbors: list[int] = []
    for jdx, (x, y) in enumerate(points):
        if math.hypot(px - x, py - y) <= eps:
            neighbors.append(jdx)
    return neighbors


def _auto_cluster_eps(points: list[tuple[float, float]]) -> float:
    if len(points) <= 1:
        return 8.0
    nearest: list[float] = []
    for idx, (x1, y1) in enumerate(points):
        best = float("inf")
        for jdx, (x2, y2) in enumerate(points):
            if idx == jdx:
                continue
            best = min(best, math.hypot(x1 - x2, y1 - y2))
        if best != float("inf"):
            nearest.append(best)
    if not nearest:
        return 8.0
    nearest.sort()
    median = nearest[len(nearest) // 2]
    return _clamp(median * 1.8, 6.0, 28.0)


def _cluster_svg_points(
    points_svg: list[tuple[float, float]],
    *,
    scatter_cluster_context: dict[str, Any] | None,
) -> list[int]:
    if not points_svg:
        return []
    if not isinstance(scatter_cluster_context, dict):
        return _dbscan_points(points_svg, eps=_auto_cluster_eps(points_svg), min_samples=1)

    x_ticks = scatter_cluster_context.get("x_ticks")
    y_ticks = scatter_cluster_context.get("y_ticks")
    eps = float(scatter_cluster_context.get("eps") or 6.0)
    min_samples = int(scatter_cluster_context.get("min_samples") or 3)
    if not isinstance(x_ticks, list) or not isinstance(y_ticks, list) or len(x_ticks) < 2 or len(y_ticks) < 2:
        return _dbscan_points(points_svg, eps=_auto_cluster_eps(points_svg), min_samples=1)

    data_points = [
        (
            _pixel_to_data(px, x_ticks),
            _pixel_to_data(py, y_ticks),
        )
        for px, py in points_svg
    ]
    return _dbscan_points(data_points, eps=eps, min_samples=min_samples)


def _pixel_to_data(pixel: float, ticks: list[tuple[float, float]]) -> float:
    ticks_sorted = sorted(ticks, key=lambda item: item[0])
    for idx in range(len(ticks_sorted) - 1):
        p1, d1 = ticks_sorted[idx]
        p2, d2 = ticks_sorted[idx + 1]
        if min(p1, p2) <= pixel <= max(p1, p2):
            if p2 == p1:
                return d1
            ratio = (pixel - p1) / (p2 - p1)
            return d1 + ratio * (d2 - d1)

    p1, d1 = ticks_sorted[0]
    p2, d2 = ticks_sorted[1]
    if pixel < min(p1, p2):
        if p2 == p1:
            return d1
        ratio = (pixel - p1) / (p2 - p1)
        return d1 + ratio * (d2 - d1)

    p1, d1 = ticks_sorted[-2]
    p2, d2 = ticks_sorted[-1]
    if p2 == p1:
        return d2
    ratio = (pixel - p1) / (p2 - p1)
    return d1 + ratio * (d2 - d1)


def _group_points_by_label(points: list[tuple[float, float]], labels: list[int]) -> list[list[tuple[float, float]]]:
    grouped: dict[int, list[tuple[float, float]]] = {}
    for point, label in zip(points, labels):
        grouped.setdefault(label, []).append(point)
    clusters = [cluster for _, cluster in sorted(grouped.items(), key=lambda item: item[0]) if cluster]
    return clusters


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted({(round(x, 6), round(y, 6)) for x, y in points})
    if len(unique) <= 1:
        return unique

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def _invoke_multimodal_or_text(llm: Any, prompt_text: str, image_path: Path) -> Any:
    data_url = _image_data_url(image_path)
    if data_url:
        try:
            from langchain_core.messages import HumanMessage  # type: ignore

            return llm.invoke(
                [
                    HumanMessage(
                        content=[
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ]
                    )
                ]
            )
        except Exception:
            pass
    return llm.invoke(prompt_text)


def _image_data_url(path: Path) -> str | None:
    if not path.exists():
        return None
    suffix = path.suffix.lower()
    mime = "application/octet-stream"
    if suffix == ".png":
        mime = "image/png"
    elif suffix in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif suffix == ".webp":
        mime = "image/webp"
    elif suffix == ".svg":
        mime = "image/svg+xml"
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _coerce_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())
        return "\n".join(chunks).strip()
    return str(content)


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
