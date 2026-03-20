from __future__ import annotations

import json
import math
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Any

from chart_agent.config import get_svg_perception_mode

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"


def perceive_svg(
    svg_path: str | None,
    question: str | None = None,
    llm: Any | None = None,
    *,
    perception_mode: str | None = None,
) -> dict[str, Any]:
    issues: list[str] = []
    suggested_next_actions: list[str] = []

    if not svg_path:
        issues.append("svg missing: path not provided")
        suggested_next_actions.append("ASK_FOR_SVG")
        return {
            "mapping_ok": False,
            "mapping_confidence": 0.0,
            "primitives_summary": {},
            "issues": issues,
            "suggested_next_actions": suggested_next_actions,
            "error": "svg path not provided",
        }

    if not os.path.exists(svg_path):
        issues.append("svg missing: file does not exist")
        suggested_next_actions.append("ASK_FOR_SVG")
        return {
            "mapping_ok": False,
            "mapping_confidence": 0.0,
            "primitives_summary": {},
            "issues": issues,
            "suggested_next_actions": suggested_next_actions,
            "error": "svg file not found",
        }

    try:
        with open(svg_path, "r", encoding="utf-8") as handle:
            content = handle.read()
        tree = ET.parse(svg_path)
        root = tree.getroot()
    except Exception as exc:  # pragma: no cover - filesystem guard
        issues.append("svg read error")
        suggested_next_actions.append("RETRY_PARSE_SVG")
        return {
            "mapping_ok": False,
            "mapping_confidence": 0.0,
            "primitives_summary": {},
            "issues": issues,
            "suggested_next_actions": suggested_next_actions,
            "error": str(exc),
        }

    axes_bounds = _extract_axes_bounds(root)
    x_ticks = _parse_axis_ticks(root, content, axis_id="matplotlib.axis_1", is_x=True)
    y_ticks = _parse_axis_ticks(root, content, axis_id="matplotlib.axis_2", is_x=False)
    existing_points = _extract_scatter_points(root)
    existing_point_colors = _extract_scatter_point_colors(root)
    point_size_svg, point_marker = _extract_point_size(root)
    x_labels = _extract_x_tick_labels(root, content)
    areas = _extract_area_collections(root)
    area_top_boundary, area_fills = _extract_area_top_boundary(areas)
    line_count = _count_line_series(root)

    num_circles = len(re.findall(r"<circle\b", content))
    num_points = len(existing_points)

    chart_type, chart_type_confidence = _rule_based_chart_type(
        areas=areas,
        num_points=num_points,
    )

    mapping_ok = False
    if chart_type == "scatter":
        mapping_ok = len(x_ticks) >= 2 and len(y_ticks) >= 2 and num_points > 0
    elif chart_type == "area":
        mapping_ok = len(y_ticks) >= 2 and len(area_top_boundary) > 0
    elif chart_type == "line":
        mapping_ok = len(x_ticks) >= 2 and len(y_ticks) >= 2
    mapping_confidence = _score_mapping_confidence(len(x_ticks), len(y_ticks), num_points)

    if not mapping_ok:
        issues.append("mapping incomplete from SVG.")
        suggested_next_actions.append("FALLBACK_MAPPING")

    primitives_summary = {
        "num_circles": num_circles,
        "num_points": num_points,
        "num_xticks": len(x_ticks),
        "num_yticks": len(y_ticks),
        "num_areas": len(areas),
        "num_lines": line_count,
    }

    mapping_info = {
        "axes_bounds": axes_bounds,
        "x_ticks": x_ticks,
        "y_ticks": y_ticks,
        "existing_points_svg": existing_points,
        "existing_point_colors": existing_point_colors,
        "dominant_point_color": _dominant_color(existing_point_colors),
        "point_size_svg": point_size_svg,
        "point_marker": point_marker,
        "x_labels": x_labels,
        "area_top_boundary": area_top_boundary,
        "area_fills": area_fills,
    }

    perception_mode = get_svg_perception_mode(perception_mode)
    llm_meta = None
    if llm is not None:
        if perception_mode == "llm_summary":
            llm_result = _llm_svg_summary_perception(
                question or "",
                _build_svg_summary_payload(
                    primitives_summary=primitives_summary,
                    mapping_info=mapping_info,
                    chart_type_guess=chart_type,
                ),
                llm,
            )
        else:
            llm_result = _llm_chart_type(
                question or "",
                {
                    "num_points": num_points,
                    "num_areas": len(areas),
                    "num_lines": line_count,
                    "num_xticks": len(x_ticks),
                    "num_yticks": len(y_ticks),
                },
                llm,
            )
        if llm_result:
            chart_type = llm_result["chart_type"]
            llm_meta = llm_result.get("llm_meta")
            if chart_type == "scatter":
                mapping_ok = len(x_ticks) >= 2 and len(y_ticks) >= 2 and num_points > 0
            elif chart_type == "area":
                mapping_ok = len(y_ticks) >= 2 and len(area_top_boundary) > 0
            elif chart_type == "line":
                mapping_ok = len(x_ticks) >= 2 and len(y_ticks) >= 2

    if llm_meta:
        mapping_info["llm_meta"] = llm_meta

    return {
        "chart_type": chart_type,
        "chart_type_confidence": chart_type_confidence,
        "perception_mode": perception_mode,
        "mapping_ok": mapping_ok,
        "mapping_confidence": mapping_confidence,
        "primitives_summary": primitives_summary,
        "mapping_info": mapping_info,
        "issues": issues,
        "suggested_next_actions": suggested_next_actions,
    }


def _rule_based_chart_type(
    *,
    areas: list[dict[str, Any]],
    num_points: int,
) -> tuple[str, float]:
    if areas:
        return "area", 0.7
    if num_points > 0:
        return "scatter", 0.6
    return "unknown", 0.0


def _build_svg_summary_payload(
    *,
    primitives_summary: dict[str, Any],
    mapping_info: dict[str, Any],
    chart_type_guess: str,
) -> dict[str, Any]:
    return {
        "chart_type_guess": chart_type_guess,
        "primitives_summary": primitives_summary,
        "axes_bounds": mapping_info.get("axes_bounds"),
        "x_tick_values": _compact_tick_values(mapping_info.get("x_ticks")),
        "y_tick_values": _compact_tick_values(mapping_info.get("y_ticks")),
        "x_labels": _compact_list(mapping_info.get("x_labels"), limit=12),
        "area_fill_colors": _compact_list(mapping_info.get("area_fills"), limit=8),
        "point_color_summary": {
            "dominant": mapping_info.get("dominant_point_color"),
            "sample": _compact_list(mapping_info.get("existing_point_colors"), limit=8),
        },
    }


def _compact_tick_values(values: Any, limit: int = 12) -> list[Any]:
    if not isinstance(values, list):
        return []
    compact = [item[1] if isinstance(item, tuple) and len(item) >= 2 else item for item in values]
    return _compact_list(compact, limit=limit)


def _compact_list(values: Any, limit: int = 12) -> list[Any]:
    if not isinstance(values, list):
        return []
    if len(values) <= limit:
        return values
    head = max(1, limit // 2)
    tail = max(1, limit - head)
    return values[:head] + ["..."] + values[-tail:]


def _extract_axes_bounds(root: ET.Element) -> dict[str, float] | None:
    axes = root.find(f'.//{{{SVG_NS}}}g[@id="axes_1"]')
    if axes is None:
        return None
    patch = axes.find(f'.//{{{SVG_NS}}}g[@id="patch_2"]')
    if patch is None:
        return None
    path = patch.find(f'.//{{{SVG_NS}}}path')
    if path is None:
        return None
    d_attr = path.get("d", "")
    coords = re.findall(r"[ML]\s+([\d.]+)\s+([\d.]+)", d_attr)
    if len(coords) < 4:
        return None
    x_coords = [float(c[0]) for c in coords]
    y_coords = [float(c[1]) for c in coords]
    return {
        "x_min": min(x_coords),
        "x_max": max(x_coords),
        "y_min": min(y_coords),
        "y_max": max(y_coords),
    }


def _parse_axis_ticks(
    root: ET.Element, content: str, *, axis_id: str, is_x: bool
) -> list[tuple[float, float]]:
    axis = root.find(f'.//{{{SVG_NS}}}g[@id="{axis_id}"]')
    if axis is None:
        return []

    ticks: list[tuple[float, float]] = []
    # Scientific-notation scale marker is for y-axis in our SVG datasets.
    # Applying it to x-axis corrupts year/category mapping (e.g., 1973 -> 1.973e9).
    scale = 1.0 if is_x else _extract_axis_scale(content, axis_id)
    for g in axis.findall(f'.//{{{SVG_NS}}}g'):
        tick_id = g.get("id", "")
        if not tick_id.startswith("xtick_") and not tick_id.startswith("ytick_"):
            continue

        pixel_value = _extract_tick_position(g, is_x)
        if pixel_value is None:
            continue

        data_value = _extract_tick_label(content, g)
        if data_value is None:
            continue

        ticks.append((pixel_value, data_value * scale))

    ticks.sort(key=lambda t: t[1])
    return ticks


def _extract_axis_scale(content: str, axis_id: str) -> float:
    if not content or not axis_id:
        return 1.0
    num_pattern = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:e[+-]?\d+)?"
    attr_pattern = (
        rf'<g[^>]*id="{re.escape(axis_id)}"[^>]*\sdata-axis-scale="({num_pattern})"'
    )
    attr_match = re.search(attr_pattern, content)
    if attr_match:
        try:
            attr_val = float(attr_match.group(1))
        except ValueError:
            attr_val = 1.0
        if math.isfinite(attr_val) and attr_val != 0:
            return attr_val
    marker = f'id="{axis_id}"'
    idx = content.find(marker)
    if idx < 0:
        return 1.0
    window = content[idx : idx + 60000]
    sci_patterns = [
        r"<!--\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+)e[+-]?\d+)\s*-->",
        r"<!--\s*10\^([+-]?\d+)\s*-->",
        r"<!--\s*[×x]\s*10\^([+-]?\d+)\s*-->",
    ]
    scale = None
    for pattern in sci_patterns:
        match = re.search(pattern, window, re.IGNORECASE)
        if not match:
            continue
        token = match.group(1)
        try:
            if "^" in pattern:
                scale = 10.0 ** float(token)
            else:
                scale = float(token)
        except ValueError:
            scale = None
        if scale is not None:
            break
    if scale is None:
        return 1.0
    return scale if math.isfinite(scale) and scale != 0 else 1.0


def _extract_tick_position(tick_group: ET.Element, is_x: bool) -> float | None:
    for use_elem in tick_group.findall(f'.//{{{SVG_NS}}}use'):
        attr = use_elem.get("x" if is_x else "y")
        if attr:
            try:
                return float(attr)
            except ValueError:
                continue

    line = tick_group.find(f'.//{{{SVG_NS}}}path[@d]')
    if line is None:
        return None
    d_attr = line.get("d", "")
    if is_x:
        match = re.search(r"M\s+([\d.]+)", d_attr)
    else:
        match = re.search(r"M\s+[\d.]+\s+([\d.]+)", d_attr)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _extract_tick_label(content: str, tick_group: ET.Element) -> float | None:
    text_id = None
    for g in tick_group.findall(f'.//{{{SVG_NS}}}g'):
        if g.get("id", "").startswith("text_"):
            text_id = g.get("id")
            break

    if text_id:
        pattern = rf'<g\s+id="{re.escape(text_id)}"[^>]*>\s*<!--\s*([\d.\-]+)\s*-->'
        match = re.search(pattern, content)
        if match:
            return _to_float(match.group(1))

    tick_id = tick_group.get("id", "")
    if tick_id:
        pattern = rf'<g\s+id="{re.escape(tick_id)}"[^>]*>.*?<!--\s*([\d.\-]+)\s*-->'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return _to_float(match.group(1))

    for text in tick_group.findall(f'.//{{{SVG_NS}}}text'):
        raw = (text.text or "").strip()
        if not raw:
            continue
        val = _to_float(raw)
        if val is not None:
            return val

    return None


def _extract_scatter_points(root: ET.Element) -> list[tuple[float, float]]:
    axes = root.find(f'.//{{{SVG_NS}}}g[@id="axes_1"]')
    if axes is None:
        return []

    points: list[tuple[float, float]] = []
    for path_collection in _iter_path_collections(axes):
        for use_elem in path_collection.findall(f'.//{{{SVG_NS}}}use'):
            x_attr = use_elem.get("x")
            y_attr = use_elem.get("y")
            if not x_attr or not y_attr:
                continue
            try:
                points.append((float(x_attr), float(y_attr)))
            except ValueError:
                continue
    return points


def _extract_point_size(root: ET.Element) -> tuple[float | None, str | None]:
    axes = root.find(f'.//{{{SVG_NS}}}g[@id="axes_1"]')
    if axes is None:
        return None, None
    use_elem = None
    for path_collection in _iter_path_collections(axes):
        use_elem = path_collection.find(f'.//{{{SVG_NS}}}use')
        if use_elem is not None:
            break
    if use_elem is None:
        return None, None
    href = use_elem.get(f'{{{XLINK_NS}}}href')
    if not href or not href.startswith("#"):
        return None, None
    path_id = href[1:]
    path_def = root.find(f'.//{{{SVG_NS}}}path[@id="{path_id}"]')
    if path_def is None:
        return None, None
    d_attr = path_def.get("d", "")
    coords = re.findall(r"-?[\d.]+", d_attr)
    if len(coords) < 4:
        return None, None
    y_coords = []
    for idx in range(1, len(coords), 2):
        try:
            y_coords.append(float(coords[idx]))
        except ValueError:
            continue
    if not y_coords:
        return None, None
    diameter = max(y_coords) - min(y_coords)
    if diameter <= 0:
        return None, None
    return diameter, "circle"


def _extract_scatter_point_colors(root: ET.Element) -> list[str]:
    axes = root.find(f'.//{{{SVG_NS}}}g[@id="axes_1"]')
    if axes is None:
        return []

    colors = []
    for path_collection in _iter_path_collections(axes):
        for use_elem in path_collection.findall(f'.//{{{SVG_NS}}}use'):
            fill = _extract_element_fill(use_elem)
            if fill:
                colors.append(fill)
    if colors:
        return colors

    for circle in axes.findall(f'.//{{{SVG_NS}}}circle'):
        fill = _extract_element_fill(circle)
        if fill:
            colors.append(fill)
    return colors


def _iter_path_collections(axes: ET.Element) -> list[ET.Element]:
    collections: list[tuple[int, ET.Element]] = []
    for group in axes.findall(f'.//{{{SVG_NS}}}g'):
        gid = str(group.get("id") or "")
        match = re.fullmatch(r"PathCollection_(\d+)", gid)
        if not match:
            continue
        collections.append((int(match.group(1)), group))
    collections.sort(key=lambda item: item[0])
    return [group for _, group in collections]


def _extract_x_tick_labels(root: ET.Element, content: str) -> list[tuple[float, str]]:
    axis = root.find(f'.//{{{SVG_NS}}}g[@id="matplotlib.axis_1"]')
    if axis is None:
        return []

    labels: list[tuple[float, str]] = []
    for g in axis.findall(f'.//{{{SVG_NS}}}g'):
        tick_id = g.get("id", "")
        if not tick_id.startswith("xtick_"):
            continue

        pixel_x = _extract_tick_position(g, True)
        if pixel_x is None:
            continue

        label_text = _extract_tick_label_text(content, g)
        if not label_text:
            continue

        labels.append((pixel_x, label_text))

    return labels


def _extract_tick_label_text(content: str, tick_group: ET.Element) -> str | None:
    if content:
        text_id = None
        for g in tick_group.findall(f'.//{{{SVG_NS}}}g'):
            if g.get("id", "").startswith("text_"):
                text_id = g.get("id")
                break

        if text_id:
            pattern = rf'<g\s+id="{re.escape(text_id)}"[^>]*>\s*<!--\s*([^<]+?)\s*-->'
            match = re.search(pattern, content)
            if match:
                return match.group(1).strip()

        tick_id = tick_group.get("id", "")
        if tick_id:
            pattern = rf'<g\s+id="{re.escape(tick_id)}"[^>]*>.*?<!--\s*([^<]+?)\s*-->'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                return match.group(1).strip()

    text_elem = tick_group.find(f'.//{{{SVG_NS}}}text')
    if text_elem is not None and text_elem.text:
        return text_elem.text.strip()
    return None


def _closest_label(x_value: float, labels: list[tuple[float, str]]) -> str | None:
    if not labels:
        return None
    closest = min(labels, key=lambda item: abs(item[0] - x_value))
    return closest[1]


def _extract_fill_color(style: str) -> str:
    match = re.search(r"fill:\s*(#[0-9a-fA-F]{6})", style)
    if match:
        return match.group(1).lower()
    return "#1f77b4"


def _extract_element_fill(elem: ET.Element) -> str:
    fill_attr = str(elem.get("fill") or "").strip().lower()
    if re.fullmatch(r"#[0-9a-f]{3}(?:[0-9a-f]{3})?", fill_attr):
        return fill_attr

    style = str(elem.get("style") or "")
    match = re.search(r"fill:\s*(#[0-9a-fA-F]{3,6})", style)
    if match:
        return match.group(1).lower()
    return ""


def _dominant_color(colors: list[str]) -> str:
    if not colors:
        return ""
    return Counter(colors).most_common(1)[0][0]


def _count_line_series(root: ET.Element) -> int:
    axes = root.find(f'.//{{{SVG_NS}}}g[@id="axes_1"]')
    if axes is None:
        return 0
    count = 0
    for g in axes.findall(f'.//{{{SVG_NS}}}g'):
        gid = g.get("id", "")
        if not gid.startswith("line2d_"):
            continue
        path = g.find(f'.//{{{SVG_NS}}}path')
        if path is None:
            continue
        style = path.get("style", "")
        if "stroke:" in style and "fill: none" in style:
            count += 1
    return count


def _llm_chart_type(question: str, summary: dict[str, Any], llm: Any) -> dict[str, Any] | None:
    prompt = (
        "Choose the chart type for an SVG-based plot. "
        "Return JSON only with key: chart_type. "
        "chart_type must be one of: scatter, line, area, graph, unknown. "
        f"\nQuestion: {question}\nPrimitives: {summary}"
    )
    try:
        response = llm.invoke(prompt)
    except Exception:
        return None
    content = getattr(response, "content", "")
    payload = _safe_json_loads(content)
    if not payload:
        return None
    chart_type = payload.get("chart_type", "unknown")
    if chart_type not in ("scatter", "line", "area", "graph", "unknown"):
        chart_type = "unknown"
    return {
        "chart_type": chart_type,
        "llm_meta": {
            "llm_used": True,
            "llm_success": True,
            "mode": "rules",
            "llm_raw": content,
        },
    }


def _llm_svg_summary_perception(
    question: str,
    svg_summary: dict[str, Any],
    llm: Any,
) -> dict[str, Any] | None:
    prompt = (
        "You are perceiving an SVG chart from a compact structured summary. "
        "Infer the chart type from the summary instead of relying on the user question alone. "
        "Return JSON only with key: chart_type. "
        "chart_type must be one of: scatter, line, area, graph, unknown. "
        f"\nQuestion: {question}\nSVG Summary: {json.dumps(svg_summary, ensure_ascii=True)}"
    )
    try:
        response = llm.invoke(prompt)
    except Exception:
        return None
    content = getattr(response, "content", "")
    payload = _safe_json_loads(content)
    if not payload:
        return None
    chart_type = payload.get("chart_type", "unknown")
    if chart_type not in ("scatter", "line", "area", "graph", "unknown"):
        chart_type = "unknown"
    return {
        "chart_type": chart_type,
        "llm_meta": {
            "llm_used": True,
            "llm_success": True,
            "mode": "llm_summary",
            "llm_raw": content,
            "svg_summary": svg_summary,
        },
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


def _extract_area_collections(root: ET.Element) -> list[dict[str, Any]]:
    axes = root.find(f'.//{{{SVG_NS}}}g[@id="axes_1"]')
    if axes is None:
        return []

    areas = []
    for g in axes.findall(f'.//{{{SVG_NS}}}g'):
        gid = g.get("id", "")
        if not gid.startswith("FillBetweenPolyCollection_"):
            continue
        path = g.find(f'.//{{{SVG_NS}}}path')
        if path is None:
            continue
        d_attr = path.get("d", "")
        coords = re.findall(r"[ML]\s+([\d.]+)\s+([\d.]+)", d_attr)
        if len(coords) < 4:
            continue
        points = [(float(c[0]), float(c[1])) for c in coords]
        style = path.get("style", "")
        fill = _extract_fill_color(style)
        areas.append({"points": points, "fill": fill})
    return areas


def _extract_area_top_boundary(
    areas: list[dict[str, Any]],
) -> tuple[list[tuple[float, float]], list[str]]:
    if not areas:
        return [], []
    fills = [area.get("fill", "#1f77b4") for area in areas]

    top_map: dict[float, float] = {}
    for area in areas:
        points = area.get("points", [])
        boundary = _extract_area_boundary(points)
        for x_val, y_val in boundary:
            if x_val not in top_map or y_val < top_map[x_val]:
                top_map[x_val] = y_val

    if not top_map:
        return [], fills
    top = sorted(top_map.items(), key=lambda item: item[0])
    return top, fills


def _extract_area_boundary(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(points) < 4:
        return []
    turn_idx = None
    for idx in range(1, len(points)):
        if points[idx][0] < points[idx - 1][0]:
            turn_idx = idx
            break
    if turn_idx is None:
        return points
    boundary_a = points[:turn_idx]
    boundary_b = points[turn_idx:]

    top_map: dict[float, float] = {}
    for x_val, y_val in boundary_a + boundary_b:
        if x_val not in top_map or y_val < top_map[x_val]:
            top_map[x_val] = y_val

    top = sorted(top_map.items(), key=lambda item: item[0])
    return top


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _score_mapping_confidence(x_ticks: int, y_ticks: int, points: int) -> float:
    if x_ticks < 2 or y_ticks < 2:
        return 0.2
    base = 0.4
    base += min(0.3, points / 200.0)
    base += min(0.3, min(x_ticks, y_ticks) / 10.0)
    return round(base, 3)
