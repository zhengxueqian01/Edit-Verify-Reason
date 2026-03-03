from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from typing import Any

from chart_agent.perception.svg_renderer import default_output_paths, render_svg_to_png

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"

ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)


def _parse_svg_tree(svg_path: str) -> ET.ElementTree:
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    return ET.parse(svg_path, parser=parser)


def update_scatter_svg(
    svg_path: str,
    new_points: list[dict[str, float]],
    mapping_info: dict[str, Any],
    output_path: str | None = None,
    svg_output_path: str | None = None,
    chart_type: str = "scatter",
    question: str = "",
    llm: Any | None = None,
) -> str:
    if not new_points:
        raise ValueError("No new points to render.")

    x_ticks = mapping_info.get("x_ticks", [])
    y_ticks = mapping_info.get("y_ticks", [])
    if len(x_ticks) < 2 or len(y_ticks) < 2:
        raise ValueError("Insufficient ticks to map data points.")

    tree = _parse_svg_tree(svg_path)
    root = tree.getroot()
    axes = root.find(f'.//{{{SVG_NS}}}g[@id="axes_1"]')
    if axes is None:
        raise ValueError("SVG axes group not found.")

    draw_points = new_points
    if llm is None:
        raise ValueError("Scatter rendering requires LLM in strict mode.")
    llm_points = _llm_refine_points(question, new_points, llm)
    if not llm_points:
        mapping_info["llm_meta"] = {"llm_used": True, "llm_success": False}
        raise ValueError("LLM failed to produce scatter points for rendering.")
    draw_points = llm_points
    mapping_info["llm_meta"] = {
        "llm_used": True,
        "llm_success": True,
        "llm_points_count": len(llm_points),
    }

    points_svg = []
    for point in draw_points:
        x_val = float(point["x"])
        y_val = float(point["y"])
        svg_x = _interpolate_axis(x_val, x_ticks)
        svg_y = _interpolate_axis(y_val, y_ticks)
        points_svg.append((svg_x, svg_y))

    href, clip_path = _extract_marker_info(axes)

    update_group = _ensure_update_group(axes)
    _clear_children(update_group)

    if href:
        for svg_x, svg_y in points_svg:
            g_attrs: dict[str, str] = {}
            if clip_path:
                g_attrs["clip-path"] = clip_path
            g = ET.SubElement(update_group, f"{{{SVG_NS}}}g", g_attrs)
            use = ET.SubElement(g, f"{{{SVG_NS}}}use")
            use.set(f"{{{XLINK_NS}}}href", href)
            use.set("x", f"{svg_x:.6f}")
            use.set("y", f"{svg_y:.6f}")
            use.set("style", "fill: #ff1493; fill-opacity: 0.85")
    else:
        radius = _fallback_radius(mapping_info)
        for svg_x, svg_y in points_svg:
            circle = ET.SubElement(update_group, f"{{{SVG_NS}}}circle")
            circle.set("cx", f"{svg_x:.6f}")
            circle.set("cy", f"{svg_y:.6f}")
            circle.set("r", f"{radius:.6f}")
            circle.set("style", "fill: #ff1493; stroke: #000000; stroke-width: 0.5; fill-opacity: 0.85")

    svg_out, png_out = default_output_paths(svg_path, chart_type)
    target_svg = svg_output_path or svg_out
    target_png = output_path or png_out
    os.makedirs(os.path.dirname(target_svg), exist_ok=True)
    tree.write(target_svg, encoding="utf-8", xml_declaration=True)
    return render_svg_to_png(target_svg, target_png)


def _llm_refine_points(question: str, points: list[dict[str, float]], llm: Any) -> list[dict[str, float]]:
    prompt = (
        "You are validating scatter points to render on chart.\n"
        f"Question: {question}\n"
        f"Candidate points: {json.dumps(points, ensure_ascii=False)}\n"
        "Return JSON only with key points: [{\"x\": number, \"y\": number}]"
    )
    try:
        response = llm.invoke(prompt)
    except Exception:
        return []
    content = getattr(response, "content", "")
    payload = _safe_json_loads(content)
    if not payload:
        return []
    raw = payload.get("points", [])
    if not isinstance(raw, list):
        return []
    out: list[dict[str, float]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            out.append({"x": float(item.get("x")), "y": float(item.get("y"))})
        except Exception:
            continue
    return out


def _safe_json_loads(content: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(content)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_marker_info(axes: ET.Element) -> tuple[str | None, str | None]:
    path_collection = axes.find(f'.//{{{SVG_NS}}}g[@id="PathCollection_1"]')
    if path_collection is None:
        return None, None

    href = None
    clip_path = None
    for g in path_collection.findall(f'.//{{{SVG_NS}}}g'):
        use = g.find(f'.//{{{SVG_NS}}}use')
        if use is not None:
            href = use.get(f"{{{XLINK_NS}}}href") or use.get("href")
            clip_path = g.get("clip-path")
            break
    if href:
        return href, clip_path

    use = path_collection.find(f'.//{{{SVG_NS}}}use')
    if use is None:
        return None, None
    return use.get(f"{{{XLINK_NS}}}href") or use.get("href"), None


def _ensure_update_group(axes: ET.Element) -> ET.Element:
    update_group = axes.find(f'.//{{{SVG_NS}}}g[@id="PathCollection_update"]')
    if update_group is not None:
        return update_group
    return ET.SubElement(axes, f"{{{SVG_NS}}}g", {"id": "PathCollection_update"})


def _clear_children(group: ET.Element) -> None:
    for child in list(group):
        group.remove(child)


def _interpolate_axis(value: float, ticks: list[tuple[float, float]]) -> float:
    ticks_sorted = sorted(ticks, key=lambda t: t[1])
    for idx in range(len(ticks_sorted) - 1):
        pixel1, data1 = ticks_sorted[idx]
        pixel2, data2 = ticks_sorted[idx + 1]
        if data1 <= value <= data2 or data2 <= value <= data1:
            if data2 == data1:
                return pixel1
            ratio = (value - data1) / (data2 - data1)
            return pixel1 + ratio * (pixel2 - pixel1)

    pixel1, data1 = ticks_sorted[0]
    pixel2, data2 = ticks_sorted[1]
    if value < min(data1, data2):
        if data2 == data1:
            return pixel1
        ratio = (value - data1) / (data2 - data1)
        return pixel1 + ratio * (pixel2 - pixel1)

    pixel1, data1 = ticks_sorted[-2]
    pixel2, data2 = ticks_sorted[-1]
    if data2 == data1:
        return pixel2
    ratio = (value - data1) / (data2 - data1)
    return pixel1 + ratio * (pixel2 - pixel1)


def _fallback_radius(mapping_info: dict[str, Any]) -> float:
    point_size_svg = mapping_info.get("point_size_svg")
    if not point_size_svg:
        return 3.5
    try:
        return max(2.0, float(point_size_svg) / 2.0)
    except (TypeError, ValueError):
        return 3.5
