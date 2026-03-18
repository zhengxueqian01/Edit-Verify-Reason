from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from collections import Counter
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
    new_points: list[dict[str, Any]],
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

    # Scatter add should use the structured points as-is. LLM point rewriting
    # drops per-point color and has been the main source of mixed-color errors.
    draw_points = new_points
    mapping_info["llm_meta"] = {
        "llm_used": False,
        "llm_success": False,
        "llm_points_count": len(draw_points),
    }

    point_entries: list[dict[str, Any]] = []
    for point in draw_points:
        x_val = float(point["x"])
        y_val = float(point["y"])
        svg_x = _interpolate_axis(x_val, x_ticks)
        svg_y = _interpolate_axis(y_val, y_ticks)
        point_entries.append(
            {
                "svg_x": svg_x,
                "svg_y": svg_y,
                "requested_color": _point_requested_color(point) or mapping_info.get("requested_point_color", ""),
            }
        )

    update_group = _ensure_update_group(axes)
    _clear_children(update_group)

    for entry in point_entries:
        href, clip_path, point_fill, point_style = _select_marker_info(
            axes=axes,
            requested_color=entry["requested_color"],
            existing_colors=mapping_info.get("existing_point_colors", []),
        )
        svg_x = float(entry["svg_x"])
        svg_y = float(entry["svg_y"])
        if href:
            g_attrs: dict[str, str] = {}
            if clip_path:
                g_attrs["clip-path"] = clip_path
            g = ET.SubElement(update_group, f"{{{SVG_NS}}}g", g_attrs)
            use = ET.SubElement(g, f"{{{SVG_NS}}}use")
            use.set(f"{{{XLINK_NS}}}href", href)
            use.set("x", f"{svg_x:.6f}")
            use.set("y", f"{svg_y:.6f}")
            use.set("style", point_style)
        else:
            radius = _fallback_radius(mapping_info)
            circle = ET.SubElement(update_group, f"{{{SVG_NS}}}circle")
            circle.set("cx", f"{svg_x:.6f}")
            circle.set("cy", f"{svg_y:.6f}")
            circle.set("r", f"{radius:.6f}")
            circle.set(
                "style",
                point_style
                or f"fill: {point_fill}; stroke: #000000; stroke-width: 0.5; fill-opacity: 0.85",
            )

    svg_out, png_out = default_output_paths(svg_path, chart_type)
    target_svg = svg_output_path or svg_out
    target_png = output_path or png_out
    os.makedirs(os.path.dirname(target_svg), exist_ok=True)
    tree.write(target_svg, encoding="utf-8", xml_declaration=True)
    return render_svg_to_png(target_svg, target_png)


def _point_requested_color(point: dict[str, Any]) -> str:
    if not isinstance(point, dict):
        return ""
    for key in ("color", "point_color", "fill", "rgb"):
        value = point.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _select_marker_info(
    *,
    axes: ET.Element,
    requested_color: Any,
    existing_colors: list[str],
) -> tuple[str | None, str | None, str, str]:
    marker_templates = _extract_marker_templates(axes)
    aggregated_styles: dict[str, str] = {}
    for template in marker_templates:
        for fill, style in template["styles_by_fill"].items():
            aggregated_styles.setdefault(fill, style)

    point_fill, point_style = _resolve_new_point_style(
        requested_color=requested_color,
        existing_colors=existing_colors,
        template_styles_by_fill=aggregated_styles,
    )
    chosen_template = _pick_marker_template(marker_templates, point_fill)
    if not chosen_template:
        return None, None, point_fill, point_style
    return chosen_template.get("href"), chosen_template.get("clip_path"), point_fill, point_style


def _extract_marker_templates(axes: ET.Element) -> list[dict[str, Any]]:
    templates: list[dict[str, Any]] = []
    for path_collection in _iter_path_collections(axes):
        href = None
        clip_path = None
        styles_by_fill: dict[str, str] = {}
        for use in path_collection.findall(f'.//{{{SVG_NS}}}use'):
            style = str(use.get("style") or "").strip()
            fill = _extract_fill_from_style(style)
            if fill and style and fill not in styles_by_fill:
                styles_by_fill[fill] = style
            if href is None:
                href = use.get(f"{{{XLINK_NS}}}href") or use.get("href")
            if clip_path is None:
                parent = _find_parent_group(path_collection, use)
                if parent is not None:
                    clip_path = parent.get("clip-path")
        if href:
            templates.append(
                {
                    "id": str(path_collection.get("id") or ""),
                    "href": href,
                    "clip_path": clip_path,
                    "styles_by_fill": styles_by_fill,
                }
            )
    return templates


def _pick_marker_template(marker_templates: list[dict[str, Any]], target_fill: str) -> dict[str, Any] | None:
    if target_fill:
        for template in marker_templates:
            if target_fill in template.get("styles_by_fill", {}):
                return template
    return marker_templates[0] if marker_templates else None


def _find_parent_group(path_collection: ET.Element, use: ET.Element) -> ET.Element | None:
    for group in path_collection.findall(f'.//{{{SVG_NS}}}g'):
        if use in list(group):
            return group
    return None


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


def _resolve_new_point_style(
    requested_color: Any,
    existing_colors: list[str],
    template_styles_by_fill: dict[str, str],
) -> tuple[str, str]:
    normalized_existing = [str(color or "").strip().lower() for color in existing_colors if str(color or "").strip()]
    target_fill = _resolve_requested_color(requested_color, normalized_existing)
    if not target_fill and normalized_existing:
        target_fill = Counter(normalized_existing).most_common(1)[0][0]
    if not target_fill:
        target_fill = "#ff1493"

    template_style = template_styles_by_fill.get(target_fill)
    if not template_style and template_styles_by_fill:
        fallback_fill = next(iter(template_styles_by_fill))
        template_style = template_styles_by_fill[fallback_fill]
    if template_style:
        return target_fill, _replace_style_fill(template_style, target_fill)
    return target_fill, f"fill: {target_fill}; stroke: #000000; stroke-width: 0.5; fill-opacity: 0.85"


def _resolve_requested_color(requested_color: Any, existing_colors: list[str]) -> str:
    selector = str(requested_color or "").strip().lower()
    if not selector:
        return ""
    if _is_hex_color(selector):
        return selector
    selector_aliases = _color_aliases(selector)
    for fill in existing_colors:
        if fill == selector:
            return fill
        if selector_aliases & _color_aliases(fill):
            return fill
    return ""


def _replace_style_fill(style: str, fill: str) -> str:
    updated = _replace_style_attr(style, "fill", fill)
    return updated or f"fill: {fill}; fill-opacity: 0.85"


def _replace_style_attr(style: str, name: str, value: str) -> str:
    pattern = rf"(^|;)\s*{re.escape(name)}\s*:\s*[^;]+"
    replacement = rf"\1{name}: {value}"
    if re.search(pattern, style):
        return re.sub(pattern, replacement, style)
    if style and not style.rstrip().endswith(";"):
        style = style.rstrip() + "; "
    return f"{style}{name}: {value}".strip()


def _extract_fill_from_style(style: str) -> str:
    match = re.search(r"fill:\s*(#[0-9a-fA-F]{3,6})", style)
    if match:
        return match.group(1).lower()
    return ""


def _is_hex_color(value: str) -> bool:
    return re.fullmatch(r"#[0-9a-f]{3}(?:[0-9a-f]{3})?", value) is not None


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
        "white": {"#ffffff"},
        "brown": {"#8c564b", "#a0522d"},
    }
    for name, values in canonical.items():
        if token == name or token in values:
            aliases.add(name)
            aliases.update(values)
    return aliases
