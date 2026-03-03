from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Any

from PIL import Image, ImageDraw

SVG_NS = "http://www.w3.org/2000/svg"


def update_scatter_png(
    image_path: str,
    svg_path: str,
    new_points: list[dict[str, float]],
    mapping_info: dict[str, Any],
    output_path: str | None = None,
    chart_type: str = "scatter",
) -> str:
    if not new_points:
        raise ValueError("No new points to render.")

    x_ticks = mapping_info.get("x_ticks", [])
    y_ticks = mapping_info.get("y_ticks", [])
    if len(x_ticks) < 2 or len(y_ticks) < 2:
        raise ValueError("Insufficient ticks to map data points.")

    view_box = _get_view_box(svg_path)
    if view_box is None:
        raise ValueError("SVG viewBox not found.")

    points_svg = []
    for point in new_points:
        x_val = float(point["x"])
        y_val = float(point["y"])
        svg_x = _interpolate_axis(x_val, x_ticks)
        svg_y = _interpolate_axis(y_val, y_ticks)
        points_svg.append((svg_x, svg_y))

    img = Image.open(image_path).convert("RGBA")
    svg_min_x, svg_min_y, svg_w, svg_h = view_box
    scale_x = img.width / svg_w
    scale_y = img.height / svg_h

    radius = _estimate_radius(mapping_info, scale_x, scale_y)

    new_point_color = (255, 20, 147, 220)

    draw = ImageDraw.Draw(img)
    size = max(4, int(round(radius * 2)))
    half_size = size // 2
    for svg_x, svg_y in points_svg:
        px = (svg_x - svg_min_x) * scale_x
        py = (svg_y - svg_min_y) * scale_y
        draw.ellipse(
            [px - half_size, py - half_size, px + half_size, py + half_size],
            fill=new_point_color,
            outline=(0, 0, 0),
            width=1,
        )

    target = output_path or _default_output_path(image_path, chart_type)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    img.save(target)
    return target


def _get_view_box(svg_path: str) -> tuple[float, float, float, float] | None:
    tree = ET.parse(svg_path)
    root = tree.getroot()
    view_box = root.get("viewBox")
    if not view_box:
        return None
    parts = view_box.strip().split()
    if len(parts) != 4:
        return None
    try:
        return tuple(float(p) for p in parts)  # type: ignore[return-value]
    except ValueError:
        return None


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


def _estimate_radius(mapping_info: dict[str, Any], scale_x: float, scale_y: float) -> float:
    point_size_svg = mapping_info.get("point_size_svg")
    if not point_size_svg:
        return 4.0
    scale = (scale_x + scale_y) / 2.0
    return max(2.0, (float(point_size_svg) * scale) / 2.0)


def _default_output_path(image_path: str, chart_type: str) -> str:
    base, ext = os.path.splitext(os.path.basename(image_path))
    return os.path.join("output", chart_type, f"{base}_updated{ext or '.png'}")
