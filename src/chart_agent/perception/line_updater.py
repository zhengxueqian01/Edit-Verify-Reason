from __future__ import annotations

import json
import os
import re
import xml.etree.ElementTree as ET
from typing import Any

from chart_agent.prompts.prompt import build_line_update_parse_prompt

from PIL import Image, ImageDraw

SVG_NS = "http://www.w3.org/2000/svg"


def update_line_png(
    image_path: str,
    svg_path: str,
    question: str,
    mapping_info: dict[str, Any],
    output_path: str | None = None,
    llm: Any | None = None,
) -> str:
    x_ticks = mapping_info.get("x_ticks", [])
    y_ticks = mapping_info.get("y_ticks", [])
    if len(x_ticks) < 2 or len(y_ticks) < 2:
        raise ValueError("Insufficient axis ticks for line mapping.")

    view_box = _get_view_box(svg_path)
    if view_box is None:
        raise ValueError("SVG viewBox not found.")

    values, llm_meta = _parse_values(question, llm)
    if llm_meta:
        mapping_info["llm_meta"] = llm_meta

    if not values:
        raise ValueError("No new series values found in question.")

    x_positions = _compute_x_positions(values, x_ticks)
    y_positions = [_data_to_pixel(val, y_ticks) for val in values]

    img = Image.open(image_path).convert("RGBA")
    svg_min_x, svg_min_y, svg_w, svg_h = view_box
    scale_x = img.width / svg_w
    scale_y = img.height / svg_h

    points_px = [
        (
            (x - svg_min_x) * scale_x,
            (y - svg_min_y) * scale_y,
        )
        for x, y in zip(x_positions, y_positions)
    ]

    line_style = _extract_line_style(svg_path)
    line_color = _parse_hex_color(line_style.get("stroke", "#dc143c"))
    line_width = _scale_line_width(line_style.get("stroke_width", 2.0), scale_x, scale_y)

    draw = ImageDraw.Draw(img)
    draw.line(points_px, fill=line_color, width=line_width)

    target = output_path or _default_output_path(image_path)
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


def _parse_values(question: str, llm: Any | None) -> tuple[list[float], dict[str, Any] | None]:
    if llm is not None:
        result = _parse_with_llm(question, llm)
        if result is not None:
            values, meta = result
            if values:
                return values, meta

    values = _parse_with_regex(question)
    meta = {"llm_used": llm is not None, "llm_success": False}
    return values, meta


def _parse_with_llm(
    question: str, llm: Any
) -> tuple[list[float], dict[str, Any]] | None:
    prompt = build_line_update_parse_prompt(question=question)
    try:
        response = llm.invoke(prompt)
    except Exception:
        return None
    content = getattr(response, "content", "")
    payload = _safe_json_loads(content)
    if not payload:
        return None
    raw = payload.get("values", [])
    values = _coerce_values(raw)
    return values, {"llm_used": True, "llm_success": True, "llm_raw": content}


def _parse_with_regex(question: str) -> list[float]:
    match = re.search(r"\[(.*?)\]", question, re.DOTALL)
    if match:
        candidates = re.findall(r"-?\d+(?:\.\d+)?", match.group(1))
        return [float(val) for val in candidates]
    candidates = re.findall(r"-?\d+(?:\.\d+)?", question)
    return [float(val) for val in candidates]


def _coerce_values(raw: Any) -> list[float]:
    if not isinstance(raw, list):
        return []
    values = []
    for item in raw:
        try:
            values.append(float(item))
        except (TypeError, ValueError):
            continue
    return values


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


def _compute_x_positions(values: list[float], x_ticks: list[tuple[float, float]]) -> list[float]:
    if len(values) == len(x_ticks):
        return [tick[0] for tick in x_ticks]

    x_min = min(t[0] for t in x_ticks)
    x_max = max(t[0] for t in x_ticks)
    if len(values) == 1:
        return [(x_min + x_max) / 2.0]
    step = (x_max - x_min) / (len(values) - 1)
    return [x_min + idx * step for idx in range(len(values))]


def _data_to_pixel(value: float, ticks: list[tuple[float, float]]) -> float:
    ticks_sorted = sorted(ticks, key=lambda t: t[1])
    for idx in range(len(ticks_sorted) - 1):
        pixel1, data1 = ticks_sorted[idx]
        pixel2, data2 = ticks_sorted[idx + 1]
        if data1 <= value <= data2 or data2 <= value <= data1:
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


def _default_output_path(image_path: str) -> str:
    base, ext = os.path.splitext(os.path.basename(image_path))
    return os.path.join("output", "line", f"{base}_updated{ext or '.png'}")


def _extract_line_style(svg_path: str) -> dict[str, float | str]:
    tree = ET.parse(svg_path)
    root = tree.getroot()
    axes = root.find(f'.//{{{SVG_NS}}}g[@id="axes_1"]')
    if axes is None:
        return {}
    for g in axes.findall(f'.//{{{SVG_NS}}}g'):
        gid = g.get("id", "")
        if not gid.startswith("line2d_"):
            continue
        path = g.find(f'.//{{{SVG_NS}}}path')
        if path is None:
            continue
        style = path.get("style", "")
        stroke = _extract_style_value(style, "stroke")
        stroke_width = _extract_style_value(style, "stroke-width")
        if stroke:
            return {
                "stroke": stroke,
                "stroke_width": float(stroke_width) if stroke_width else 2.0,
            }
    return {}


def _extract_style_value(style: str, key: str) -> str | None:
    match = re.search(rf"{re.escape(key)}:\\s*([^;]+)", style)
    if match:
        return match.group(1).strip()
    return None


def _parse_hex_color(value: str) -> tuple[int, int, int, int]:
    value = value.strip()
    if value.startswith("#"):
        hex_value = value[1:]
        if len(hex_value) == 6:
            r = int(hex_value[0:2], 16)
            g = int(hex_value[2:4], 16)
            b = int(hex_value[4:6], 16)
            return r, g, b, 220
    return 220, 20, 60, 220


def _scale_line_width(stroke_width: float | str, scale_x: float, scale_y: float) -> int:
    try:
        width = float(stroke_width)
    except (TypeError, ValueError):
        width = 2.0
    scale = (scale_x + scale_y) / 2.0
    return max(1, int(round(width * scale)))
