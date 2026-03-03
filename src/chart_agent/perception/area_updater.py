from __future__ import annotations

import json
import os
import re
import xml.etree.ElementTree as ET
from typing import Any

from PIL import Image, ImageDraw

SVG_NS = "http://www.w3.org/2000/svg"


def update_area_png(
    image_path: str,
    svg_path: str,
    question: str,
    mapping_info: dict[str, Any],
    output_path: str | None = None,
    llm: Any | None = None,
) -> str:
    top_boundary = mapping_info.get("area_top_boundary", [])
    y_ticks = mapping_info.get("y_ticks", [])
    if not top_boundary or len(y_ticks) < 2:
        raise ValueError("Insufficient area mapping info.")

    view_box = _get_view_box(svg_path)
    if view_box is None:
        raise ValueError("SVG viewBox not found.")

    values, llm_meta = _parse_values(question, llm)
    if llm_meta:
        mapping_info["llm_meta"] = llm_meta

    if not values:
        raise ValueError("No new series values found in question.")

    top_points = _ensure_sorted(top_boundary)
    if len(values) != len(top_points):
        values = _resample_values(values, len(top_points))

    base_data = [_pixel_to_data(p[1], y_ticks) for p in top_points]
    new_data = [base + val for base, val in zip(base_data, values)]
    new_top = [
        (x, _data_to_pixel(y_val, y_ticks)) for (x, _), y_val in zip(top_points, new_data)
    ]

    polygon = new_top + list(reversed(top_points))

    img = Image.open(image_path).convert("RGBA")
    svg_min_x, svg_min_y, svg_w, svg_h = view_box
    scale_x = img.width / svg_w
    scale_y = img.height / svg_h

    polygon_px = [
        ((x - svg_min_x) * scale_x, (y - svg_min_y) * scale_y) for x, y in polygon
    ]

    fill = _choose_fill(mapping_info)

    draw = ImageDraw.Draw(img)
    draw.polygon(polygon_px, fill=fill, outline=(0, 0, 0, 180))

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
    prompt = (
        "You are parsing a stacked area chart update. "
        "Extract the new series values in order. "
        "Return JSON only with keys: values (list of numbers)."
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


def _resample_values(values: list[float], target_len: int) -> list[float]:
    if target_len <= 0:
        return []
    if not values:
        return []
    if len(values) == target_len:
        return values
    if len(values) == 1:
        return [values[0] for _ in range(target_len)]

    src_len = len(values)
    resampled = []
    for idx in range(target_len):
        t = idx * (src_len - 1) / (target_len - 1)
        left = int(t)
        right = min(left + 1, src_len - 1)
        if right == left:
            resampled.append(values[left])
            continue
        ratio = t - left
        resampled.append(values[left] + ratio * (values[right] - values[left]))
    return resampled


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


def _pixel_to_data(pixel: float, ticks: list[tuple[float, float]]) -> float:
    ticks_sorted = sorted(ticks, key=lambda t: t[0])
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


def _ensure_sorted(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(points) < 2:
        return points
    if points[0][0] <= points[-1][0]:
        return points
    return list(reversed(points))


def _choose_fill(mapping_info: dict[str, Any]) -> tuple[int, int, int, int]:
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    existing = mapping_info.get("area_fills", [])
    for color in palette:
        if color not in existing:
            return _parse_hex_color(color)
    return _parse_hex_color(palette[-1])


def _parse_hex_color(value: str) -> tuple[int, int, int, int]:
    value = value.strip()
    if value.startswith("#"):
        hex_value = value[1:]
        if len(hex_value) == 6:
            r = int(hex_value[0:2], 16)
            g = int(hex_value[2:4], 16)
            b = int(hex_value[4:6], 16)
            return r, g, b, 190
    return 31, 119, 180, 190


def _default_output_path(image_path: str) -> str:
    base, ext = os.path.splitext(os.path.basename(image_path))
    return os.path.join("output", "area", f"{base}_updated{ext or '.png'}")
