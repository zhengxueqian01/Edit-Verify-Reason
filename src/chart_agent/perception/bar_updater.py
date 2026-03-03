from __future__ import annotations

import json
import os
import re
import xml.etree.ElementTree as ET
from typing import Any

from PIL import Image, ImageDraw

SVG_NS = "http://www.w3.org/2000/svg"


def update_bar_png(
    image_path: str,
    svg_path: str,
    question: str,
    mapping_info: dict[str, Any],
    output_path: str | None = None,
    llm: Any | None = None,
) -> str:
    bars = mapping_info.get("bars", [])
    y_ticks = mapping_info.get("y_ticks", [])
    if not bars or len(y_ticks) < 2:
        raise ValueError("Insufficient bar mapping info.")

    view_box = _get_view_box(svg_path)
    if view_box is None:
        raise ValueError("SVG viewBox not found.")

    label, update_value, mode, llm_meta = _parse_update(question, bars, llm)
    if llm_meta:
        mapping_info["llm_meta"] = llm_meta
    if not label:
        raise ValueError("No matching category label found in question.")
    if update_value is None:
        raise ValueError("No update value found in question.")

    bar = next(b for b in bars if b["label"] == label)
    current_value = _pixel_to_data(bar["y_min"], y_ticks)
    if mode == "absolute":
        new_value = update_value
    else:
        new_value = current_value + update_value

    if new_value <= current_value:
        raise ValueError("Only positive bar increases are supported.")

    new_top = _data_to_pixel(new_value, y_ticks)

    img = Image.open(image_path).convert("RGBA")
    svg_min_x, svg_min_y, svg_w, svg_h = view_box
    scale_x = img.width / svg_w
    scale_y = img.height / svg_h

    x0 = (bar["x_min"] - svg_min_x) * scale_x
    x1 = (bar["x_max"] - svg_min_x) * scale_x
    y_old = (bar["y_min"] - svg_min_y) * scale_y
    y_new = (new_top - svg_min_y) * scale_y

    fill = _parse_hex_color(bar.get("fill", "#1f77b4"))

    draw = ImageDraw.Draw(img)
    draw.rectangle([x0, y_new, x1, y_old], fill=fill, outline=(0, 0, 0, 255), width=2)

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


def _match_label(question: str, labels: list[str]) -> str | None:
    lowered = question.lower()
    for label in labels:
        if label.lower() in lowered:
            return label
    return None


def _extract_update_value(question: str) -> tuple[float | None, str]:
    patterns = [
        r"(?:increase|add|增(?:加|长)|提升|增加)\s*(?:by\s*)?(-?\d+(?:\.\d+)?)",
        r"(?:=|to|为|变为)\s*(-?\d+(?:\.\d+)?)",
    ]
    for idx, pattern in enumerate(patterns):
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            return value, "relative" if idx == 0 else "absolute"

    match = re.search(r"(-?\d+(?:\.\d+)?)", question)
    if match:
        return float(match.group(1)), "relative"
    return None, "relative"


def _parse_update(
    question: str, bars: list[dict[str, Any]], llm: Any | None
) -> tuple[str | None, float | None, str, dict[str, Any] | None]:
    labels = [b["label"] for b in bars]
    if llm is not None:
        result = _parse_with_llm(question, labels, llm)
        if result is not None:
            label, value, mode, llm_meta = result
            if label and value is not None:
                return label, value, mode, llm_meta

    label = _match_label(question, labels)
    value, mode = _extract_update_value(question)
    return label, value, mode, {"llm_used": llm is not None, "llm_success": False}


def _parse_with_llm(
    question: str, labels: list[str], llm: Any
) -> tuple[str | None, float | None, str, dict[str, Any]] | None:
    prompt = (
        "You are parsing a bar chart update request. "
        "Choose the target category from this list and extract the update value. "
        "Return JSON only with keys: label, mode (relative|absolute), value. "
        f"\nCategories: {labels}\nQuestion: {question}"
    )
    try:
        response = llm.invoke(prompt)
    except Exception:
        return None
    content = getattr(response, "content", "")
    payload = _safe_json_loads(content)
    if not payload:
        return None
    label = payload.get("label")
    mode = payload.get("mode", "relative")
    value = payload.get("value")
    if label not in labels:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if mode not in ("relative", "absolute"):
        mode = "relative"
    return label, value, mode, {"llm_used": True, "llm_success": True, "llm_raw": content}


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


def _parse_hex_color(value: str) -> tuple[int, int, int, int]:
    value = value.strip()
    if value.startswith("#"):
        hex_value = value[1:]
        if len(hex_value) == 6:
            r = int(hex_value[0:2], 16)
            g = int(hex_value[2:4], 16)
            b = int(hex_value[4:6], 16)
            return r, g, b, 220
    return 31, 119, 180, 220


def _default_output_path(image_path: str) -> str:
    base, ext = os.path.splitext(os.path.basename(image_path))
    return os.path.join("output", "bar", f"{base}_updated{ext or '.png'}")
