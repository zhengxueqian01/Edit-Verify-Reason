from __future__ import annotations

import json
import os
import re
from typing import Any


def render_bar_from_text(
    text_spec: str,
    llm: Any | None = None,
    output_path: str | None = None,
    image_size: tuple[int, int] = (640, 420),
) -> tuple[str, dict[str, Any]]:
    categories = []
    if llm is not None:
        categories = _parse_with_llm(text_spec, llm)

    if not categories:
        categories = _parse_with_regex(text_spec)

    if not categories:
        raise ValueError("No bar categories parsed from text_spec.")

    labels = [item["label"] for item in categories]
    values = [item["value"] for item in categories]

    width, height = image_size
    dpi = 100
    fig_width = width / dpi
    fig_height = height / dpi

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    colors = [palette[idx % len(palette)] for idx in range(len(labels))]
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    ax.bar(labels, values, color=colors)
    _apply_y_padding(ax, values, 0.2)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=30)

    target = output_path or _default_output_path()
    os.makedirs(os.path.dirname(target), exist_ok=True)
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    return target, {"categories": categories}


def _parse_with_llm(text_spec: str, llm: Any) -> list[dict[str, Any]]:
    prompt = (
        "You are extracting bar chart data from text. "
        "Return JSON only with key categories: list of {label, value}."
        f"\nText: {text_spec}"
    )
    try:
        response = llm.invoke(prompt)
    except Exception:
        return []
    content = getattr(response, "content", "")
    payload = _safe_json_loads(content)
    if not payload:
        return []
    raw = payload.get("categories", [])
    cleaned = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        try:
            value = float(item.get("value"))
        except (TypeError, ValueError):
            continue
        if label:
            cleaned.append({"label": label, "value": value})
    return cleaned


def _parse_with_regex(text_spec: str) -> list[dict[str, Any]]:
    pairs = re.findall(r"([A-Za-z0-9_\u4e00-\u9fff\-]+)\s*[:=]\s*(-?\d+(?:\.\d+)?)", text_spec)
    return [{"label": label, "value": float(val)} for label, val in pairs]


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


def _apply_y_padding(ax, values: list[float], padding_ratio: float) -> None:
    if not values:
        return
    y_min = min(values)
    y_max = max(values)
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    span = y_max - y_min
    pad = span * padding_ratio
    ax.set_ylim(y_min - pad, y_max + pad)


def _default_output_path() -> str:
    return os.path.join("output", "bar", "text_bar.png")
