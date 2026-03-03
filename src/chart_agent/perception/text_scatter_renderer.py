from __future__ import annotations

import os
import re
from typing import Any

_POINT_PATTERN = re.compile(r"\((-?\d+(?:\.\d+)?)[\s,]+(-?\d+(?:\.\d+)?)\)")
_BRACKET_PAIR_PATTERN = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)


def render_scatter_from_text(
    text_spec: str,
    output_path: str | None = None,
    image_size: tuple[int, int] = (432, 432),
) -> tuple[str, dict[str, Any]]:
    points = _parse_points(text_spec)
    if not points:
        raise ValueError("No points found in text_spec.")

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    x_min, x_max = _pad_range(x_min, x_max, 0.2)
    y_min, y_max = _pad_range(y_min, y_max, 0.2)

    width, height = image_size
    dpi = 100
    fig_width = width / dpi
    fig_height = height / dpi

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    ax.scatter(xs, ys, c="#1f77b4", s=20)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, linewidth=0.5, alpha=0.4)

    target = output_path or _default_output_path()
    os.makedirs(os.path.dirname(target), exist_ok=True)
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    return target, {"points": points}


def _parse_points(text_spec: str) -> list[tuple[float, float]]:
    points = []
    for match in _POINT_PATTERN.findall(text_spec):
        points.append((float(match[0]), float(match[1])))
    for match in _BRACKET_PAIR_PATTERN.findall(text_spec):
        points.append((float(match[0]), float(match[1])))
    return points


def _default_output_path() -> str:
    return os.path.join("output", "scatter", "text_scatter.png")


def _pad_range(v_min: float, v_max: float, padding_ratio: float) -> tuple[float, float]:
    span = v_max - v_min
    pad = span * padding_ratio
    return v_min - pad, v_max + pad
