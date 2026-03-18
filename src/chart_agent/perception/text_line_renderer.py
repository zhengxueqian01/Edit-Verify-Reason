from __future__ import annotations

import json
import math
import os
import re
from typing import Any


def render_line_from_text(
    text_spec: str,
    llm: Any | None = None,
    output_path: str | None = None,
    image_size: tuple[int, int] = (640, 420),
) -> tuple[str, dict[str, Any] | None]:
    series = []
    llm_meta = None
    x_values = None
    if llm is not None:
        series, x_values, llm_meta = _parse_with_llm(text_spec, llm)

    if not series:
        series, x_values = _parse_with_json(text_spec)

    if not series:
        series = _parse_with_regex(text_spec)

    if not series:
        raise ValueError("No line series parsed from text_spec.")

    width, height = image_size
    dpi = 100
    fig_width = width / dpi
    fig_height = height / dpi

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    if x_values is None:
        max_len = max(len(item["values"]) for item in series)
        x_values = list(range(1, max_len + 1))

    for idx, item in enumerate(series):
        values = item["values"]
        label = item["label"]
        color = palette[idx % len(palette)]
        ax.plot(
            x_values[: len(values)],
            values,
            color=color,
            linewidth=2,
            marker="o",
            markersize=3,
            label=label,
        )

    _apply_y_padding(ax, [v for item in series for v in item["values"]], 0.2)
    if x_values:
        x_min = min(x_values)
        x_max = max(x_values)
        pad = (x_max - x_min) * 0.05 if x_max != x_min else 1.0
        ax.set_xlim(x_min - pad, x_max + pad)
        if all(float(v).is_integer() for v in x_values):
            years = sorted(set(int(v) for v in x_values))
            if len(years) <= 8:
                ticks = years
            else:
                step = max(1, math.ceil(len(years) / 8))
                ticks = years[::step]
                if ticks[-1] != years[-1]:
                    ticks.append(years[-1])
            ax.set_xticks(ticks)
    if len(series) > 1:
        ax.legend(fontsize=8)
    ax.set_xlabel("")
    ax.set_ylabel("")
    _lock_in_matplotlib_y_axis_format(fig, ax)

    target = output_path or _default_output_path()
    os.makedirs(os.path.dirname(target), exist_ok=True)
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    meta = {"series": series, "years": x_values}
    if llm_meta:
        meta["llm_meta"] = llm_meta
    return target, meta


def _parse_with_llm(
    text_spec: str, llm: Any
) -> tuple[list[dict[str, Any]], list[Any] | None, dict[str, Any] | None]:
    prompt = (
        "Return ONLY valid JSON. Do not include any extra text.\n"
        "Schema: {\"years\": [year,...], \"series\": [{\"label\": string, \"values\": [number, ...]}]}\n"
        "If labels are missing, use \"Series 1\", \"Series 2\", etc.\n"
        "If years are present in the text, include them in the years field in order.\n"
        f"Text: {text_spec}"
    )
    try:
        response = llm.invoke(prompt)
    except Exception:
        return [], None, None
    content = getattr(response, "content", "")
    payload = _safe_json_loads(content)
    if not payload:
        return [], None, {"llm_used": True, "llm_success": False, "llm_raw": content}
    raw = payload.get("series", [])
    series = _coerce_series(raw)
    years = payload.get("years")
    if not isinstance(years, list):
        years = None
    return series, years, {"llm_used": True, "llm_success": True, "llm_raw": content}


def _parse_with_json(text_spec: str) -> tuple[list[dict[str, Any]], list[Any] | None]:
    data = _load_json_array(text_spec)
    if data is None:
        return [], None
    if not isinstance(data, list):
        return [], None
    if not data or not isinstance(data[0], dict):
        return [], None
    year_key = None
    for key in data[0].keys():
        if key.lower() == "year":
            year_key = key
            break
    keys = [k for k in data[0].keys() if k != year_key]
    series = []
    for key in keys:
        values = []
        for row in data:
            try:
                values.append(float(row[key]))
            except Exception:
                continue
        if values:
            series.append({"label": key, "values": values})
    x_values = None
    if year_key:
        x_values = []
        for row in data:
            x_values.append(row.get(year_key))
    return series, x_values


def _load_json_array(text_spec: str) -> list[Any] | None:
    try:
        parsed = json.loads(text_spec)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    start = text_spec.find("[")
    end = text_spec.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text_spec[start : end + 1]
    try:
        parsed = json.loads(snippet)
    except Exception:
        return None
    return parsed if isinstance(parsed, list) else None


def _parse_with_regex(text_spec: str) -> list[dict[str, Any]]:
    match = re.search(r"\[(.*?)\]", text_spec, re.DOTALL)
    if match:
        candidates = re.findall(r"-?\d+(?:\.\d+)?", match.group(1))
        values = [float(val) for val in candidates]
        if values:
            return [{"label": "Series 1", "values": values}]
    return []


def _coerce_series(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    series = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "Series"))
        values = item.get("values", [])
        coerced = []
        if isinstance(values, list):
            for v in values:
                try:
                    coerced.append(float(v))
                except (TypeError, ValueError):
                    continue
        if coerced:
            series.append({"label": label, "values": coerced})
    return series


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
    span = y_max - y_min
    if span:
        lower_pad = span * 0.05
        upper_pad = span * 0.15
    else:
        margin = max(abs(y_max) * 0.05, 1.0)
        lower_pad = margin
        upper_pad = margin * 3
    ax.set_ylim(y_min - lower_pad, y_max + upper_pad)


def _lock_in_matplotlib_y_axis_format(fig, ax) -> dict[str, Any]:
    fig.canvas.draw()
    formatter = ax.yaxis.get_major_formatter()
    offset_text = formatter.get_offset() if hasattr(formatter, "get_offset") else ""
    tick_values = [float(v) for v in ax.get_yticks()]
    tick_labels = [tick.get_text() for tick in ax.get_yticklabels()]

    if hasattr(formatter, "set_locs"):
        formatter.set_locs(ax.get_yticks())
    ax.yaxis.set_major_formatter(formatter)

    return {
        "offset_text": offset_text,
        "tick_values": tick_values,
        "tick_labels": tick_labels,
    }


def _default_output_path() -> str:
    return os.path.join("output", "line", "text_line.png")
