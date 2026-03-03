from __future__ import annotations

import json
import os
import re
import xml.etree.ElementTree as ET
from typing import Any

from chart_agent.perception.svg_renderer import default_output_paths, render_svg_to_png

SVG_NS = "http://www.w3.org/2000/svg"

ET.register_namespace("", SVG_NS)


def _parse_svg_tree(svg_path: str) -> ET.ElementTree:
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    return ET.parse(svg_path, parser=parser)


def update_area_svg(
    svg_path: str,
    question: str,
    mapping_info: dict[str, Any],
    output_path: str | None = None,
    svg_output_path: str | None = None,
    llm: Any | None = None,
) -> str:
    ops = _resolve_area_ops(question)
    if len(ops) > 1:
        return _run_area_ops_sequence(
            svg_path=svg_path,
            question=question,
            mapping_info=mapping_info,
            ops=ops,
            output_path=output_path,
            svg_output_path=svg_output_path,
            llm=llm,
        )
    op = ops[0] if ops else "add"
    if op == "delete":
        return _update_area_remove_series(
            svg_path,
            question,
            mapping_info,
            output_path=output_path,
            svg_output_path=svg_output_path,
            llm=llm,
        )
    if op == "change":
        try:
            return _update_area_year_point(
                svg_path,
                question,
                mapping_info,
                output_path=output_path,
                svg_output_path=svg_output_path,
                llm=llm,
            )
        except ValueError as exc:
            if "No valid area update request found in question." not in str(exc):
                raise
    return _update_area_add_series(
        svg_path,
        question,
        mapping_info,
        output_path=output_path,
        svg_output_path=svg_output_path,
        llm=llm,
    )


def _update_area_add_series(
    svg_path: str,
    question: str,
    mapping_info: dict[str, Any],
    output_path: str | None = None,
    svg_output_path: str | None = None,
    llm: Any | None = None,
) -> str:

    top_boundary = mapping_info.get("area_top_boundary", [])
    y_ticks = mapping_info.get("y_ticks", [])
    if not top_boundary or len(y_ticks) < 2:
        raise ValueError("Insufficient area mapping info.")

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
    new_top = [(x, _data_to_pixel(y_val, y_ticks)) for (x, _), y_val in zip(top_points, new_data)]

    polygon = new_top + list(reversed(top_points))

    with open(svg_path, "r", encoding="utf-8") as handle:
        content = handle.read()

    tree = _parse_svg_tree(svg_path)
    root = tree.getroot()
    axes = root.find(f'.//{{{SVG_NS}}}g[@id="axes_1"]')
    if axes is None:
        raise ValueError("SVG axes group not found.")

    fill = _choose_fill(mapping_info)
    path_d = _format_path(polygon)

    update_group, update_path = _ensure_update_area_path(axes)
    clip_path = _extract_first_area_clip(axes)
    update_path.set("d", path_d)
    update_path.set(
        "style",
        f"fill: {fill}; fill-opacity: 0.75; stroke: #000000; stroke-width: 0.5",
    )
    if clip_path:
        update_path.set("clip-path", clip_path)

    label = _extract_series_label(question)
    if not label and llm is not None:
        label = _parse_label_with_llm(question, llm)
    if label:
        legend, legend_items = _extract_legend_items(root, content)
        if legend is not None:
            _append_legend_item(legend, legend_items, label, fill)

    svg_out, png_out = default_output_paths(svg_path, "area")
    target_svg = svg_output_path or svg_out
    target_png = output_path or png_out
    os.makedirs(os.path.dirname(target_svg), exist_ok=True)
    tree.write(target_svg, encoding="utf-8", xml_declaration=True)
    return render_svg_to_png(target_svg, target_png)


def _resolve_area_ops(question: str) -> list[str]:
    clauses = [c.strip() for c in re.split(r"[；;\n]+", question) if c.strip()]
    ordered: list[str] = []
    for clause in clauses:
        op = _detect_area_op(clause)
        if op:
            ordered.append(op)
    if ordered:
        return ordered
    op = _detect_area_op(question)
    return [op] if op else ["add"]


def _detect_area_op(text: str) -> str | None:
    if _has_delete_intent(text):
        return "delete"
    if _has_year_update(text):
        return "change"
    if _has_add_intent(text):
        return "add"
    return None


def _has_add_intent(question: str) -> bool:
    return bool(re.search(r"(新增|添加|add|append|insert|new\s+series)", question, re.IGNORECASE))


def _refresh_mapping_info(svg_path: str, question: str, llm: Any | None) -> dict[str, Any]:
    from chart_agent.perception.svg_perceiver import perceive_svg

    perceived = perceive_svg(svg_path, question=question, llm=llm)
    mapping = perceived.get("mapping_info")
    return mapping if isinstance(mapping, dict) else {}


def _run_area_ops_sequence(
    *,
    svg_path: str,
    question: str,
    mapping_info: dict[str, Any],
    ops: list[str],
    output_path: str | None,
    svg_output_path: str | None,
    llm: Any | None,
) -> str:
    clauses = [c.strip() for c in re.split(r"[；;\n]+", question) if c.strip()]
    if not clauses:
        clauses = [question]
    svg_out, png_out = default_output_paths(svg_path, "area")
    final_svg = svg_output_path or svg_out
    final_png = output_path or png_out
    os.makedirs(os.path.dirname(final_svg), exist_ok=True)
    os.makedirs(os.path.dirname(final_png), exist_ok=True)

    current_svg = svg_path
    current_mapping = mapping_info
    for idx, op in enumerate(ops):
        is_last = idx == len(ops) - 1
        clause = clauses[min(idx, len(clauses) - 1)]
        step_svg = final_svg if is_last else os.path.join(
            os.path.dirname(final_svg),
            f".area_step_{idx+1}_{os.path.basename(final_svg)}",
        )
        step_png = final_png if is_last else os.path.join(
            os.path.dirname(final_png),
            f".area_step_{idx+1}_{os.path.basename(final_png)}",
        )
        if idx > 0:
            current_mapping = _refresh_mapping_info(current_svg, clause, llm)

        if op == "delete":
            _update_area_remove_series(
                current_svg,
                clause,
                current_mapping,
                output_path=step_png,
                svg_output_path=step_svg,
                llm=llm,
            )
        elif op == "change":
            _update_area_year_point(
                current_svg,
                clause,
                current_mapping,
                output_path=step_png,
                svg_output_path=step_svg,
                llm=llm,
            )
        else:
            _update_area_add_series(
                current_svg,
                clause,
                current_mapping,
                output_path=step_png,
                svg_output_path=step_svg,
                llm=llm,
            )
        current_svg = step_svg

    return final_png


def _find_area_path(axes: ET.Element) -> tuple[ET.Element | None, ET.Element | None]:
    for g in axes.findall(f'.//{{{SVG_NS}}}g'):
        gid = g.get("id", "")
        if not gid.startswith("FillBetweenPolyCollection_"):
            continue
        path = g.find(f'.//{{{SVG_NS}}}path')
        if path is not None:
            return g, path
    update = axes.find(f'.//{{{SVG_NS}}}g[@id="FillBetweenPolyCollection_update"]')
    if update is not None:
        path = update.find(f'.//{{{SVG_NS}}}path')
        if path is not None:
            return update, path
    return None, None


def _ensure_update_area_path(axes: ET.Element) -> tuple[ET.Element, ET.Element]:
    update = axes.find(f'.//{{{SVG_NS}}}g[@id="FillBetweenPolyCollection_update"]')
    if update is None:
        update = ET.SubElement(axes, f"{{{SVG_NS}}}g", {"id": "FillBetweenPolyCollection_update"})
    path = update.find(f'.//{{{SVG_NS}}}path')
    if path is None:
        path = ET.SubElement(update, f"{{{SVG_NS}}}path")
    return update, path


def _extract_first_area_clip(axes: ET.Element) -> str | None:
    for g in axes.findall(f'.//{{{SVG_NS}}}g'):
        gid = g.get("id", "")
        if not gid.startswith("FillBetweenPolyCollection_"):
            continue
        path = g.find(f'.//{{{SVG_NS}}}path')
        if path is None:
            continue
        clip = path.get("clip-path")
        if clip:
            return clip
    return None


def _update_area_remove_series(
    svg_path: str,
    question: str,
    mapping_info: dict[str, Any],
    output_path: str | None,
    svg_output_path: str | None,
    llm: Any | None = None,
) -> str:
    x_ticks = mapping_info.get("x_ticks", [])
    y_ticks = mapping_info.get("y_ticks", [])
    if len(x_ticks) < 2 or len(y_ticks) < 2:
        raise ValueError("Insufficient ticks to map area data.")

    with open(svg_path, "r", encoding="utf-8") as handle:
        content = handle.read()

    tree = _parse_svg_tree(svg_path)
    root = tree.getroot()
    axes = root.find(f'.//{{{SVG_NS}}}g[@id="axes_1"]')
    if axes is None:
        raise ValueError("SVG axes group not found.")

    legend, legend_items = _extract_legend_items(root, content)
    labels = [item["label"] for item in legend_items if item.get("label")]
    label = _match_label_with_llm(question, labels, llm) if llm is not None else None
    if not label and llm is None:
        label = _match_label(question, labels)
    if not label:
        raise ValueError("No matching area series found in question.")

    target_fill = None
    for item in legend_items:
        if item.get("label") == label:
            target_fill = item.get("fill")
            break
    if not target_fill:
        raise ValueError("No matching legend color for selected series.")

    areas = _extract_area_groups(axes)
    if not areas:
        raise ValueError("No stacked area collections found in SVG.")

    target_idx = _find_area_by_fill(areas, target_fill)
    if target_idx is None:
        raise ValueError("No stacked area series matches selected legend color.")

    x_values, series_values = _area_series_values(areas, y_ticks)
    remaining = [idx for idx in range(len(areas)) if idx != target_idx]
    if not remaining:
        raise ValueError("Cannot remove the only remaining area series.")

    cumulative = [0.0 for _ in x_values]
    for new_pos, idx in enumerate(remaining):
        values = series_values[idx]
        top_boundary = []
        bottom_boundary = []
        for x_val, base_val, delta in zip(x_values, cumulative, values):
            bottom_data = base_val
            top_data = base_val + delta
            bottom_boundary.append((x_val, _data_to_pixel(bottom_data, y_ticks)))
            top_boundary.append((x_val, _data_to_pixel(top_data, y_ticks)))
        cumulative = [base + delta for base, delta in zip(cumulative, values)]

        path_d = _format_path(top_boundary + list(reversed(bottom_boundary)))
        area_path = areas[idx]["path"]
        area_path.set("d", path_d)

    for idx, area in enumerate(areas):
        if idx == target_idx:
            axes.remove(area["group"])

    if legend is not None:
        _remove_legend_item(root, legend, legend_items, label)

    svg_out, png_out = default_output_paths(svg_path, "area")
    target_svg = svg_output_path or svg_out
    target_png = output_path or png_out
    os.makedirs(os.path.dirname(target_svg), exist_ok=True)
    tree.write(target_svg, encoding="utf-8", xml_declaration=True)
    return render_svg_to_png(target_svg, target_png)


def _update_area_year_point(
    svg_path: str,
    question: str,
    mapping_info: dict[str, Any],
    output_path: str | None,
    svg_output_path: str | None,
    llm: Any | None,
) -> str:
    x_ticks = mapping_info.get("x_ticks", [])
    y_ticks = mapping_info.get("y_ticks", [])
    if len(x_ticks) < 2 or len(y_ticks) < 2:
        raise ValueError("Insufficient ticks to map area data.")

    with open(svg_path, "r", encoding="utf-8") as handle:
        content = handle.read()

    tree = _parse_svg_tree(svg_path)
    root = tree.getroot()
    axes = root.find(f'.//{{{SVG_NS}}}g[@id="axes_1"]')
    if axes is None:
        raise ValueError("SVG axes group not found.")

    legend, legend_items = _extract_legend_items(root, content)
    labels = [item["label"] for item in legend_items if item.get("label")]
    parsed = _parse_year_value_update(question, labels, llm)
    if parsed is None:
        raise ValueError("No valid area update request found in question.")
    label, year_value, update_value, mode = parsed

    target_fill = None
    for item in legend_items:
        if item.get("label") == label:
            target_fill = item.get("fill")
            break
    if not target_fill:
        raise ValueError("No matching legend color for selected series.")

    areas = _extract_area_groups(axes)
    if not areas:
        raise ValueError("No stacked area collections found in SVG.")

    target_idx = _find_area_by_fill(areas, target_fill)
    if target_idx is None:
        raise ValueError("No stacked area series matches selected legend color.")

    x_values, series_values = _area_series_values(areas, y_ticks)
    target_x = _data_to_pixel(year_value, x_ticks)
    year_idx = min(range(len(x_values)), key=lambda i: abs(x_values[i] - target_x))
    current_val = series_values[target_idx][year_idx]
    new_val = current_val + update_value if mode == "relative" else update_value
    if new_val < 0:
        new_val = 0.0
    series_values[target_idx][year_idx] = new_val

    cumulative = [0.0 for _ in x_values]
    for idx, area in enumerate(areas):
        values = series_values[idx]
        top_boundary = []
        bottom_boundary = []
        for x_val, base_val, delta in zip(x_values, cumulative, values):
            bottom_data = base_val
            top_data = base_val + delta
            bottom_boundary.append((x_val, _data_to_pixel(bottom_data, y_ticks)))
            top_boundary.append((x_val, _data_to_pixel(top_data, y_ticks)))
        cumulative = [base + delta for base, delta in zip(cumulative, values)]
        path_d = _format_path(top_boundary + list(reversed(bottom_boundary)))
        area["path"].set("d", path_d)

    svg_out, png_out = default_output_paths(svg_path, "area")
    target_svg = svg_output_path or svg_out
    target_png = output_path or png_out
    os.makedirs(os.path.dirname(target_svg), exist_ok=True)
    tree.write(target_svg, encoding="utf-8", xml_declaration=True)
    return render_svg_to_png(target_svg, target_png)


def _format_path(points: list[tuple[float, float]]) -> str:
    if not points:
        return ""
    head = points[0]
    segments = [f"M {head[0]:.6f} {head[1]:.6f}"]
    for x_val, y_val in points[1:]:
        segments.append(f"L {x_val:.6f} {y_val:.6f}")
    segments.append("Z")
    return " ".join(segments)


def _parse_values(question: str, llm: Any | None) -> tuple[list[float], dict[str, Any] | None]:
    if llm is None:
        values = _parse_with_regex(question)
        return values, {"llm_used": False, "llm_success": False}
    result = _parse_with_llm(question, llm)
    if result is not None:
        values, meta = result
        if values:
            return values, meta
    return [], {"llm_used": True, "llm_success": False}


def _has_year_update(question: str) -> bool:
    if "[" in question:
        return False
    has_year = bool(re.search(r"\b(19|20)\d{2}\b", question) or re.search(r"\d+\s*年", question))
    if not has_year:
        return False
    # Avoid misclassifying QA text like “哪一年/which year” as a change operation.
    has_update_verb = bool(
        re.search(
            r"(改为|变为|调整到|设为|设置为|increase|decrease|reduce|raise|update|change|set\s+to|to\s+\d)",
            question,
            re.IGNORECASE,
        )
    )
    return has_update_verb


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


def _safe_json_loads(content: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except (TypeError, json.JSONDecodeError):
            return None
    if not isinstance(payload, dict):
        return None
    return payload


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


def _choose_fill(mapping_info: dict[str, Any]) -> str:
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    existing = mapping_info.get("area_fills", [])
    for color in palette:
        if color not in existing:
            return color
    return palette[-1]


def _has_delete_intent(question: str) -> bool:
    return bool(re.search(r"(delete|remove|drop|删|删除|去掉|移除|去除|剔除)", question, re.IGNORECASE))


def _match_label(question: str, labels: list[str]) -> str | None:
    lowered = question.lower()
    for label in labels:
        if label.lower() in lowered:
            return label
    return None


def _match_label_with_llm(question: str, labels: list[str], llm: Any) -> str | None:
    prompt = (
        "Extract the single label to delete from the area chart question.\n"
        f"Candidate labels: {json.dumps(labels, ensure_ascii=False)}\n"
        "Return JSON only with key: label (exact candidate label string).\n"
        f"Question: {question}"
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
    if not isinstance(label, str):
        return None
    label_norm = label.strip().lower()
    for candidate in labels:
        if candidate.lower() == label_norm:
            return candidate
    return None


def _parse_year_value_update(
    question: str, labels: list[str], llm: Any | None
) -> tuple[str, float, float, str] | None:
    if llm is not None:
        result = _parse_update_with_llm(question, llm)
        if result is not None:
            label, year_value, update_value, mode = result
            if label and year_value is not None and update_value is not None:
                return label, year_value, update_value, mode
        # Fallback to rule parsing when LLM output is malformed or incomplete.
    label = _match_label(question, labels)
    year_value = _extract_year(question)
    update_value, mode = _extract_update_value(question)
    if label and year_value is not None and update_value is not None:
        return label, year_value, update_value, mode
    return None


def _extract_series_label(question: str) -> str | None:
    quoted = re.search(r"[\"“”']([^\"“”']+)[\"“”']", question)
    if quoted:
        return quoted.group(1).strip()

    if "：" in question:
        prefix, remainder = question.split("：", 1)
    elif ":" in question:
        prefix, remainder = question.split(":", 1)
    elif "[" in question:
        prefix, remainder = question.split("[", 1)[0], ""
    else:
        return None
    cleaned = re.sub(
        r"^(新增|添加|add|new)\s*(一个|一条)?\s*(类别|系列|area|series)?\s*",
        "",
        prefix,
        flags=re.IGNORECASE,
    ).strip(" -:")
    if cleaned.lower() in {"area", "series", "category", "类别"}:
        cleaned = ""
    if not cleaned and remainder:
        if "：" in remainder:
            cleaned = remainder.split("：", 1)[0]
        elif ":" in remainder:
            cleaned = remainder.split(":", 1)[0]
        elif "[" in remainder:
            cleaned = remainder.split("[", 1)[0]
        else:
            cleaned = remainder
        cleaned = cleaned.strip(" -:")
    return cleaned or None


def _parse_update_with_llm(
    question: str, llm: Any
) -> tuple[str | None, float | None, float | None, str] | None:
    prompt = (
        "Extract an area update request with series label, year and value. "
        "Return JSON only with keys: label, year, value, mode (absolute|relative). "
        f"\nQuestion: {question}"
    )
    try:
        response = llm.invoke(prompt)
    except Exception:
        return None
    content = getattr(response, "content", "")
    payload = _safe_json_loads(content)
    if not isinstance(payload, dict):
        return None
    label = payload.get("label")
    year = payload.get("year")
    value = payload.get("value")
    mode = payload.get("mode", "absolute")
    try:
        year_val = float(year)
    except (TypeError, ValueError):
        year_val = None
    try:
        value_val = float(value)
    except (TypeError, ValueError):
        value_val = None
    if mode not in ("absolute", "relative"):
        mode = "absolute"
    if isinstance(label, str):
        label = label.strip()
    else:
        label = None
    return label, year_val, value_val, mode


def _extract_year(question: str) -> float | None:
    match = re.search(r"(19|20)\d{2}", question)
    if match:
        return float(match.group(0))
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*年", question)
    if match:
        return float(match.group(1))
    return None


def _extract_update_value(question: str) -> tuple[float | None, str]:
    rel_patterns = [
        r"(?:increase|add|increase by|add by|增加|提升|上升|减少|下降|降低)\s*(?:by\s*)?(-?\d+(?:\.\d+)?)",
        r"([+-]\d+(?:\.\d+)?)",
    ]
    abs_patterns = [
        r"(?:=|to|为|变为|改为)\s*(-?\d+(?:\.\d+)?)",
    ]
    for pattern in abs_patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            return float(match.group(1)), "absolute"
    for pattern in rel_patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            if re.search(r"(减少|下降|降低|decrease|reduce)", question, re.IGNORECASE):
                value = -abs(value)
            return value, "relative"
    match = re.search(r"(-?\d+(?:\.\d+)?)", question)
    if match:
        if re.search(r"(增加|提升|上升|减少|下降|降低|increase|add|decrease|reduce)", question, re.IGNORECASE):
            return float(match.group(1)), "relative"
        return float(match.group(1)), "absolute"
    return None, "absolute"


def _parse_label_with_llm(question: str, llm: Any) -> str | None:
    prompt = (
        "Extract the new series/legend label from the question. "
        "Return JSON only with key: label. "
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
    label = payload.get("label")
    if not isinstance(label, str):
        return None
    label = label.strip()
    if not label:
        return None
    if label.lower() in {"area", "series", "category", "类别"}:
        return None
    return label


def _extract_legend_items(
    root: ET.Element, content: str
) -> tuple[ET.Element | None, list[dict[str, Any]]]:
    legend = root.find(f'.//{{{SVG_NS}}}g[@id="legend_1"]')
    if legend is None:
        return None, []

    items: list[dict[str, Any]] = []
    pending_fill = None
    pending_patch = None

    for child in list(legend):
        fill = _extract_fill_from_group(child)
        if fill:
            pending_fill = fill
            pending_patch = child
            continue

        label = _extract_text_label(child, content)
        if label:
            items.append(
                {
                    "label": label,
                    "fill": pending_fill,
                    "text": child,
                    "patch": pending_patch,
                }
            )
            pending_fill = None
            pending_patch = None

    return legend, items


def _append_legend_item(
    legend: ET.Element, items: list[dict[str, Any]], label: str, fill: str
) -> None:
    metrics = _legend_entry_metrics(items)
    text_x, text_y = _legend_text_anchor(items)
    if text_y is None:
        text_x, text_y = _legend_text_anchor_from_legend(legend)
    if metrics is None:
        if text_y is None:
            return
        height = 7.0
        step = _legend_step(items)
        next_y_min = text_y + step - height * 0.85
        x_min = text_x - 28.0
        x_max = text_x - 8.0
    else:
        x_min, x_max, last_y_max, height = metrics
        step = _legend_step(items)
        next_y_min = last_y_max + step

    patch = ET.SubElement(legend, f"{{{SVG_NS}}}g", {"id": "patch_update"})
    path = ET.SubElement(patch, f"{{{SVG_NS}}}path")
    path.set(
        "d",
        f"M {x_min:.6f} {next_y_min:.6f} L {x_max:.6f} {next_y_min:.6f} "
        f"L {x_max:.6f} {next_y_min + height:.6f} L {x_min:.6f} {next_y_min + height:.6f} z",
    )
    path.set("style", f"fill: {fill}")

    text = ET.SubElement(legend, f"{{{SVG_NS}}}text")
    text.set("x", f"{text_x:.6f}")
    text.set("y", f"{next_y_min + height * 0.85:.6f}")
    text.set("font-size", "10")
    text.set("font-family", "Times New Roman")
    text.set("fill", "#000000")
    text.text = label


def _legend_bounds(items: list[dict[str, Any]]) -> tuple[float | None, float | None, float, float]:
    y_min = None
    y_max = None
    x_min = 0.0
    x_max = 0.0
    for item in items:
        patch = item.get("patch")
        if patch is None:
            continue
        path = patch.find(f'.//{{{SVG_NS}}}path')
        if path is None:
            continue
        coords = re.findall(r"-?[\d.]+", path.get("d", ""))
        if len(coords) < 8:
            continue
        xs = [float(c) for c in coords[0::2]]
        ys = [float(c) for c in coords[1::2]]
        x_min = min(xs)
        x_max = max(xs)
        if y_min is None or min(ys) < y_min:
            y_min = min(ys)
        if y_max is None or max(ys) > y_max:
            y_max = max(ys)
    return y_min, y_max, x_min, x_max


def _legend_entry_metrics(
    items: list[dict[str, Any]],
) -> tuple[float, float, float, float] | None:
    boxes = []
    for item in items:
        patch = item.get("patch")
        if patch is None:
            continue
        path = patch.find(f'.//{{{SVG_NS}}}path')
        if path is None:
            continue
        coords = re.findall(r"-?[\d.]+", path.get("d", ""))
        if len(coords) < 4:
            continue
        xs = [float(c) for c in coords[0::2]]
        ys = [float(c) for c in coords[1::2]]
        boxes.append((min(xs), max(xs), min(ys), max(ys)))

    if not boxes:
        return None

    x_min, x_max = boxes[0][0], boxes[0][1]
    last_y_max = max(box[3] for box in boxes)
    heights = [box[3] - box[2] for box in boxes if box[3] > box[2]]
    if heights:
        heights.sort()
        height = heights[len(heights) // 2]
    else:
        height = 7.0
    return x_min, x_max, last_y_max, height


def _legend_step(items: list[dict[str, Any]]) -> float:
    ys = _legend_text_positions(items)
    if len(ys) >= 2:
        return ys[1] - ys[0]
    return 14.0


def _legend_text_anchor(items: list[dict[str, Any]]) -> tuple[float, float | None]:
    for item in items:
        text = item.get("text")
        if text is None:
            continue
        g = text.find(f'.//{{{SVG_NS}}}g')
        if g is None:
            continue
        transform = g.get("transform", "")
        match = re.search(r"translate\(([-\d.]+)\s+([-\d.]+)\)", transform)
        if match:
            return float(match.group(1)), float(match.group(2))
    return 0.0, None


def _legend_text_anchor_from_legend(legend: ET.Element) -> tuple[float, float | None]:
    for g in legend.findall(f'.//{{{SVG_NS}}}g'):
        transform = g.get("transform", "")
        match = re.search(r"translate\(([-\d.]+)\s+([-\d.]+)\)", transform)
        if match:
            return float(match.group(1)), float(match.group(2))
    return 0.0, None


def _legend_text_positions(items: list[dict[str, Any]]) -> list[float]:
    ys = []
    for item in items:
        _, y = _legend_text_anchor([item])
        if y is not None:
            ys.append(y)
    return sorted(set(ys))


def _extract_text_label(group: ET.Element, content: str) -> str | None:
    gid = group.get("id", "")
    if not gid.startswith("text_"):
        return None
    pattern = rf'<g\s+id="{re.escape(gid)}"[^>]*>.*?<!--\s*(.*?)\s*-->'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _extract_fill_from_group(group: ET.Element) -> str | None:
    path = group.find(f'.//{{{SVG_NS}}}path')
    if path is None:
        return None
    fill = path.get("fill")
    if fill:
        return fill
    style = path.get("style", "")
    return _extract_fill_color(style)


def _extract_fill_color(style: str) -> str | None:
    match = re.search(r"fill:\s*(#[0-9a-fA-F]{6})", style)
    if match:
        return match.group(1)
    return None


def _extract_area_groups(axes: ET.Element) -> list[dict[str, Any]]:
    groups = []
    for g in axes.findall(f'.//{{{SVG_NS}}}g'):
        gid = g.get("id", "")
        if not gid.startswith("FillBetweenPolyCollection_"):
            continue
        path = g.find(f'.//{{{SVG_NS}}}path')
        if path is None:
            continue
        style = path.get("style", "")
        fill = _extract_fill_color(style) or path.get("fill") or ""
        points = _extract_points(path.get("d", ""))
        groups.append({"group": g, "path": path, "fill": fill.lower(), "points": points, "id": gid})

    def _id_key(item: dict[str, Any]) -> int:
        match = re.search(r"_(\d+)$", item["id"])
        return int(match.group(1)) if match else 0

    return sorted(groups, key=_id_key)


def _extract_points(d_attr: str) -> list[tuple[float, float]]:
    coords = re.findall(r"[ML]\s+(-?[\d.]+)\s+(-?[\d.]+)", d_attr)
    points = []
    for x_str, y_str in coords:
        try:
            points.append((float(x_str), float(y_str)))
        except ValueError:
            continue
    return points


def _find_area_by_fill(areas: list[dict[str, Any]], fill: str) -> int | None:
    fill_norm = fill.lower()
    for idx, area in enumerate(areas):
        if area.get("fill") == fill_norm:
            return idx
    return None


def _area_series_values(
    areas: list[dict[str, Any]], y_ticks: list[tuple[float, float]]
) -> tuple[list[float], list[list[float]]]:
    x_values = []
    series_values = []

    base_bounds = _top_bottom_by_x(areas[0]["points"])
    x_values = sorted(base_bounds.keys())
    for area in areas:
        bounds = _top_bottom_by_x(area["points"])
        if set(bounds.keys()) != set(x_values):
            raise ValueError("Area series do not share a common x grid.")
        values = []
        for x_val in x_values:
            top_y, bottom_y = bounds[x_val]
            top_data = _pixel_to_data(top_y, y_ticks)
            bottom_data = _pixel_to_data(bottom_y, y_ticks)
            values.append(top_data - bottom_data)
        series_values.append(values)

    return x_values, series_values


def _top_bottom_by_x(points: list[tuple[float, float]]) -> dict[float, tuple[float, float]]:
    bounds: dict[float, list[float]] = {}
    for x_val, y_val in points:
        bounds.setdefault(x_val, []).append(y_val)
    result: dict[float, tuple[float, float]] = {}
    for x_val, ys in bounds.items():
        result[x_val] = (min(ys), max(ys))
    return result


def _remove_legend_item(
    root: ET.Element, legend: ET.Element, items: list[dict[str, Any]], label: str
) -> None:
    for item in items:
        if item.get("label") != label:
            continue
        patch = item.get("patch")
        text = item.get("text")
        if text is not None:
            _move_nested_defs(root, text)
        if patch is not None:
            legend.remove(patch)
        if text is not None:
            legend.remove(text)
        break


def _move_nested_defs(root: ET.Element, group: ET.Element) -> None:
    defs_nodes = list(group.findall(f'.//{{{SVG_NS}}}defs'))
    if not defs_nodes:
        return
    root_defs = _ensure_root_defs(root)
    for defs_node in defs_nodes:
        for child in list(defs_node):
            root_defs.append(child)


def _ensure_root_defs(root: ET.Element) -> ET.Element:
    defs_node = root.find(f'./{{{SVG_NS}}}defs')
    if defs_node is not None:
        return defs_node
    return ET.SubElement(root, f"{{{SVG_NS}}}defs")
