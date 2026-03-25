from __future__ import annotations

import copy
import html
import json
import math
import os
import re
import xml.etree.ElementTree as ET
from typing import Any

from chart_agent.perception.svg_renderer import default_output_paths, render_svg_to_png

SVG_NS = "http://www.w3.org/2000/svg"
REFERENCE_LINE_FIGSIZE = (8, 5)
REFERENCE_LINE_DPI = 300

ET.register_namespace("", SVG_NS)


def _parse_svg_tree(svg_path: str) -> ET.ElementTree:
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    return ET.parse(svg_path, parser=parser)


def update_line_svg(
    svg_path: str,
    question: str,
    mapping_info: dict[str, Any],
    output_path: str | None = None,
    svg_output_path: str | None = None,
    llm: Any | None = None,
    operation_target: dict[str, Any] | None = None,
    data_change: dict[str, Any] | None = None,
) -> str:
    ops = _resolve_line_ops(question, operation_target=operation_target, data_change=data_change)
    if len(ops) > 1:
        return _run_line_ops_sequence(
            svg_path=svg_path,
            question=question,
            mapping_info=mapping_info,
            ops=ops,
            output_path=output_path,
            svg_output_path=svg_output_path,
            llm=llm,
            operation_target=operation_target,
            data_change=data_change,
        )
    op = ops[0] if ops else "add"
    if op == "delete":
        return _remove_line_series(
            svg_path,
            question,
            mapping_info,
            output_path=output_path,
            svg_output_path=svg_output_path,
            llm=llm,
            operation_target=operation_target,
            data_change=data_change,
        )
    if op == "change":
        try:
            return _update_line_point(
                svg_path,
                question,
                mapping_info,
                output_path=output_path,
                svg_output_path=svg_output_path,
                llm=llm,
                operation_target=operation_target,
                data_change=data_change,
            )
        except ValueError as exc:
            if "No valid line update request found in question." not in str(exc):
                raise
    return _add_line_series(
        svg_path,
        question,
        mapping_info,
        output_path=output_path,
        svg_output_path=svg_output_path,
        llm=llm,
        operation_target=operation_target,
        data_change=data_change,
    )


def _add_line_series(
    svg_path: str,
    question: str,
    mapping_info: dict[str, Any],
    output_path: str | None = None,
    svg_output_path: str | None = None,
    llm: Any | None = None,
    operation_target: dict[str, Any] | None = None,
    data_change: dict[str, Any] | None = None,
) -> str:

    x_ticks = mapping_info.get("x_ticks", [])
    y_ticks = mapping_info.get("y_ticks", [])
    if len(x_ticks) < 2 or len(y_ticks) < 2:
        raise ValueError("Insufficient axis ticks for line mapping.")

    values, llm_meta = _parse_values(question, llm)
    if llm_meta:
        mapping_info["llm_meta"] = llm_meta
    if not values:
        raise ValueError("No new series values found in question.")

    x_positions = _compute_x_positions(values, x_ticks)
    y_positions = [_data_to_pixel(val, y_ticks) for val in values]
    points_svg = list(zip(x_positions, y_positions))

    with open(svg_path, "r", encoding="utf-8") as handle:
        content = handle.read()

    tree = _parse_svg_tree(svg_path)
    root = tree.getroot()
    axes = root.find(f'.//{{{SVG_NS}}}g[@id="axes_1"]')
    if axes is None:
        raise ValueError("SVG axes group not found.")

    line_group, line_path = _ensure_update_line_path(axes)
    line_style = _extract_line_style(root)
    stroke = line_style.get("stroke", "#dc143c")
    stroke_width = line_style.get("stroke_width", 2.0)
    stroke_linecap = str(line_style.get("stroke_linecap", "") or "").strip()
    use_markers = bool(line_style.get("has_markers", False))
    style_parts = [
        "fill: none",
        f"stroke: {stroke}",
        f"stroke-width: {stroke_width}",
    ]
    if stroke_linecap:
        style_parts.append(f"stroke-linecap: {stroke_linecap}")
    line_path.set(
        "style",
        "; ".join(style_parts),
    )

    line_path.set("d", _format_path(points_svg))
    _draw_line_markers(axes, points_svg, stroke, enabled=use_markers)

    label = str((operation_target or {}).get("category_name") or "").strip()
    if not label and isinstance(data_change, dict):
        add_block = data_change.get("add") if isinstance(data_change.get("add"), dict) else data_change
        if isinstance(add_block, dict):
            label = str(add_block.get("category_name") or add_block.get("category") or "").strip()
    if not label:
        label = _extract_series_label(question)
    if not label and llm is not None:
        label = _parse_label_with_llm(question, llm)
    if label:
        legend, legend_items = _extract_legend_items(root, content)
        if legend is not None:
            _append_legend_item(legend, legend_items, label, stroke, show_marker=use_markers)

    svg_out, png_out = default_output_paths(svg_path, "line")
    target_svg = svg_output_path or svg_out
    target_png = output_path or png_out
    os.makedirs(os.path.dirname(target_svg), exist_ok=True)
    tree.write(target_svg, encoding="utf-8", xml_declaration=True)
    return render_svg_to_png(target_svg, target_png)


def _resolve_line_ops(
    question: str,
    *,
    operation_target: dict[str, Any] | None = None,
    data_change: dict[str, Any] | None = None,
) -> list[str]:
    structured_ops = _detect_structured_line_ops(operation_target, data_change)
    if structured_ops:
        return structured_ops
    clauses = [c.strip() for c in re.split(r"[；;\n]+", question) if c.strip()]
    ordered: list[str] = []
    for clause in clauses:
        op = _detect_line_op(clause)
        if op:
            ordered.append(op)
    if ordered:
        return ordered
    op = _detect_line_op(question)
    return [op] if op else ["add"]


def _detect_line_op(text: str) -> str | None:
    lowered = text.strip().lower()
    if lowered.startswith("operation:"):
        if "operation: delete" in lowered:
            return "delete"
        if "operation: change" in lowered:
            return "change"
        if "operation: add" in lowered:
            return "add"
    if _has_delete_intent(text):
        return "delete"
    if _has_year_update(text):
        return "change"
    if _has_add_intent(text):
        return "add"
    return None


def _detect_structured_line_ops(
    operation_target: dict[str, Any] | None,
    data_change: dict[str, Any] | None,
) -> list[str]:
    payload = data_change if isinstance(data_change, dict) else {}
    ops: list[str] = []
    if payload:
        for key in payload.keys():
            normalized = _normalize_line_op_token(str(key))
            if not normalized:
                continue
            if normalized == "delete" and _extract_structured_line_delete_labels(operation_target, payload):
                if normalized not in ops:
                    ops.append(normalized)
            elif normalized == "change" and _extract_structured_line_changes(payload):
                if normalized not in ops:
                    ops.append(normalized)
            elif normalized == "add":
                add_block = payload.get("add") if isinstance(payload.get("add"), dict) else payload
                if isinstance(add_block, dict) and any(key in add_block for key in ("values", "category_name", "category")):
                    if normalized not in ops:
                        ops.append(normalized)
        if not ops and _extract_structured_line_changes(payload):
            ops.append("change")
        if not ops and _extract_structured_line_delete_labels(operation_target, payload):
            ops.append("delete")
    target = operation_target if isinstance(operation_target, dict) else {}
    if not ops and any(key in target for key in ("del_category", "del_categories")):
        ops.append("delete")
    if not ops and any(key in target for key in ("year", "years")):
        ops.append("change")
    if not ops and any(key in target for key in ("add_category", "add_categories")):
        ops.append("add")
    return ops


def _normalize_line_op_token(token: str) -> str:
    lowered = str(token or "").strip().lower()
    if lowered in {"del", "delete", "remove", "drop", "del_categories"}:
        return "delete"
    if lowered in {"change", "changes", "update", "modify"}:
        return "change"
    if lowered in {"add", "append", "insert"}:
        return "add"
    return ""


def _has_add_intent(question: str) -> bool:
    return bool(re.search(r"(新增|添加|add|append|insert|new\s+series)", question, re.IGNORECASE))


def _refresh_mapping_info(svg_path: str, question: str, llm: Any | None) -> dict[str, Any]:
    from chart_agent.perception.svg_perceiver import perceive_svg

    perceived = perceive_svg(svg_path, question=question, llm=llm)
    mapping = perceived.get("mapping_info")
    return mapping if isinstance(mapping, dict) else {}


def _run_line_ops_sequence(
    *,
    svg_path: str,
    question: str,
    mapping_info: dict[str, Any],
    ops: list[str],
    output_path: str | None,
    svg_output_path: str | None,
    llm: Any | None,
    operation_target: dict[str, Any] | None,
    data_change: dict[str, Any] | None,
) -> str:
    clauses = [c.strip() for c in re.split(r"[；;\n]+", question) if c.strip()]
    if not clauses:
        clauses = [question]
    svg_out, png_out = default_output_paths(svg_path, "line")
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
            f".line_step_{idx+1}_{os.path.basename(final_svg)}",
        )
        step_png = final_png if is_last else os.path.join(
            os.path.dirname(final_png),
            f".line_step_{idx+1}_{os.path.basename(final_png)}",
        )
        if idx > 0:
            current_mapping = _refresh_mapping_info(current_svg, clause, llm)

        if op == "delete":
            _remove_line_series(
                current_svg,
                clause,
                current_mapping,
                output_path=step_png,
                svg_output_path=step_svg,
                llm=llm,
                operation_target=operation_target,
                data_change=data_change,
            )
        elif op == "change":
            _update_line_point(
                current_svg,
                clause,
                current_mapping,
                output_path=step_png,
                svg_output_path=step_svg,
                llm=llm,
                operation_target=operation_target,
                data_change=data_change,
            )
        else:
            _add_line_series(
                current_svg,
                clause,
                current_mapping,
                output_path=step_png,
                svg_output_path=step_svg,
                llm=llm,
                operation_target=operation_target,
                data_change=data_change,
            )
        current_svg = step_svg

    return final_png


def _remove_update_overlay(root: ET.Element) -> None:
    axes = root.find(f'.//{{{SVG_NS}}}g[@id="axes_1"]')
    if axes is not None:
        for gid in ("line2d_update", "line2d_update_markers"):
            node = axes.find(f'.//{{{SVG_NS}}}g[@id="{gid}"]')
            if node is not None:
                axes.remove(node)
    legend = root.find(f'.//{{{SVG_NS}}}g[@id="legend_1"]')
    if legend is not None:
        for gid in ("line2d_update_legend", "text_update_legend"):
            node = legend.find(f'./{{{SVG_NS}}}g[@id="{gid}"]')
            if node is not None:
                legend.remove(node)


def _find_line_path(axes: ET.Element) -> tuple[ET.Element | None, ET.Element | None]:
    for g in axes.findall(f'.//{{{SVG_NS}}}g'):
        gid = g.get("id", "")
        if not gid.startswith("line2d_"):
            continue
        path = g.find(f'.//{{{SVG_NS}}}path')
        if path is not None:
            return g, path
    update = axes.find(f'.//{{{SVG_NS}}}g[@id="line2d_update"]')
    if update is not None:
        path = update.find(f'.//{{{SVG_NS}}}path')
        if path is not None:
            return update, path
    return None, None


def _ensure_update_line_path(axes: ET.Element) -> tuple[ET.Element, ET.Element]:
    update = axes.find(f'.//{{{SVG_NS}}}g[@id="line2d_update"]')
    if update is None:
        update = ET.SubElement(axes, f"{{{SVG_NS}}}g", {"id": "line2d_update"})
    path = update.find(f'.//{{{SVG_NS}}}path')
    if path is None:
        path = ET.SubElement(update, f"{{{SVG_NS}}}path")
    return update, path


def _ensure_update_marker_group(axes: ET.Element) -> ET.Element:
    group = axes.find(f'.//{{{SVG_NS}}}g[@id="line2d_update_markers"]')
    if group is None:
        group = ET.SubElement(axes, f"{{{SVG_NS}}}g", {"id": "line2d_update_markers"})
    _clear_children(group)
    return group


def _format_path(points: list[tuple[float, float]]) -> str:
    if not points:
        return ""
    head = points[0]
    segments = [f"M {head[0]:.6f} {head[1]:.6f}"]
    for x_val, y_val in points[1:]:
        segments.append(f"L {x_val:.6f} {y_val:.6f}")
    return " ".join(segments)


def _extract_path_points(d_attr: str) -> list[tuple[float, float]]:
    coords = re.findall(r"[ML]\s+(-?[\d.]+)\s+(-?[\d.]+)", d_attr)
    points = []
    for x_str, y_str in coords:
        try:
            points.append((float(x_str), float(y_str)))
        except ValueError:
            continue
    return points


def _draw_line_markers(
    axes: ET.Element,
    points: list[tuple[float, float]],
    stroke: str,
    *,
    enabled: bool,
) -> None:
    group = _ensure_update_marker_group(axes)
    if not enabled:
        return
    radius = 3.0
    for x_val, y_val in points:
        circle = ET.SubElement(group, f"{{{SVG_NS}}}circle")
        circle.set("cx", f"{x_val:.6f}")
        circle.set("cy", f"{y_val:.6f}")
        circle.set("r", f"{radius:.6f}")
        circle.set(
            "style",
            f"fill: {stroke}; stroke: #000000; stroke-width: 0.5; fill-opacity: 0.85",
        )


def _clear_children(group: ET.Element) -> None:
    for child in list(group):
        group.remove(child)


def _update_marker_positions(group: ET.Element, target_x: float, new_y: float) -> None:
    for elem in group.iter():
        if elem is group:
            continue
        if "x" in elem.attrib and "y" in elem.attrib:
            try:
                x_val = float(elem.get("x", "nan"))
            except ValueError:
                continue
            if abs(x_val - target_x) < 0.5:
                elem.set("y", f"{new_y:.6f}")
        elif "cx" in elem.attrib and "cy" in elem.attrib:
            try:
                x_val = float(elem.get("cx", "nan"))
            except ValueError:
                continue
            if abs(x_val - target_x) < 0.5:
                elem.set("cy", f"{new_y:.6f}")


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


def _update_line_point(
    svg_path: str,
    question: str,
    mapping_info: dict[str, Any],
    output_path: str | None,
    svg_output_path: str | None,
    llm: Any | None,
    operation_target: dict[str, Any] | None = None,
    data_change: dict[str, Any] | None = None,
) -> str:
    x_ticks = mapping_info.get("x_ticks", [])
    y_ticks = mapping_info.get("y_ticks", [])
    if len(x_ticks) < 2 or len(y_ticks) < 2:
        raise ValueError("Insufficient axis ticks for line mapping.")

    with open(svg_path, "r", encoding="utf-8") as handle:
        content = handle.read()

    tree = _parse_svg_tree(svg_path)
    root = tree.getroot()
    axes = root.find(f'.//{{{SVG_NS}}}g[@id="axes_1"]')
    if axes is None:
        raise ValueError("SVG axes group not found.")

    legend, legend_items = _extract_legend_items(root, content)
    labels = [item["label"] for item in legend_items if item.get("label")]
    structured_changes = _extract_structured_line_changes(data_change)
    if not structured_changes:
        parsed = _parse_year_value_update(question, labels, llm)
        if parsed is None:
            raise ValueError("No valid line update request found in question.")
        structured_changes = [parsed]

    for label, year_value, update_value, mode in structured_changes:
        target_stroke = None
        for item in legend_items:
            if _labels_match(str(item.get("label") or ""), label):
                target_stroke = item.get("stroke")
                break
        if not target_stroke:
            raise ValueError("No matching legend color for selected series.")

        line_group = _find_line_by_stroke(axes, target_stroke)
        if line_group is None:
            raise ValueError("No line series matches selected legend color.")
        line_path = line_group.find(f'./{{{SVG_NS}}}path')
        if line_path is None:
            raise ValueError("Line path not found for selected series.")

        points = _extract_path_points(line_path.get("d", ""))
        if not points:
            raise ValueError("Line path contains no points.")

        target_x = _data_to_pixel(year_value, x_ticks)
        idx = min(range(len(points)), key=lambda i: abs(points[i][0] - target_x))
        current_data = _pixel_to_data(points[idx][1], y_ticks)
        new_data = current_data + update_value if mode == "relative" else update_value
        new_y = _data_to_pixel(new_data, y_ticks)
        points[idx] = (points[idx][0], new_y)
        line_path.set("d", _format_path(points))
        _update_marker_positions(line_group, points[idx][0], new_y)

    _remove_update_overlay(root)

    svg_out, png_out = default_output_paths(svg_path, "line")
    target_svg = svg_output_path or svg_out
    target_png = output_path or png_out
    os.makedirs(os.path.dirname(target_svg), exist_ok=True)
    tree.write(target_svg, encoding="utf-8", xml_declaration=True)
    return render_svg_to_png(target_svg, target_png)


def _parse_with_llm(
    question: str, llm: Any
) -> tuple[list[float], dict[str, Any]] | None:
    prompt = (
        "You are parsing a line chart update. "
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


def _remove_line_series(
    svg_path: str,
    question: str,
    mapping_info: dict[str, Any],
    output_path: str | None,
    svg_output_path: str | None,
    llm: Any | None = None,
    operation_target: dict[str, Any] | None = None,
    data_change: dict[str, Any] | None = None,
) -> str:
    with open(svg_path, "r", encoding="utf-8") as handle:
        content = handle.read()

    tree = _parse_svg_tree(svg_path)
    root = tree.getroot()
    axes = root.find(f'.//{{{SVG_NS}}}g[@id="axes_1"]')
    if axes is None:
        raise ValueError("SVG axes group not found.")

    legend, legend_items = _extract_legend_items(root, content)
    labels = [item["label"] for item in legend_items if item.get("label")]
    labels_to_remove = _extract_structured_line_delete_labels(operation_target, data_change)
    if not labels_to_remove:
        labels_to_remove = _resolve_delete_labels(question, labels, llm)
    if not labels_to_remove:
        raise ValueError("No matching line series found in question.")

    strokes_by_label = {item.get("label"): item.get("stroke") for item in legend_items}
    for label in labels_to_remove:
        resolved_label = _resolve_matching_label(label, [str(key or "") for key in strokes_by_label])
        target_stroke = strokes_by_label.get(resolved_label) if resolved_label else None
        if not target_stroke:
            raise ValueError("No matching legend color for selected series.")
        line_group = _find_line_by_stroke(axes, target_stroke)
        if line_group is None:
            raise ValueError("No line series matches selected legend color.")
        axes.remove(line_group)
        if legend is not None:
            _remove_legend_item(root, legend, legend_items, resolved_label or label)

    _rescale_line_chart_after_removal(root, axes, content, mapping_info)
    _remove_update_overlay(root)

    svg_out, png_out = default_output_paths(svg_path, "line")
    target_svg = svg_output_path or svg_out
    target_png = output_path or png_out
    os.makedirs(os.path.dirname(target_svg), exist_ok=True)
    tree.write(target_svg, encoding="utf-8", xml_declaration=True)
    return render_svg_to_png(target_svg, target_png)


def _has_delete_intent(question: str) -> bool:
    return bool(
        re.search(
            r"(delete|deleting|deletion|remove|removing|drop|删|删除|去掉|移除|去除|剔除)",
            question,
            re.IGNORECASE,
        )
    )


def _has_year_update(question: str) -> bool:
    if "[" in question:
        return False
    has_year = bool(re.search(r"\b(19|20)\d{2}\b", question) or re.search(r"\d+\s*年", question))
    if not has_year:
        return False
    has_update_verb = bool(
        re.search(
            r"(改为|变为|调整到|设为|设置为|increase|decrease|reduce|raise|update|change|set\s+to|to\s+\d)",
            question,
            re.IGNORECASE,
        )
    )
    return has_update_verb


def _match_label(question: str, labels: list[str]) -> str | None:
    matches = _match_labels(question, labels)
    return matches[0] if matches else None


def _normalize_label_token(value: str) -> str:
    text = str(value or "").strip().lower()
    return text[1:] if text.startswith("@") else text


def _labels_match(left: str, right: str) -> bool:
    left_text = str(left or "").strip()
    right_text = str(right or "").strip()
    if not left_text or not right_text:
        return False
    if left_text.lower() == right_text.lower():
        return True
    return _normalize_label_token(left_text) == _normalize_label_token(right_text)


def _resolve_matching_label(candidate: str, labels: list[str]) -> str | None:
    text = str(candidate or "").strip()
    if not text:
        return None
    for label in labels:
        if str(label or "").strip().lower() == text.lower():
            return label
    for label in labels:
        if _labels_match(text, label):
            return label
    return None


def _match_labels(question: str, labels: list[str]) -> list[str]:
    lowered = question.lower()
    matched = []
    for label in labels:
        label_text = str(label or "").strip()
        if not label_text:
            continue
        normalized = _normalize_label_token(label_text)
        if label_text.lower() in lowered or (normalized and normalized in lowered):
            matched.append(label)
    if matched:
        return matched
    quoted = re.findall(r"[\"“”']([^\"“”']+)[\"“”']", question)
    if not quoted:
        return []
    matched = []
    for quote in quoted:
        resolved = _resolve_matching_label(quote, labels)
        if resolved:
            matched.append(resolved)
    return matched


def _match_labels_with_llm(question: str, labels: list[str], llm: Any) -> list[str]:
    prompt = (
        "Extract labels to delete from the line chart question.\n"
        f"Candidate labels: {json.dumps(labels, ensure_ascii=False)}\n"
        "Return JSON only with key: labels (array of exact candidate label strings).\n"
        f"Question: {question}"
    )
    try:
        response = llm.invoke(prompt)
    except Exception:
        return []
    content = getattr(response, "content", "")
    payload = _safe_json_loads(content)
    if not payload:
        return []
    raw = payload.get("labels", [])
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        resolved = _resolve_matching_label(item, labels)
        if resolved and resolved not in out:
            out.append(resolved)
    return out


def _resolve_delete_labels(question: str, labels: list[str], llm: Any | None) -> list[str]:
    labels_to_remove = _match_labels_with_llm(question, labels, llm) if llm is not None else []
    if labels_to_remove:
        return labels_to_remove
    return _match_labels(question, labels)


def _extract_structured_line_delete_labels(
    operation_target: dict[str, Any] | None,
    data_change: dict[str, Any] | None,
) -> list[str]:
    labels: list[str] = []
    candidates: list[Any] = []
    if isinstance(operation_target, dict):
        candidates.extend(
            [
                operation_target.get("category_name"),
                operation_target.get("category_names"),
                operation_target.get("categories"),
                operation_target.get("del_category"),
                operation_target.get("del_categories"),
            ]
        )
    if isinstance(data_change, dict):
        del_block = data_change.get("del")
        if isinstance(del_block, dict):
            candidates.extend(
                [
                    del_block.get("category_name"),
                    del_block.get("category_names"),
                    del_block.get("category"),
                ]
            )
        candidates.extend(
            [
                data_change.get("del_categories"),
                data_change.get("category_names"),
                data_change.get("categories"),
            ]
        )
    for candidate in candidates:
        if isinstance(candidate, str):
            text = candidate.strip()
            if text and text not in labels:
                labels.append(text)
        elif isinstance(candidate, list):
            for item in candidate:
                text = str(item).strip()
                if text and text not in labels:
                    labels.append(text)
    return labels


def _extract_structured_line_changes(
    data_change: dict[str, Any] | None,
) -> list[tuple[str, float, float, str]]:
    payload = data_change if isinstance(data_change, dict) else {}
    changes_sources: list[list[Any]] = []
    if isinstance(payload.get("changes"), list):
        changes_sources.append(payload.get("changes"))
    root = payload.get("change") if isinstance(payload.get("change"), dict) else None
    if isinstance(root, dict) and isinstance(root.get("changes"), list):
        changes_sources.append(root.get("changes"))
    out: list[tuple[str, float, float, str]] = []
    for changes in changes_sources:
        for change in changes:
            if not isinstance(change, dict):
                continue
            label = str(change.get("category_name") or change.get("category") or "").strip()
            if not label:
                continue
            year_values = change.get("year_to_value")
            if isinstance(year_values, dict):
                for year, value in year_values.items():
                    try:
                        out.append((label, float(year), float(value), "absolute"))
                    except (TypeError, ValueError):
                        continue
                continue
            years = change.get("years")
            values = change.get("values")
            if not isinstance(years, list) or not isinstance(values, list):
                continue
            for year, value in zip(years, values):
                try:
                    out.append((label, float(year), float(value), "absolute"))
                except (TypeError, ValueError):
                    continue
    return out


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
        r"^(新增|添加|add|new)\s*(一个|一条)?\s*(类别|折线|series|line)?\s*",
        "",
        prefix,
        flags=re.IGNORECASE,
    ).strip(" -:")
    if cleaned.lower() in {"line", "series", "category", "折线", "类别"}:
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
    if label.lower() in {"line", "series", "category", "折线", "类别"}:
        return None
    return label


def _parse_update_with_llm(
    question: str, llm: Any
) -> tuple[str | None, float | None, float | None, str] | None:
    prompt = (
        "Extract a line update request with series label, year and value. "
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


def _extract_legend_items(
    root: ET.Element, content: str
) -> tuple[ET.Element | None, list[dict[str, Any]]]:
    legend = root.find(f'.//{{{SVG_NS}}}g[@id="legend_1"]')
    if legend is None:
        return None, []

    items: list[dict[str, Any]] = []
    pending_stroke = None
    pending_patch = None

    for child in list(legend):
        stroke = _extract_stroke_from_group(child)
        if stroke:
            pending_stroke = stroke
            pending_patch = child
            continue

        label = _extract_text_label(child, content)
        if label:
            items.append(
                {
                    "label": label,
                    "stroke": pending_stroke,
                    "text": child,
                    "patch": pending_patch,
                }
            )
            pending_stroke = None
            pending_patch = None

    return legend, items


def _append_legend_item(
    legend: ET.Element,
    items: list[dict[str, Any]],
    label: str,
    stroke: str,
    *,
    show_marker: bool,
) -> None:
    y_min, y_max, x_min, x_max = _legend_bounds(items)
    text_x, last_text_y = _legend_text_anchor(items, prefer_last=True)
    if last_text_y is None:
        text_x, last_text_y = _legend_text_anchor_from_legend(legend, prefer_last=True)
    if last_text_y is None:
        return
    step = _legend_step(items)
    next_text_y = last_text_y + step

    if y_min is None or y_max is None:
        x_min = text_x - 28.0
        x_max = text_x - 8.0
        y_line = next_text_y - 3.5
    else:
        y_line = y_max + step - 3.5

    line_group = ET.SubElement(legend, f"{{{SVG_NS}}}g", {"id": "line2d_update_legend"})
    path = ET.SubElement(line_group, f"{{{SVG_NS}}}path")
    path.set("d", f"M {x_min:.6f} {y_line:.6f} L {x_max:.6f} {y_line:.6f}")
    path.set("style", f"fill: none; stroke: {stroke}; stroke-width: 1.5")

    if show_marker:
        marker = ET.SubElement(line_group, f"{{{SVG_NS}}}circle")
        marker.set("cx", f"{(x_min + x_max) / 2.0:.6f}")
        marker.set("cy", f"{y_line:.6f}")
        marker.set("r", "3")
        marker.set("style", f"fill: {stroke}; stroke: {stroke}")

    text_group = ET.SubElement(legend, f"{{{SVG_NS}}}g", {"id": "text_update_legend"})
    text_group.append(ET.Comment(f" {label} "))
    text = ET.SubElement(text_group, f"{{{SVG_NS}}}text")
    text.set("x", f"{(x_max + 8.0):.6f}")
    text.set("y", f"{next_text_y:.6f}")
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
        if len(coords) < 4:
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


def _legend_step(items: list[dict[str, Any]]) -> float:
    ys = _legend_text_positions(items)
    if len(ys) >= 2:
        return ys[1] - ys[0]
    return 14.0


def _legend_text_anchor(items: list[dict[str, Any]], prefer_last: bool = False) -> tuple[float, float | None]:
    iterable = reversed(items) if prefer_last else iter(items)
    for item in iterable:
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


def _legend_text_anchor_from_legend(legend: ET.Element, prefer_last: bool = False) -> tuple[float, float | None]:
    groups = legend.findall(f'.//{{{SVG_NS}}}g')
    iterable = reversed(groups) if prefer_last else iter(groups)
    for g in iterable:
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
        return html.unescape(match.group(1)).strip()
    for node in group.iter():
        if node.tag is ET.Comment:
            comment_text = html.unescape(str(node.text or "")).strip()
            if comment_text:
                return comment_text
        text = html.unescape(str(node.text or "")).strip()
        if text:
            return text
    return None


def _extract_stroke_from_group(group: ET.Element) -> str | None:
    path = group.find(f'.//{{{SVG_NS}}}path')
    if path is None:
        return None
    style = path.get("style", "")
    match = re.search(r"stroke:\s*(#[0-9a-fA-F]{6})", style)
    if match:
        return match.group(1)
    return None


def _find_line_by_stroke(axes: ET.Element, stroke: str) -> ET.Element | None:
    stroke_norm = stroke.lower()
    for g in axes.findall(f'.//{{{SVG_NS}}}g'):
        gid = g.get("id", "")
        if not gid.startswith("line2d_"):
            continue
        path = g.find(f'.//{{{SVG_NS}}}path')
        if path is None:
            continue
        style = path.get("style", "")
        match = re.search(r"stroke:\s*(#[0-9a-fA-F]{6})", style)
        if match and match.group(1).lower() == stroke_norm:
            return g
    return None


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


def _extract_line_style(root: ET.Element) -> dict[str, float | str]:
    axes = root.find(f'.//{{{SVG_NS}}}g[@id="axes_1"]')
    if axes is None:
        return {}
    used_strokes: list[str] = []
    first_style: dict[str, float | str] | None = None
    for g in axes.findall(f'.//{{{SVG_NS}}}g'):
        gid = g.get("id", "")
        if not gid.startswith("line2d_"):
            continue
        path = g.find(f'.//{{{SVG_NS}}}path')
        if path is None:
            continue
        if not _is_plot_line_group(path):
            continue
        style = path.get("style", "")
        stroke = _extract_style_value(style, "stroke")
        stroke_width = _extract_style_value(style, "stroke-width")
        if stroke:
            stroke_norm = stroke.strip()
            if stroke_norm:
                used_strokes.append(stroke_norm.lower())
        if first_style is None:
            stroke_linecap = _extract_style_value(style, "stroke-linecap")
            first_style = {
                "stroke_width": float(stroke_width) if stroke_width else 2.0,
                "stroke_linecap": stroke_linecap.strip() if stroke_linecap else "",
                "has_markers": _line_group_has_markers(g),
            }
    if first_style is None:
        return {}
    return {
        "stroke": _pick_unused_line_stroke(used_strokes),
        "stroke_width": first_style["stroke_width"],
        "stroke_linecap": first_style["stroke_linecap"],
        "has_markers": first_style["has_markers"],
    }


def _is_plot_line_group(path: ET.Element) -> bool:
    return bool(path.get("clip-path"))


def _pick_unused_line_stroke(used_strokes: list[str]) -> str:
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#aec7e8",
        "#ffbb78",
        "#98df8a",
        "#ff9896",
        "#c5b0d5",
        "#c49c94",
        "#f7b6d2",
        "#c7c7c7",
        "#dbdb8d",
        "#9edae5",
    ]
    used = {str(x or "").strip().lower() for x in used_strokes if str(x or "").strip()}
    for color in palette:
        if color.lower() not in used:
            return color
    return "#9467bd"


def _line_group_has_markers(group: ET.Element) -> bool:
    for elem in group.iter():
        if elem is group:
            continue
        tag = elem.tag.split("}")[-1]
        if tag in {"circle", "use"}:
            return True
    return False


def _extract_style_value(style: str, key: str) -> str | None:
    match = re.search(rf"(?:^|;)\s*{re.escape(key)}:\s*([^;]+)", style)
    if match:
        return match.group(1).strip()
    return None


def _rescale_line_chart_after_removal(
    root: ET.Element,
    axes: ET.Element,
    content: str,
    mapping_info: dict[str, Any],
) -> None:
    # Re-parse from current SVG content so we can preserve the current SVG tick/pixel mapping
    # before asking Matplotlib to recompute the next visible y-tick layout.
    y_ticks = _parse_axis_ticks(root, content, axis_id="matplotlib.axis_2", is_x=False)
    if not y_ticks:
        y_ticks = mapping_info.get("y_ticks")
    if len(y_ticks) < 2:
        return

    series_groups = _find_line_series_groups(axes)
    if not series_groups:
        return

    values: list[float] = []
    for group in series_groups:
        path = group.find(f'./{{{SVG_NS}}}path')
        if path is None:
            continue
        points = _extract_path_points(path.get("d", ""))
        for _, y_val in points:
            data_val = _pixel_to_data(y_val, y_ticks)
            if math.isfinite(data_val):
                values.append(data_val)

    if not values:
        return

    data_min = min(values)
    data_max = max(values)
    if data_min == data_max:
        pad = max(1.0, abs(data_min) * 0.1)
        data_min -= pad
        data_max += pad

    tick_count_hint = max(2, len(y_ticks))
    view_min, view_max = _compute_draw_style_y_limits(data_min, data_max)
    axis_layout = _compute_matplotlib_y_axis_layout(view_min, view_max, tick_count_hint)
    new_tick_values = axis_layout["tick_values"]
    if len(new_tick_values) < 2:
        return
    new_tick_labels = axis_layout["tick_labels"]
    y_axis_scale = axis_layout["axis_scale"]
    _set_axis_scale_metadata(root, "matplotlib.axis_2", y_axis_scale)
    new_min = axis_layout["view_min"]
    new_max = axis_layout["view_max"]

    old_ticks_sorted = sorted(y_ticks, key=lambda t: t[1])
    old_min_pixel = old_ticks_sorted[0][0]
    old_max_pixel = old_ticks_sorted[-1][0]
    pixel_top, pixel_bottom = _extract_plot_y_bounds(root, mapping_info)

    new_ticks = [
        (_map_data_to_pixel(val, new_min, new_max, pixel_bottom, pixel_top), val)
        for val in new_tick_values
    ]

    _update_y_axis_ticks(root, new_ticks, axis_scale=y_axis_scale, tick_labels=new_tick_labels)
    _update_y_axis_scale_text(root, y_axis_scale)
    _rescale_series_groups(series_groups, y_ticks, new_min, new_max, pixel_bottom, pixel_top)


def _compute_matplotlib_y_axis_layout(
    view_min: float,
    view_max: float,
    tick_count_hint: int,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import ScalarFormatter

    fig = Figure(figsize=REFERENCE_LINE_FIGSIZE, dpi=REFERENCE_LINE_DPI)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_ylim(view_min, view_max)
    formatter = ScalarFormatter()
    ax.yaxis.set_major_formatter(formatter)
    fig.canvas.draw()

    vmin, vmax = sorted(ax.yaxis.get_view_interval())
    tolerance = max(1e-12, (vmax - vmin) * 1e-9)
    tick_values = [
        float(tick)
        for tick in ax.get_yticks()
        if vmin - tolerance <= float(tick) <= vmax + tolerance
    ]
    tick_labels = formatter.format_ticks(tick_values) if tick_values else []
    axis_scale = _axis_scale_from_formatter_offset(formatter.get_offset())
    fig.clear()
    return {
        "tick_values": tick_values,
        "tick_labels": tick_labels,
        "axis_scale": axis_scale,
        "offset_text": formatter.get_offset(),
        "view_min": float(vmin),
        "view_max": float(vmax),
    }


def _compute_draw_style_y_limits(data_min: float, data_max: float) -> tuple[float, float]:
    span = data_max - data_min
    margin = span * 0.05 if span else max(abs(data_max) * 0.05, 1.0)
    return float(data_min - margin), float(data_max + margin * 3)


def _extract_plot_y_bounds(root: ET.Element, mapping_info: dict[str, Any]) -> tuple[float, float]:
    axes_bounds = mapping_info.get("axes_bounds")
    if isinstance(axes_bounds, dict):
        top = axes_bounds.get("y_min")
        bottom = axes_bounds.get("y_max")
        if isinstance(top, (int, float)) and isinstance(bottom, (int, float)):
            return float(top), float(bottom)

    patch = root.find(f'.//{{{SVG_NS}}}g[@id="patch_2"]/{{{SVG_NS}}}path')
    if patch is not None:
        points = [float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:e[-+]?\d+)?', patch.get("d", ""))]
        ys = points[1::2]
        if ys:
            return min(ys), max(ys)

    return 37.249375, 314.449375


def _axis_scale_from_formatter_offset(offset_text: str) -> float:
    if not offset_text:
        return 1.0
    text = offset_text.strip()
    sci_patterns = [
        (r"^([+-]?(?:\d+(?:\.\d+)?|\.\d+)e[+-]?\d+)$", False),
        (r"^10\^([+-]?\d+)$", True),
        (r"^[×x]\s*10\^([+-]?\d+)$", True),
    ]
    for pattern, is_power_of_ten in sci_patterns:
        match = re.match(pattern, text, re.IGNORECASE)
        if not match:
            continue
        token = match.group(1)
        try:
            value = 10.0 ** float(token) if is_power_of_ten else float(token)
        except ValueError:
            continue
        if math.isfinite(value) and value != 0:
            return value
    return 1.0


def _find_line_series_groups(axes: ET.Element) -> list[ET.Element]:
    groups = []
    for g in axes.findall(f'./{{{SVG_NS}}}g'):
        gid = g.get("id", "")
        if not gid.startswith("line2d_"):
            continue
        path = g.find(f'./{{{SVG_NS}}}path')
        if path is None:
            continue
        groups.append(g)
    return groups


def _parse_axis_ticks(
    root: ET.Element, content: str, *, axis_id: str, is_x: bool
) -> list[tuple[float, float]]:
    axis = root.find(f'.//{{{SVG_NS}}}g[@id="{axis_id}"]')
    if axis is None:
        return []

    ticks: list[tuple[float, float]] = []
    # Scientific-notation scale marker is for y-axis in our SVG datasets.
    # Applying it to x-axis corrupts year/category mapping (e.g., 1973 -> 1.973e9).
    scale = 1.0 if is_x else _extract_axis_scale(content, axis_id)
    for g in axis.findall(f'.//{{{SVG_NS}}}g'):
        tick_id = g.get("id", "")
        if not tick_id.startswith("xtick_") and not tick_id.startswith("ytick_"):
            continue

        pixel_value = _extract_tick_position(g, is_x)
        if pixel_value is None:
            continue

        data_value = _extract_tick_label(content, g)
        if data_value is None:
            continue

        ticks.append((pixel_value, data_value * scale))

    ticks.sort(key=lambda t: t[1])
    return ticks


def _extract_axis_scale(content: str, axis_id: str) -> float:
    if not content or not axis_id:
        return 1.0
    num_pattern = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:e[+-]?\d+)?"
    attr_pattern = (
        rf'<g[^>]*id="{re.escape(axis_id)}"[^>]*\sdata-axis-scale="({num_pattern})"'
    )
    attr_match = re.search(attr_pattern, content)
    if attr_match:
        try:
            attr_value = float(attr_match.group(1))
        except ValueError:
            attr_value = 1.0
        if math.isfinite(attr_value) and attr_value != 0:
            return attr_value
    marker = f'id="{axis_id}"'
    idx = content.find(marker)
    if idx < 0:
        return 1.0
    # Matplotlib axis blocks are large and nested; scan a local window after axis marker.
    window = content[idx : idx + 60000]
    # Support common axis scientific notation labels: 1e6 / 10^6 / ×10^6 / x10^6.
    sci_patterns = [
        r"<!--\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+)e[+-]?\d+)\s*-->",
        r"<!--\s*10\^([+-]?\d+)\s*-->",
        r"<!--\s*[×x]\s*10\^([+-]?\d+)\s*-->",
    ]
    scale = None
    for pattern in sci_patterns:
        m = re.search(pattern, window, re.IGNORECASE)
        if not m:
            continue
        token = m.group(1)
        try:
            if "^" in pattern:
                scale = 10.0 ** float(token)
            else:
                scale = float(token)
        except ValueError:
            scale = None
        if scale is not None:
            break
    if scale is None:
        return 1.0
    return scale if math.isfinite(scale) and scale != 0 else 1.0


def _set_axis_scale_metadata(root: ET.Element, axis_id: str, axis_scale: float) -> None:
    axis = root.find(f'.//{{{SVG_NS}}}g[@id="{axis_id}"]')
    if axis is None:
        return
    if axis_scale and axis_scale != 1.0:
        axis.set("data-axis-scale", f"{axis_scale:.12g}")
    elif "data-axis-scale" in axis.attrib:
        axis.attrib.pop("data-axis-scale")


def _extract_tick_position(tick_group: ET.Element, is_x: bool) -> float | None:
    for use_elem in tick_group.findall(f'.//{{{SVG_NS}}}use'):
        attr = use_elem.get("x" if is_x else "y")
        if attr:
            try:
                return float(attr)
            except ValueError:
                continue

    line = tick_group.find(f'.//{{{SVG_NS}}}path[@d]')
    if line is None:
        return None
    d_attr = line.get("d", "")
    if is_x:
        match = re.search(r"M\s+([\d.]+)", d_attr)
    else:
        match = re.search(r"M\s+[\d.]+\s+([\d.]+)", d_attr)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _extract_tick_label(content: str, tick_group: ET.Element) -> float | None:
    text_id = None
    for g in tick_group.findall(f'.//{{{SVG_NS}}}g'):
        if g.get("id", "").startswith("text_"):
            text_id = g.get("id")
            break

    if text_id:
        pattern = rf'<g\s+id="{re.escape(text_id)}"[^>]*>\s*<!--\s*([\d.\-]+)\s*-->'
        match = re.search(pattern, content)
        if match:
            return _to_float(match.group(1))

    tick_id = tick_group.get("id", "")
    if tick_id:
        pattern = rf'<g\s+id="{re.escape(tick_id)}"[^>]*>.*?<!--\s*([\d.\-]+)\s*-->'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return _to_float(match.group(1))

    # Fallback for regenerated SVGs where comment labels are removed.
    for text in tick_group.findall(f'.//{{{SVG_NS}}}text'):
        raw = (text.text or "").strip()
        if not raw:
            continue
        val = _to_float(raw)
        if val is not None:
            return val

    return None


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _pick_y_min(y_ticks: list[tuple[float, float]], max_value: float) -> float:
    if any(abs(tick[1]) < 1e-9 for tick in y_ticks) and max_value >= 0:
        return 0.0
    min_val = min(tick[1] for tick in y_ticks)
    return min_val


def _nice_step(value: float) -> float:
    if value <= 0:
        return 1.0
    exponent = int(math.floor(math.log10(abs(value))))
    base = 10 ** exponent
    fraction = abs(value) / base
    candidates = [1.0, 2.0, 2.5, 5.0, 10.0]
    nearest = min(candidates, key=lambda c: abs(c - fraction))
    return nearest * base


def _nice_upper_bound(min_value: float, max_value: float, tick_count: int) -> float:
    if tick_count < 2:
        return max_value
    span = max_value - min_value
    if span <= 0:
        return max_value
    step = _nice_number(span / (tick_count - 1), round_up=True)
    return min_value + step * (tick_count - 1)


def _nice_number(value: float, round_up: bool) -> float:
    if value == 0:
        return 0.0
    exponent = int(math.floor(math.log10(abs(value))))
    fraction = abs(value) / (10 ** exponent)
    if round_up:
        if fraction <= 1:
            nice_fraction = 1
        elif fraction <= 2:
            nice_fraction = 2
        elif fraction <= 5:
            nice_fraction = 5
        else:
            nice_fraction = 10
    else:
        if fraction < 1.5:
            nice_fraction = 1
        elif fraction < 3:
            nice_fraction = 2
        elif fraction < 7:
            nice_fraction = 5
        else:
            nice_fraction = 10
    return nice_fraction * (10 ** exponent)


def _build_ticks(min_value: float, max_value: float, tick_count: int) -> list[float]:
    if tick_count < 2:
        return [min_value, max_value]
    step = (max_value - min_value) / (tick_count - 1)
    return [min_value + step * idx for idx in range(tick_count)]


def _build_ticks_by_step(min_value: float, max_value: float, step: float) -> list[float]:
    if step <= 0:
        return [min_value, max_value]
    ticks: list[float] = []
    value = min_value
    limit = 0
    while value <= max_value + step * 1e-6 and limit < 200:
        ticks.append(round(value, 10))
        value += step
        limit += 1
    if not ticks:
        return [min_value, max_value]
    if abs(ticks[-1] - max_value) > step * 0.25:
        ticks.append(round(max_value, 10))
    return ticks


def _map_data_to_pixel(
    value: float,
    data_min: float,
    data_max: float,
    pixel_min: float,
    pixel_max: float,
) -> float:
    if data_max == data_min:
        return pixel_min
    ratio = (value - data_min) / (data_max - data_min)
    return pixel_min + ratio * (pixel_max - pixel_min)


def _update_y_axis_ticks(
    root: ET.Element,
    new_ticks: list[tuple[float, float]],
    *,
    axis_scale: float = 1.0,
    tick_labels: list[str] | None = None,
) -> None:
    axis = root.find(f'.//{{{SVG_NS}}}g[@id="matplotlib.axis_2"]')
    if axis is None:
        return

    tick_groups = []
    for g in axis.findall(f'.//{{{SVG_NS}}}g'):
        if g.get("id", "").startswith("ytick_"):
            tick_groups.append(g)

    if not tick_groups:
        return

    tick_groups.sort(key=lambda g: _extract_tick_position(g, False) or 0, reverse=True)
    new_ticks_sorted = sorted(new_ticks, key=lambda t: t[1])
    desired = len(new_ticks_sorted)

    if desired > len(tick_groups):
        template = tick_groups[-1]
        while len(tick_groups) < desired:
            cloned = copy.deepcopy(template)
            tick_groups.append(cloned)
            axis.insert(len(tick_groups) - 1, cloned)
        _renumber_y_tick_groups(root, tick_groups)
    elif desired < len(tick_groups):
        for extra in tick_groups[desired:]:
            axis.remove(extra)
        tick_groups = tick_groups[:desired]
        _renumber_y_tick_groups(root, tick_groups)

    count = min(len(tick_groups), desired)

    for idx in range(count):
        tick_group = tick_groups[idx]
        new_y, new_value = new_ticks_sorted[idx]
        text_x, _, text_offset = _extract_tick_text_anchor(tick_group)
        label_text = tick_labels[idx] if tick_labels and idx < len(tick_labels) else None
        _update_tick_line_position(tick_group, new_y)
        _update_tick_label(
            tick_group,
            new_value,
            text_x,
            new_y if text_offset is None else new_y + text_offset,
            axis_scale=axis_scale,
            label_text=label_text,
        )


def _renumber_y_tick_groups(root: ET.Element, tick_groups: list[ET.Element]) -> None:
    next_line2d = _next_prefixed_numeric_id(root, "line2d_")
    next_text = _next_prefixed_numeric_id(root, "text_")
    for idx, tick_group in enumerate(tick_groups, start=1):
        tick_group.set("id", f"ytick_{idx}")
        for child in tick_group.findall(f'./{{{SVG_NS}}}g'):
            gid = child.get("id", "")
            if gid.startswith("line2d_"):
                child.set("id", f"line2d_{next_line2d}")
                next_line2d += 1
            elif gid.startswith("text_"):
                child.set("id", f"text_{next_text}")
                next_text += 1


def _next_prefixed_numeric_id(root: ET.Element, prefix: str) -> int:
    max_seen = 0
    for elem in root.iter():
        gid = elem.get("id", "")
        if not gid.startswith(prefix):
            continue
        suffix = gid[len(prefix) :]
        if suffix.isdigit():
            max_seen = max(max_seen, int(suffix))
    return max_seen + 1

def _extract_tick_text_anchor(
    tick_group: ET.Element,
) -> tuple[float | None, float | None, float | None]:
    text_group = None
    for g in tick_group.findall(f'./{{{SVG_NS}}}g'):
        if g.get("id", "").startswith("text_"):
            text_group = g
            break
    if text_group is None:
        return None, None, None

    translate = _extract_text_translate(text_group)
    if translate is None:
        return None, None, None
    text_x, text_y = translate
    old_y = _extract_tick_position(tick_group, False)
    if old_y is None:
        return text_x, text_y, None
    return text_x, text_y, text_y - old_y


def _extract_text_translate(group: ET.Element) -> tuple[float, float] | None:
    for g in group.findall(f'.//{{{SVG_NS}}}g'):
        transform = g.get("transform", "")
        match = re.search(r"translate\(([-\d.]+)\s+([-\d.]+)\)", transform)
        if match:
            return float(match.group(1)), float(match.group(2))
    return None


def _update_tick_line_position(tick_group: ET.Element, new_y: float) -> None:
    for use_elem in tick_group.findall(f'.//{{{SVG_NS}}}use'):
        if "y" in use_elem.attrib:
            use_elem.set("y", f"{new_y:.6f}")


def _format_tick_label(value: float, axis_scale: float = 1.0) -> str:
    display_value = value
    if axis_scale and axis_scale != 1.0:
        display_value = value / axis_scale
    value = display_value
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.2f}"


def _update_tick_label(
    tick_group: ET.Element,
    value: float,
    text_x: float | None,
    new_y: float,
    *,
    axis_scale: float = 1.0,
    label_text: str | None = None,
) -> None:
    text_group = None
    for g in tick_group.findall(f'./{{{SVG_NS}}}g'):
        if g.get("id", "").startswith("text_"):
            text_group = g
            break
    if text_group is None:
        text_group = ET.SubElement(tick_group, f"{{{SVG_NS}}}g")

    for child in list(text_group):
        text_group.remove(child)

    # Use one consistent x-anchor for all y tick labels.
    x_val = _fallback_tick_text_x(tick_group)
    text_elem = ET.SubElement(text_group, f"{{{SVG_NS}}}text")
    text_elem.set("x", f"{x_val:.6f}")
    text_elem.set("y", f"{new_y:.6f}")
    text_elem.set("font-size", "10")
    text_elem.set("font-family", "DejaVu Sans")
    text_elem.set("fill", "#000000")
    text_elem.set("text-anchor", "end")
    text_elem.text = label_text if label_text is not None else _format_tick_label(value, axis_scale=axis_scale)


def _fallback_tick_text_x(tick_group: ET.Element) -> float:
    for use_elem in tick_group.findall(f'.//{{{SVG_NS}}}use'):
        x_attr = use_elem.get("x")
        if x_attr is None:
            continue
        try:
            return float(x_attr) - 12.0
        except ValueError:
            continue
    return 0.0


def _update_y_axis_scale_text(root: ET.Element, axis_scale: float) -> None:
    axis = root.find(f'.//{{{SVG_NS}}}g[@id="matplotlib.axis_2"]')
    if axis is None:
        return

    scale_group = None
    for child in list(axis):
        gid = child.get("id", "")
        if not gid.startswith("text_"):
            continue
        text_value = _extract_group_comment_text(child)
        if text_value and re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)e[+-]?\d+", text_value):
            scale_group = child
            break

    if axis_scale == 1.0:
        if scale_group is not None:
            axis.remove(scale_group)
        return

    scale_text = f"{axis_scale:.0e}".replace("+0", "").replace("+", "")
    scale_text = scale_text.replace("e0", "e").replace("e-0", "e-")
    if scale_group is None:
        next_text = _next_prefixed_numeric_id(axis, "text_")
        scale_group = ET.SubElement(axis, f"{{{SVG_NS}}}g", {"id": f"text_{next_text}"})
    else:
        for child in list(scale_group):
            scale_group.remove(child)

    scale_group.append(ET.Comment(f" {scale_text} "))
    text_elem = ET.SubElement(scale_group, f"{{{SVG_NS}}}text")
    text_elem.set("x", "41.880625")
    text_elem.set("y", "34.249375")
    text_elem.set("font-size", "10")
    text_elem.set("font-family", "Times New Roman")
    text_elem.set("fill", "#000000")
    text_elem.text = scale_text


def _extract_group_comment_text(group: ET.Element) -> str | None:
    for child in list(group):
        if child.tag is ET.Comment:
            return (child.text or "").strip()
    return None


def _rescale_series_groups(
    groups: list[ET.Element],
    old_ticks: list[tuple[float, float]],
    new_min: float,
    new_max: float,
    pixel_min: float,
    pixel_max: float,
) -> None:
    for group in groups:
        path = group.find(f'./{{{SVG_NS}}}path')
        if path is None:
            continue
        points = _extract_path_points(path.get("d", ""))
        if not points:
            continue
        new_points = []
        for x_val, y_val in points:
            data_val = _pixel_to_data(y_val, old_ticks)
            new_y = _map_data_to_pixel(data_val, new_min, new_max, pixel_min, pixel_max)
            new_points.append((x_val, new_y))
        path.set("d", _format_path(new_points))

        for elem in group.iter():
            if "y" in elem.attrib:
                try:
                    y_val = float(elem.get("y", "nan"))
                except ValueError:
                    continue
                data_val = _pixel_to_data(y_val, old_ticks)
                new_y = _map_data_to_pixel(data_val, new_min, new_max, pixel_min, pixel_max)
                elem.set("y", f"{new_y:.6f}")
            if "cy" in elem.attrib:
                try:
                    y_val = float(elem.get("cy", "nan"))
                except ValueError:
                    continue
                data_val = _pixel_to_data(y_val, old_ticks)
                new_y = _map_data_to_pixel(data_val, new_min, new_max, pixel_min, pixel_max)
                elem.set("cy", f"{new_y:.6f}")
