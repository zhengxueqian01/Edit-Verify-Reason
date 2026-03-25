from __future__ import annotations

from collections import Counter
import html
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

SVG_NS = "http://www.w3.org/2000/svg"


NUMERIC_ATTRS = {
    "x",
    "y",
    "x1",
    "y1",
    "x2",
    "y2",
    "cx",
    "cy",
    "r",
    "rx",
    "ry",
    "width",
    "height",
    "stroke-width",
    "font-size",
    "opacity",
    "fill-opacity",
    "stroke-opacity",
}


def compare_svgs(predicted_svg: str | Path, ground_truth_svg: str | Path) -> dict[str, Any]:
    pred_path = Path(predicted_svg).expanduser()
    gt_path = Path(ground_truth_svg).expanduser()
    pred_features = _extract_svg_features(pred_path)
    gt_features = _extract_svg_features(gt_path)

    if pred_features.get("error") or gt_features.get("error"):
        return {
            "ok": False,
            "predicted_svg_path": str(pred_path),
            "ground_truth_svg_path": str(gt_path),
            "error": pred_features.get("error") or gt_features.get("error"),
        }

    pred_chart_type = _detect_chart_type(pred_features)
    gt_chart_type = _detect_chart_type(gt_features)
    chart_type = pred_chart_type if pred_chart_type == gt_chart_type else "mixed"

    if pred_chart_type == "area" and gt_chart_type == "area":
        return _compare_area_svgs(pred_path, gt_path, pred_features, gt_features)
    if pred_chart_type == "line" and gt_chart_type == "line":
        return _compare_line_svgs(pred_path, gt_path, pred_features, gt_features)
    if pred_chart_type == "scatter" and gt_chart_type == "scatter":
        return _compare_scatter_svgs(pred_path, gt_path, pred_features, gt_features)

    metrics = {
        "tag_f1": _counter_f1(pred_features["tags"], gt_features["tags"]),
        "attr_f1": _counter_f1(pred_features["attrs"], gt_features["attrs"]),
        "text_f1": _counter_f1(pred_features["texts"], gt_features["texts"]),
        "geom_similarity": _numeric_map_similarity(pred_features["numeric"], gt_features["numeric"]),
        "path_similarity": _counter_f1(pred_features["paths"], gt_features["paths"]),
    }
    score = (
        metrics["tag_f1"] * 0.20
        + metrics["attr_f1"] * 0.20
        + metrics["text_f1"] * 0.15
        + metrics["geom_similarity"] * 0.35
        + metrics["path_similarity"] * 0.10
    )
    return {
        "ok": True,
        "chart_type": chart_type,
        "predicted_svg_path": str(pred_path),
        "ground_truth_svg_path": str(gt_path),
        "score": round(score, 4),
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
        "predicted_summary": _feature_summary(pred_features),
        "ground_truth_summary": _feature_summary(gt_features),
    }


def resolve_ground_truth_svg(case_dir: str | Path, case_id: str, payload: dict[str, Any]) -> Path | None:
    base = Path(case_dir)
    data_change = (payload or {}).get("data_change")
    has_delete = isinstance(data_change, dict) and isinstance(data_change.get("del"), dict)
    candidates: list[Path] = []
    if has_delete:
        candidates.append(base / f"{case_id}_del.svg")
    if isinstance(data_change, dict) and data_change:
        candidates.append(base / f"{case_id}_aug.svg")
    candidates.extend(
        [
            base / f"{case_id}_gt.svg",
            base / f"{case_id}_updated.svg",
            base / f"{case_id}_target.svg",
        ]
    )
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    siblings = sorted(
        p for p in base.glob(f"{case_id}_*.svg")
        if p.is_file() and p.name not in {f"{case_id}.svg", f"{case_id}_tool_aug.svg"}
    )
    return siblings[0] if siblings else None


def _extract_svg_features(svg_path: Path) -> dict[str, Any]:
    try:
        parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
        root = ET.parse(svg_path, parser=parser).getroot()
    except Exception as exc:
        return {"error": f"parse_failed: {exc}"}

    tags: Counter[str] = Counter()
    attrs: Counter[str] = Counter()
    texts: Counter[str] = Counter()
    numeric: dict[str, list[float]] = {}
    paths: Counter[str] = Counter()
    for elem in root.iter():
        if elem.tag is ET.Comment:
            continue
        tag = _local_name(elem.tag)
        tags[tag] += 1
        if tag == "path":
            normalized_d = _normalize_path_data(elem.get("d"))
            if normalized_d:
                paths[normalized_d] += 1
        for key, value in elem.attrib.items():
            local_key = _local_name(key)
            normalized_value = _normalize_attr_value(local_key, value)
            attrs[f"{tag}:{local_key}={normalized_value}"] += 1
            if local_key in NUMERIC_ATTRS:
                numeric.setdefault(f"{tag}:{local_key}", []).extend(_extract_numbers(str(value)))
        text = _normalize_text(elem.text)
        if text:
            texts[text] += 1

    return {
        "tags": tags,
        "attrs": attrs,
        "texts": texts,
        "numeric": numeric,
        "paths": paths,
        "area_top_boundary": _extract_area_top_boundary(_extract_area_paths(root)),
        "area_gap_ratio": _extract_area_gap_ratio(_extract_area_paths(root)),
        "line_series": _extract_line_series(root),
        "line_styles": _extract_line_styles(root),
        "legend_labels": _extract_legend_labels(root),
        "legend_label_counts": _extract_legend_label_counts(root),
        "y_axis_ticks": _extract_y_axis_ticks(root),
        "scatter_points": _extract_scatter_points(root),
    }


def _feature_summary(features: dict[str, Any]) -> dict[str, Any]:
    return {
        "elements": int(sum(features["tags"].values())),
        "unique_tags": len(features["tags"]),
        "text_nodes": int(sum(features["texts"].values())),
        "numeric_attr_keys": len(features["numeric"]),
        "path_signatures": int(sum(features["paths"].values())),
        "area_boundary_points": len(features.get("area_top_boundary", [])),
        "line_series": len(features.get("line_series", [])),
        "scatter_points": len(features.get("scatter_points", [])),
    }


def _detect_chart_type(features: dict[str, Any]) -> str:
    if features.get("area_top_boundary"):
        return "area"
    if features.get("line_series"):
        return "line"
    if features.get("scatter_points"):
        return "scatter"
    return "generic"


def _compare_area_svgs(
    pred_path: Path,
    gt_path: Path,
    pred_features: dict[str, Any],
    gt_features: dict[str, Any],
) -> dict[str, Any]:
    pred_boundary = pred_features.get("area_top_boundary", [])
    gt_boundary = gt_features.get("area_top_boundary", [])
    boundary_score = _polyline_similarity(pred_boundary, gt_boundary)
    pred_gap_ratio = _safe_float(pred_features.get("area_gap_ratio"), 0.0)
    gt_gap_ratio = _safe_float(gt_features.get("area_gap_ratio"), 0.0)
    gap_score = max(0.0, 1.0 - min(abs(pred_gap_ratio - gt_gap_ratio), 1.0))
    pred_labels = Counter(pred_features.get("legend_labels", []))
    gt_labels = Counter(gt_features.get("legend_labels", []))
    label_score = _counter_f1(pred_labels, gt_labels)
    duplicate_score = _counter_f1(
        pred_features.get("legend_label_counts", Counter()),
        gt_features.get("legend_label_counts", Counter()),
    )
    score = (
        boundary_score * 0.55
        + gap_score * 0.15
        + label_score * 0.15
        + duplicate_score * 0.15
    )
    return {
        "ok": True,
        "chart_type": "area",
        "predicted_svg_path": str(pred_path),
        "ground_truth_svg_path": str(gt_path),
        "score": round(score, 4),
        "metrics": {
            "top_boundary_similarity": round(boundary_score, 4),
            "gap_score": round(gap_score, 4),
            "predicted_gap_ratio": round(pred_gap_ratio, 4),
            "ground_truth_gap_ratio": round(gt_gap_ratio, 4),
            "label_score": round(label_score, 4),
            "legend_count_score": round(duplicate_score, 4),
        },
        "predicted_summary": _feature_summary(pred_features),
        "ground_truth_summary": _feature_summary(gt_features),
    }


def _compare_line_svgs(
    pred_path: Path,
    gt_path: Path,
    pred_features: dict[str, Any],
    gt_features: dict[str, Any],
) -> dict[str, Any]:
    pred_lines = pred_features.get("line_series", [])
    gt_lines = gt_features.get("line_series", [])
    pred_labels = Counter(pred_features.get("legend_labels", []))
    gt_labels = Counter(gt_features.get("legend_labels", []))
    category_score = _counter_f1(pred_labels, gt_labels)
    y_axis_score = _numeric_list_similarity(
        pred_features.get("y_axis_ticks", []),
        gt_features.get("y_axis_ticks", []),
    )
    line_scores = _match_polylines_relaxed(pred_lines, gt_lines)
    geometry_score = sum(line_scores) / len(line_scores) if line_scores else 0.0
    style_scores = _match_line_styles(
        pred_lines,
        pred_features.get("line_styles", []),
        gt_lines,
        gt_features.get("line_styles", []),
    )
    style_score = sum(style_scores) / len(style_scores) if style_scores else 0.0
    score = category_score * 0.35 + y_axis_score * 0.30 + geometry_score * 0.20 + style_score * 0.15
    return {
        "ok": True,
        "chart_type": "line",
        "predicted_svg_path": str(pred_path),
        "ground_truth_svg_path": str(gt_path),
        "score": round(score, 4),
        "metrics": {
            "category_score": round(category_score, 4),
            "y_axis_score": round(y_axis_score, 4),
            "predicted_line_count": len(pred_lines),
            "ground_truth_line_count": len(gt_lines),
            "geometry_score": round(geometry_score, 4),
            "average_line_similarity": round(geometry_score, 4),
            "style_score": round(style_score, 4),
        },
        "predicted_summary": _feature_summary(pred_features),
        "ground_truth_summary": _feature_summary(gt_features),
    }


def _compare_scatter_svgs(
    pred_path: Path,
    gt_path: Path,
    pred_features: dict[str, Any],
    gt_features: dict[str, Any],
) -> dict[str, Any]:
    pred_points = pred_features.get("scatter_points", [])
    gt_points = gt_features.get("scatter_points", [])
    matches = _match_points(pred_points, gt_points, tolerance=1.5)
    matched = len(matches)
    total = max(len(pred_points), len(gt_points), 1)
    point_score = matched / total
    color_matched = sum(1 for pred, gt in matches if _normalize_color(pred.get("fill")) == _normalize_color(gt.get("fill")))
    color_score = color_matched / matched if matched else 0.0
    score = point_score * 0.70 + color_score * 0.30
    return {
        "ok": True,
        "chart_type": "scatter",
        "predicted_svg_path": str(pred_path),
        "ground_truth_svg_path": str(gt_path),
        "score": round(score, 4),
        "metrics": {
            "matched_points": matched,
            "predicted_point_count": len(pred_points),
            "ground_truth_point_count": len(gt_points),
            "point_match_ratio": round(point_score, 4),
            "color_matched_points": color_matched,
            "color_match_ratio": round(color_score, 4),
        },
        "predicted_summary": _feature_summary(pred_features),
        "ground_truth_summary": _feature_summary(gt_features),
    }


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _normalize_text(value: str | None) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text[:120]


def _normalize_attr_value(key: str, value: str) -> str:
    raw = str(value or "").strip()
    nums = _extract_numbers(raw)
    if nums and key in NUMERIC_ATTRS and len(nums) == 1 and raw.replace(str(nums[0]), "").strip(" %pxem,") == "":
        return f"{nums[0]:.3f}"
    return raw[:120]


def _normalize_path_data(value: str | None) -> str:
    if not value:
        return ""
    tokens = re.findall(r"[A-Za-z]|-?\d+(?:\.\d+)?", value)
    normalized: list[str] = []
    for token in tokens[:80]:
        if re.fullmatch(r"[A-Za-z]", token):
            normalized.append(token.upper())
            continue
        try:
            normalized.append(f"{float(token):.1f}")
        except ValueError:
            continue
    return " ".join(normalized)


def _extract_numbers(text: str) -> list[float]:
    out: list[float] = []
    for item in re.findall(r"-?\d+(?:\.\d+)?", text or ""):
        try:
            out.append(float(item))
        except ValueError:
            continue
    return out


def _counter_f1(left: Counter[str], right: Counter[str]) -> float:
    if not left and not right:
        return 1.0
    overlap = sum(min(left[key], right[key]) for key in set(left) | set(right))
    left_total = sum(left.values())
    right_total = sum(right.values())
    if left_total == 0 or right_total == 0:
        return 0.0
    precision = overlap / left_total
    recall = overlap / right_total
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _numeric_map_similarity(left: dict[str, list[float]], right: dict[str, list[float]]) -> float:
    keys = sorted(set(left) | set(right))
    if not keys:
        return 1.0
    scores: list[float] = []
    for key in keys:
        scores.append(_numeric_list_similarity(left.get(key, []), right.get(key, [])))
    return sum(scores) / len(scores)


def _numeric_list_similarity(left: list[float], right: list[float]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    a = sorted(left)
    b = sorted(right)
    overlap = min(len(a), len(b))
    if overlap == 0:
        return 0.0
    diffs: list[float] = []
    for idx in range(overlap):
        x = a[idx]
        y = b[idx]
        scale = max(abs(x), abs(y), 1.0)
        diffs.append(min(abs(x - y) / scale, 1.0))
    length_penalty = abs(len(a) - len(b)) / max(len(a), len(b), 1)
    distance = (sum(diffs) / len(diffs)) if diffs else 1.0
    similarity = 1.0 - min(1.0, distance * 0.8 + length_penalty * 0.2)
    return max(0.0, similarity)


def _extract_path_points(path_d: str | None) -> list[tuple[float, float]]:
    if not path_d:
        return []
    points: list[tuple[float, float]] = []
    for x_raw, y_raw in re.findall(r"[ML]\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)", path_d):
        try:
            points.append((float(x_raw), float(y_raw)))
        except ValueError:
            continue
    return points


def _extract_area_paths(root: ET.Element) -> list[list[tuple[float, float]]]:
    axes = root.find(f'.//{{{SVG_NS}}}g[@id="axes_1"]')
    if axes is None:
        return []
    paths: list[list[tuple[float, float]]] = []
    for group in axes.findall(f'.//{{{SVG_NS}}}g'):
        group_id = str(group.get("id", ""))
        if not group_id.startswith("FillBetweenPolyCollection_"):
            continue
        path = group.find(f'.//{{{SVG_NS}}}path')
        if path is None:
            continue
        coords = _extract_path_points(path.get("d"))
        if coords:
            paths.append(coords)
    return paths


def _extract_area_top_boundary(area_paths: list[list[tuple[float, float]]]) -> list[tuple[float, float]]:
    top_map: dict[float, float] = {}
    for points in area_paths:
        for x_val, y_val in _extract_area_boundary(points):
            if x_val not in top_map or y_val < top_map[x_val]:
                top_map[x_val] = y_val
    return sorted(top_map.items(), key=lambda item: item[0])


def _extract_area_gap_ratio(area_paths: list[list[tuple[float, float]]]) -> float:
    intervals_by_x: dict[float, list[tuple[float, float]]] = {}
    for points in area_paths:
        bounds = _area_interval_by_x(points)
        for x_val, interval in bounds.items():
            intervals_by_x.setdefault(x_val, []).append(interval)

    if not intervals_by_x:
        return 0.0

    gap_total = 0.0
    span_total = 0.0
    for x_val, intervals in intervals_by_x.items():
        ordered = sorted(intervals, key=lambda item: (item[0], item[1]))
        if not ordered:
            continue
        min_top = min(item[0] for item in ordered)
        max_bottom = max(item[1] for item in ordered)
        span = max_bottom - min_top
        if span <= 0:
            continue
        span_total += span
        merged: list[list[float]] = []
        for top, bottom in ordered:
            if not merged:
                merged.append([top, bottom])
                continue
            prev = merged[-1]
            if top <= prev[1]:
                prev[1] = max(prev[1], bottom)
            else:
                gap_total += top - prev[1]
                merged.append([top, bottom])
    if span_total <= 0:
        return 0.0
    return gap_total / span_total


def _extract_area_boundary(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(points) < 4:
        return []
    turn_idx = None
    for idx in range(1, len(points)):
        if points[idx][0] < points[idx - 1][0]:
            turn_idx = idx
            break
    if turn_idx is None:
        return points

    top_map: dict[float, float] = {}
    for x_val, y_val in points[:turn_idx] + points[turn_idx:]:
        if x_val not in top_map or y_val < top_map[x_val]:
            top_map[x_val] = y_val
    return sorted(top_map.items(), key=lambda item: item[0])


def _area_interval_by_x(points: list[tuple[float, float]]) -> dict[float, tuple[float, float]]:
    bounds: dict[float, list[float]] = {}
    for x_val, y_val in points:
        bounds.setdefault(x_val, []).append(y_val)
    out: dict[float, tuple[float, float]] = {}
    for x_val, ys in bounds.items():
        out[x_val] = (min(ys), max(ys))
    return out


def _extract_line_series(root: ET.Element) -> list[list[tuple[float, float]]]:
    axes = root.find(f'.//{{{SVG_NS}}}g[@id="axes_1"]')
    if axes is None:
        return []
    lines: list[list[tuple[float, float]]] = []
    for group in axes.findall(f'./{{{SVG_NS}}}g'):
        group_id = str(group.get("id", ""))
        if not group_id.startswith("line2d_"):
            continue
        path = group.find(f'./{{{SVG_NS}}}path')
        if path is None:
            continue
        style = str(path.get("style", ""))
        if "stroke:" not in style or "fill: none" not in style:
            continue
        points = _extract_path_points(path.get("d"))
        if points:
            lines.append(points)
    return lines


def _extract_line_styles(root: ET.Element) -> list[dict[str, Any]]:
    axes = root.find(f'.//{{{SVG_NS}}}g[@id="axes_1"]')
    if axes is None:
        return []
    styles: list[dict[str, Any]] = []
    for group in axes.findall(f'./{{{SVG_NS}}}g'):
        group_id = str(group.get("id", ""))
        if not group_id.startswith("line2d_"):
            continue
        path = group.find(f'./{{{SVG_NS}}}path')
        if path is None:
            continue
        style = str(path.get("style", ""))
        if "stroke:" not in style or "fill: none" not in style:
            continue
        points = _extract_path_points(path.get("d"))
        if not points:
            continue
        style_map = _parse_style_map(style)
        styles.append(
            {
                "stroke_width": _safe_float(style_map.get("stroke-width"), 1.0),
                "stroke_dasharray": _normalize_dasharray(style_map.get("stroke-dasharray")),
                "stroke_linecap": str(style_map.get("stroke-linecap") or "").strip().lower(),
                "stroke_linejoin": str(style_map.get("stroke-linejoin") or "").strip().lower(),
                "has_markers": _line_group_has_markers(group),
            }
        )
    return styles


def _extract_legend_labels(root: ET.Element) -> list[str]:
    counts = _extract_legend_label_counts(root)
    return list(counts.keys())


def _extract_legend_label_counts(root: ET.Element) -> Counter[str]:
    legend = root.find(f'.//{{{SVG_NS}}}g[@id="legend_1"]')
    if legend is None:
        return Counter()
    labels: Counter[str] = Counter()
    for group in legend.findall(f'./{{{SVG_NS}}}g'):
        gid = str(group.get("id", ""))
        if not gid.startswith("text_"):
            for text_node in group.findall(f'./{{{SVG_NS}}}text'):
                text = _normalize_text("".join(text_node.itertext()))
                if text:
                    labels[text] += 1
            continue
        extracted = False
        for child in list(group):
            if child.tag is ET.Comment:
                text = _normalize_text(child.text)
                if text:
                    labels[text] += 1
                extracted = True
                break
        if extracted:
            continue
        for text_node in group.findall(f'./{{{SVG_NS}}}text'):
            text = _normalize_text("".join(text_node.itertext()))
            if text:
                labels[text] += 1
    for text_node in legend.findall(f'./{{{SVG_NS}}}text'):
        text = _normalize_text("".join(text_node.itertext()))
        if text:
            labels[text] += 1
    return labels


def _extract_y_axis_ticks(root: ET.Element) -> list[float]:
    axis = root.find(f'.//{{{SVG_NS}}}g[@id="matplotlib.axis_2"]')
    if axis is None:
        return []
    scale = _safe_float(axis.get("data-axis-scale"), 1.0)
    for group in axis.findall(f'./{{{SVG_NS}}}g'):
        gid = str(group.get("id", ""))
        if not gid.startswith("text_"):
            continue
        for child in list(group):
            if child.tag is ET.Comment:
                text = (child.text or "").strip()
                if re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)e[+-]?\d+", text):
                    try:
                        scale = float(text)
                    except ValueError:
                        scale = 1.0
                break
        else:
            for text_node in group.findall(f'./{{{SVG_NS}}}text'):
                text = _normalize_text("".join(text_node.itertext()))
                if re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)e[+-]?\d+", text):
                    scale = _safe_float(text, scale)
                    break
    ticks: list[float] = []
    for tick_group in axis.findall(f'./{{{SVG_NS}}}g'):
        gid = str(tick_group.get("id", ""))
        if not gid.startswith("ytick_"):
            continue
        for group in tick_group.findall(f'./{{{SVG_NS}}}g'):
            child_id = str(group.get("id", ""))
            if not child_id.startswith("text_"):
                continue
            for child in list(group):
                if child.tag is ET.Comment:
                    raw = (child.text or "").strip()
                    nums = _extract_numbers(raw)
                    if nums:
                        ticks.append(nums[0] * scale)
                    break
            else:
                for text_node in group.findall(f'./{{{SVG_NS}}}text'):
                    raw = _normalize_text("".join(text_node.itertext()))
                    nums = _extract_numbers(raw)
                    if nums:
                        ticks.append(nums[0] * scale)
                        break
            break
    return ticks


def _extract_scatter_points(root: ET.Element) -> list[dict[str, Any]]:
    axes = root.find(f'.//{{{SVG_NS}}}g[@id="axes_1"]')
    if axes is None:
        return []

    points: list[dict[str, Any]] = []
    for group in axes.findall(f'.//{{{SVG_NS}}}g'):
        group_id = str(group.get("id", ""))
        if not (
            group_id == "PathCollection_update"
            or group_id.startswith("PathCollection_")
        ):
            continue
        for use_elem in group.findall(f'.//{{{SVG_NS}}}use'):
            x_attr = use_elem.get("x")
            y_attr = use_elem.get("y")
            if not x_attr or not y_attr:
                continue
            try:
                points.append(
                    {
                        "x": float(x_attr),
                        "y": float(y_attr),
                        "fill": _extract_element_fill(use_elem),
                    }
                )
            except ValueError:
                continue
        for circle in group.findall(f'.//{{{SVG_NS}}}circle'):
            cx_attr = circle.get("cx")
            cy_attr = circle.get("cy")
            if not cx_attr or not cy_attr:
                continue
            try:
                points.append(
                    {
                        "x": float(cx_attr),
                        "y": float(cy_attr),
                        "fill": _extract_element_fill(circle),
                    }
                )
            except ValueError:
                continue
    return points


def _extract_element_fill(elem: ET.Element) -> str:
    fill = str(elem.get("fill") or "").strip()
    if fill and fill.lower() != "none":
        return fill.lower()
    style_map = _parse_style_map(str(elem.get("style") or ""))
    style_fill = str(style_map.get("fill") or "").strip()
    if style_fill and style_fill.lower() != "none":
        return style_fill.lower()
    return ""


def _normalize_color(value: Any) -> str:
    return str(value or "").strip().lower()


def _parse_style_map(style: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in str(style or "").split(";"):
        if ":" not in item:
            continue
        key, value = item.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if key and value:
            out[key] = value
    return out


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_dasharray(value: Any) -> tuple[float, ...]:
    numbers = _extract_numbers(str(value or ""))
    return tuple(round(item, 3) for item in numbers)


def _line_group_has_markers(group: ET.Element) -> bool:
    for child in list(group):
        tag = _local_name(child.tag)
        if tag in {"use", "circle"}:
            return True
    return False


def _polyline_similarity(left: list[tuple[float, float]], right: list[tuple[float, float]]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0

    left_map = {round(x, 3): y for x, y in left}
    right_map = {round(x, 3): y for x, y in right}
    shared_x = sorted(set(left_map) & set(right_map))
    if not shared_x:
        return 0.0

    left_y = [left_map[x] for x in shared_x]
    right_y = [right_map[x] for x in shared_x]
    left_norm = _normalize_series(left_y)
    right_norm = _normalize_series(right_y)
    diffs: list[float] = []
    for left_item, right_item in zip(left_norm, right_norm):
        diffs.append(min(abs(left_item - right_item), 1.0))
    return max(0.0, 1.0 - (sum(diffs) / len(diffs)))


def _normalize_series(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    span = high - low
    if math.isclose(span, 0.0):
        return [0.0 for _ in values]
    return [(value - low) / span for value in values]


def _match_polylines(
    left: list[list[tuple[float, float]]],
    right: list[list[tuple[float, float]]],
) -> list[float]:
    if not left or not right:
        return []
    used_right: set[int] = set()
    scores: list[float] = []
    for left_line in left:
        best_idx = None
        best_score = -1.0
        for idx, right_line in enumerate(right):
            if idx in used_right:
                continue
            current = _polyline_similarity(left_line, right_line)
            if current > best_score:
                best_score = current
                best_idx = idx
        if best_idx is not None:
            used_right.add(best_idx)
            scores.append(best_score)
    return scores


def _match_polylines_relaxed(
    left: list[list[tuple[float, float]]],
    right: list[list[tuple[float, float]]],
) -> list[float]:
    if not left or not right:
        return []
    used_right: set[int] = set()
    scores: list[float] = []
    for left_line in left:
        best_idx = None
        best_score = -1.0
        for idx, right_line in enumerate(right):
            if idx in used_right:
                continue
            current = _polyline_similarity_relaxed(left_line, right_line)
            if current > best_score:
                best_score = current
                best_idx = idx
        if best_idx is not None:
            used_right.add(best_idx)
            scores.append(best_score)
    return scores


def _match_line_styles(
    left_lines: list[list[tuple[float, float]]],
    left_styles: list[dict[str, Any]],
    right_lines: list[list[tuple[float, float]]],
    right_styles: list[dict[str, Any]],
) -> list[float]:
    if not left_lines or not right_lines or not left_styles or not right_styles:
        return []
    used_right: set[int] = set()
    scores: list[float] = []
    overlap = min(len(left_lines), len(left_styles))
    for idx in range(overlap):
        left_line = left_lines[idx]
        left_style = left_styles[idx]
        best_idx = None
        best_geom = -1.0
        for right_idx, right_line in enumerate(right_lines):
            if right_idx in used_right or right_idx >= len(right_styles):
                continue
            current = _polyline_similarity_relaxed(left_line, right_line)
            if current > best_geom:
                best_geom = current
                best_idx = right_idx
        if best_idx is None:
            continue
        used_right.add(best_idx)
        scores.append(_line_style_similarity(left_style, right_styles[best_idx]))
    return scores


def _line_style_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    width_score = _numeric_list_similarity(
        [_safe_float(left.get("stroke_width"), 1.0)],
        [_safe_float(right.get("stroke_width"), 1.0)],
    )
    left_dash = tuple(left.get("stroke_dasharray") or ())
    right_dash = tuple(right.get("stroke_dasharray") or ())
    if not left_dash and not right_dash:
        dash_score = 1.0
    else:
        dash_score = _numeric_list_similarity(list(left_dash), list(right_dash))
    cap_score = 1.0 if str(left.get("stroke_linecap") or "") == str(right.get("stroke_linecap") or "") else 0.0
    join_score = 1.0 if str(left.get("stroke_linejoin") or "") == str(right.get("stroke_linejoin") or "") else 0.0
    marker_score = 1.0 if bool(left.get("has_markers")) == bool(right.get("has_markers")) else 0.0
    return width_score * 0.35 + dash_score * 0.35 + cap_score * 0.10 + join_score * 0.10 + marker_score * 0.10


def _polyline_similarity_relaxed(
    left: list[tuple[float, float]],
    right: list[tuple[float, float]],
) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    overlap = min(len(left), len(right))
    if overlap == 0:
        return 0.0
    left_y = [point[1] for point in left[:overlap]]
    right_y = [point[1] for point in right[:overlap]]
    left_norm = _normalize_series(left_y)
    right_norm = _normalize_series(right_y)
    diffs = [min(abs(a - b), 1.0) for a, b in zip(left_norm, right_norm)]
    base = 1.0 - (sum(diffs) / len(diffs) if diffs else 1.0)
    length_penalty = abs(len(left) - len(right)) / max(len(left), len(right), 1)
    return max(0.0, base * (1.0 - 0.2 * length_penalty))


def _match_points(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    tolerance: float,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if not left or not right:
        return []
    used_right: set[int] = set()
    matched: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for left_point in left:
        x1 = _safe_float(left_point.get("x"), float("nan"))
        y1 = _safe_float(left_point.get("y"), float("nan"))
        best_idx = None
        best_distance = None
        for idx, right_point in enumerate(right):
            if idx in used_right:
                continue
            x2 = _safe_float(right_point.get("x"), float("nan"))
            y2 = _safe_float(right_point.get("y"), float("nan"))
            distance = math.hypot(x1 - x2, y1 - y2)
            if distance > tolerance:
                continue
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_idx = idx
        if best_idx is not None:
            used_right.add(best_idx)
            matched.append((left_point, right[best_idx]))
    return matched
