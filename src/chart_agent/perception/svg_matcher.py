from __future__ import annotations

from collections import Counter
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
    operation = str((payload or {}).get("operation") or "").strip().lower()
    candidates: list[Path] = []
    if operation == "delete":
        candidates.append(base / f"{case_id}_del.svg")
    if operation:
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
        root = ET.parse(svg_path).getroot()
    except Exception as exc:
        return {"error": f"parse_failed: {exc}"}

    tags: Counter[str] = Counter()
    attrs: Counter[str] = Counter()
    texts: Counter[str] = Counter()
    numeric: dict[str, list[float]] = {}
    paths: Counter[str] = Counter()
    for elem in root.iter():
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
        "line_series": _extract_line_series(root),
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
    return {
        "ok": True,
        "chart_type": "area",
        "predicted_svg_path": str(pred_path),
        "ground_truth_svg_path": str(gt_path),
        "score": round(boundary_score, 4),
        "metrics": {
            "top_boundary_similarity": round(boundary_score, 4),
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
    line_scores = _match_polylines(pred_lines, gt_lines)
    matched = sum(1 for score in line_scores if score >= 0.98)
    total = max(len(pred_lines), len(gt_lines), 1)
    score = matched / total
    return {
        "ok": True,
        "chart_type": "line",
        "predicted_svg_path": str(pred_path),
        "ground_truth_svg_path": str(gt_path),
        "score": round(score, 4),
        "metrics": {
            "matched_lines": matched,
            "predicted_line_count": len(pred_lines),
            "ground_truth_line_count": len(gt_lines),
            "line_match_ratio": round(score, 4),
            "average_line_similarity": round(sum(line_scores) / len(line_scores), 4) if line_scores else 0.0,
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
    matched = _match_points(pred_points, gt_points, tolerance=1.5)
    total = max(len(pred_points), len(gt_points), 1)
    score = matched / total
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
            "point_match_ratio": round(score, 4),
        },
        "predicted_summary": _feature_summary(pred_features),
        "ground_truth_summary": _feature_summary(gt_features),
    }


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _normalize_text(value: str | None) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
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


def _extract_scatter_points(root: ET.Element) -> list[tuple[float, float]]:
    axes = root.find(f'.//{{{SVG_NS}}}g[@id="axes_1"]')
    if axes is None:
        return []

    points: list[tuple[float, float]] = []
    for group_id in ("PathCollection_1", "PathCollection_update"):
        group = axes.find(f'.//{{{SVG_NS}}}g[@id="{group_id}"]')
        if group is None:
            continue
        for use_elem in group.findall(f'.//{{{SVG_NS}}}use'):
            x_attr = use_elem.get("x")
            y_attr = use_elem.get("y")
            if not x_attr or not y_attr:
                continue
            try:
                points.append((float(x_attr), float(y_attr)))
            except ValueError:
                continue
        for circle in group.findall(f'.//{{{SVG_NS}}}circle'):
            cx_attr = circle.get("cx")
            cy_attr = circle.get("cy")
            if not cx_attr or not cy_attr:
                continue
            try:
                points.append((float(cx_attr), float(cy_attr)))
            except ValueError:
                continue
    return points


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


def _match_points(
    left: list[tuple[float, float]],
    right: list[tuple[float, float]],
    *,
    tolerance: float,
) -> int:
    if not left or not right:
        return 0
    used_right: set[int] = set()
    matched = 0
    for x1, y1 in left:
        best_idx = None
        best_distance = None
        for idx, (x2, y2) in enumerate(right):
            if idx in used_right:
                continue
            distance = math.hypot(x1 - x2, y1 - y2)
            if distance > tolerance:
                continue
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_idx = idx
        if best_idx is not None:
            used_right.add(best_idx)
            matched += 1
    return matched
