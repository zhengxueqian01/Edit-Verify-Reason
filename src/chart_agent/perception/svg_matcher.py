from __future__ import annotations

from collections import Counter
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


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
    }


def _feature_summary(features: dict[str, Any]) -> dict[str, Any]:
    return {
        "elements": int(sum(features["tags"].values())),
        "unique_tags": len(features["tags"]),
        "text_nodes": int(sum(features["texts"].values())),
        "numeric_attr_keys": len(features["numeric"]),
        "path_signatures": int(sum(features["paths"].values())),
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
