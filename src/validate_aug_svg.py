#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from chart_agent.perception import area_svg_updater as area_ops
from chart_agent.perception.svg_perceiver import perceive_svg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate predicted area SVGs against *_aug.svg ground truth."
    )
    parser.add_argument("--pred-root", help="Prediction run dir, e.g. output/dataset_records/.../run_xxx")
    parser.add_argument("--gt-root", help="Ground-truth dataset dir, e.g. dataset/task1/add-change")
    parser.add_argument("--pred-svg", help="Single predicted SVG path.")
    parser.add_argument("--gt-svg", help="Single ground-truth SVG path.")
    parser.add_argument("--tol", type=float, default=1e-2, help="Max absolute error tolerance per point.")
    parser.add_argument(
        "--mode",
        choices=["top-only", "full"],
        default="top-only",
        help="Validation mode: top-only compares only the highest series; full compares all series.",
    )
    parser.add_argument("--out", default="", help="Output report path (json).")
    return parser.parse_args()


def _normalize_label(label: str) -> str:
    text = (label or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def _build_fill_to_label(items: list[dict[str, Any]]) -> tuple[dict[str, str], list[str]]:
    by_fill: dict[str, str] = {}
    ordered_labels: list[str] = []
    for item in items:
        label = str(item.get("label") or "").strip()
        if not label:
            continue
        ordered_labels.append(label)
        fill = str(item.get("fill") or "").strip().lower()
        if fill:
            by_fill[fill] = label
    return by_fill, ordered_labels


def _extract_series_by_label(svg_path: Path) -> dict[str, Any]:
    content = svg_path.read_text(encoding="utf-8")
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    root = ET.parse(svg_path, parser=parser).getroot()
    axes = root.find(f'.//{{{area_ops.SVG_NS}}}g[@id="axes_1"]')
    if axes is None:
        raise ValueError(f"axes_1 missing: {svg_path}")

    _, legend_items = area_ops._extract_legend_items(root, content)
    fill_to_label, _ = _build_fill_to_label(legend_items)

    perceived = perceive_svg(str(svg_path), question="", llm=None)
    mapping = perceived.get("mapping_info", {}) if isinstance(perceived, dict) else {}
    y_ticks = mapping.get("y_ticks", []) if isinstance(mapping, dict) else []
    if len(y_ticks) < 2:
        raise ValueError(f"insufficient y_ticks: {svg_path}")

    areas = area_ops._extract_area_groups(axes)
    if not areas:
        raise ValueError(f"no area groups found: {svg_path}")
    _, series_values = area_ops._area_series_values(areas, y_ticks)

    series_items: list[dict[str, Any]] = []
    for idx, area in enumerate(areas):
        fill = str(area.get("fill") or "").strip().lower()
        label = fill_to_label.get(fill, "")
        series_items.append(
            {
                "idx": idx,
                "fill": fill,
                "label": label or "",
                "values": series_values[idx],
            }
        )

    return {
        "series_items": series_items,
    }


def _compare_one(pred_svg: Path, gt_svg: Path, tol: float) -> dict[str, Any]:
    pred = _extract_series_by_label(pred_svg)
    gt = _extract_series_by_label(gt_svg)

    pred_items = pred["series_items"]
    gt_items = gt["series_items"]
    pred_used: set[int] = set()
    gt_used: set[int] = set()
    label_metrics: list[dict[str, Any]] = []

    def _append_metric(pred_item: dict[str, Any], gt_item: dict[str, Any], mode: str) -> bool:
        pv = pred_item["values"]
        gv = gt_item["values"]
        n = min(len(pv), len(gv))
        if n == 0:
            mae = float("inf")
            max_err = float("inf")
            ok = False
        else:
            diffs = [abs(float(pv[i]) - float(gv[i])) for i in range(n)]
            mae = float(statistics.fmean(diffs))
            max_err = float(max(diffs))
            ok = max_err <= tol
        label_metrics.append(
            {
                "label_pred": pred_item.get("label") or f"__pred_idx_{pred_item['idx']}",
                "label_gt": gt_item.get("label") or f"__gt_idx_{gt_item['idx']}",
                "points_compared": n,
                "mae": mae,
                "max_err": max_err,
                "ok": ok,
                "match_mode": mode,
            }
        )
        return ok

    # First pass: match by normalized label (order-independent).
    gt_norm_to_idx: dict[str, int] = {}
    for g in gt_items:
        norm = _normalize_label(g.get("label") or "")
        if norm and norm not in gt_norm_to_idx:
            gt_norm_to_idx[norm] = g["idx"]

    all_ok = True
    for p in pred_items:
        plabel = str(p.get("label") or "")
        pnorm = _normalize_label(plabel)
        if not pnorm:
            continue
        gidx = gt_norm_to_idx.get(pnorm)
        if gidx is None:
            continue
        if gidx in gt_used:
            continue
        pred_used.add(p["idx"])
        gt_used.add(gidx)
        g = gt_items[gidx]
        if not _append_metric(p, g, "label"):
            all_ok = False

    # Second pass: for unmatched items, match by smallest MAE (shape/value similarity).
    candidates: list[tuple[float, int, int]] = []
    for p in pred_items:
        if p["idx"] in pred_used:
            continue
        for g in gt_items:
            if g["idx"] in gt_used:
                continue
            pv = p["values"]
            gv = g["values"]
            n = min(len(pv), len(gv))
            if n == 0:
                continue
            diffs = [abs(float(pv[i]) - float(gv[i])) for i in range(n)]
            mae = float(statistics.fmean(diffs))
            candidates.append((mae, p["idx"], g["idx"]))

    for _, pidx, gidx in sorted(candidates, key=lambda x: x[0]):
        if pidx in pred_used or gidx in gt_used:
            continue
        p = pred_items[pidx]
        g = gt_items[gidx]
        pred_used.add(pidx)
        gt_used.add(gidx)
        if not _append_metric(p, g, "shape"):
            all_ok = False

    missing_in_pred = []
    for g in gt_items:
        if g["idx"] not in gt_used:
            missing_in_pred.append(g.get("label") or f"__gt_idx_{g['idx']}")
    extra_in_pred = []
    for p in pred_items:
        if p["idx"] not in pred_used:
            extra_in_pred.append(p.get("label") or f"__pred_idx_{p['idx']}")

    if missing_in_pred or extra_in_pred:
        all_ok = False

    return {
        "ok": all_ok,
        "pred_svg": str(pred_svg),
        "gt_svg": str(gt_svg),
        "tol": tol,
        "common_label_count": len([m for m in label_metrics if m.get("match_mode") == "label"]),
        "shape_matched_count": len([m for m in label_metrics if m.get("match_mode") == "shape"]),
        "missing_in_pred": missing_in_pred,
        "extra_in_pred": extra_in_pred,
        "label_metrics": label_metrics,
        "note": "Series matching is order-independent: first by normalized label, then by shape/value similarity (MAE).",
    }


def _top_series_item(series_items: list[dict[str, Any]]) -> dict[str, Any]:
    if not series_items:
        raise ValueError("No series found.")
    best = None
    best_max = float("-inf")
    for item in series_items:
        vals = item.get("values") or []
        if not vals:
            continue
        cur = max(float(v) for v in vals)
        if cur > best_max:
            best_max = cur
            best = item
    if best is None:
        raise ValueError("No numeric series values found.")
    return best


def _compare_top_only(pred_svg: Path, gt_svg: Path, tol: float) -> dict[str, Any]:
    pred = _extract_series_by_label(pred_svg)
    gt = _extract_series_by_label(gt_svg)
    p = _top_series_item(pred["series_items"])
    g = _top_series_item(gt["series_items"])
    pv = p.get("values", [])
    gv = g.get("values", [])
    n = min(len(pv), len(gv))
    if n == 0:
        mae = float("inf")
        max_err = float("inf")
        ok = False
    else:
        diffs = [abs(float(pv[i]) - float(gv[i])) for i in range(n)]
        mae = float(statistics.fmean(diffs))
        max_err = float(max(diffs))
        ok = max_err <= tol
    return {
        "ok": ok,
        "pred_svg": str(pred_svg),
        "gt_svg": str(gt_svg),
        "tol": tol,
        "mode_used": "top-only",
        "top_pred_label": p.get("label") or f"__pred_idx_{p.get('idx')}",
        "top_gt_label": g.get("label") or f"__gt_idx_{g.get('idx')}",
        "points_compared": n,
        "mae": mae,
        "max_err": max_err,
        "note": "Only the highest series is compared.",
    }


def _find_pred_svg(case_dir: Path) -> Path | None:
    case_id = case_dir.name
    exact = case_dir / f"{case_id}_updated.svg"
    if exact.exists():
        return exact
    candidates = sorted(case_dir.glob("*_updated.svg"))
    if candidates:
        return candidates[0]
    return None


def _run_batch(pred_root: Path, gt_root: Path, tol: float, mode: str) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    total = 0
    ok_count = 0
    fail_count = 0

    for case_dir in sorted(pred_root.iterdir()):
        if not case_dir.is_dir() or not case_dir.name.isdigit():
            continue
        case_id = case_dir.name
        pred_svg = _find_pred_svg(case_dir)
        gt_svg = gt_root / case_id / f"{case_id}_aug.svg"
        total += 1

        if pred_svg is None:
            items.append({"case": case_id, "ok": False, "error": "pred svg not found"})
            fail_count += 1
            continue
        if not gt_svg.exists():
            items.append({"case": case_id, "ok": False, "error": f"gt svg missing: {gt_svg}"})
            fail_count += 1
            continue

        try:
            if mode == "top-only":
                result = _compare_top_only(pred_svg, gt_svg, tol)
            else:
                result = _compare_one(pred_svg, gt_svg, tol)
            result["case"] = case_id
            items.append(result)
            if result["ok"]:
                ok_count += 1
            else:
                fail_count += 1
        except Exception as exc:
            items.append({"case": case_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
            fail_count += 1

    return {
        "mode": "batch",
        "pred_root": str(pred_root),
        "gt_root": str(gt_root),
        "tol": tol,
        "mode_used": mode,
        "total": total,
        "ok": ok_count,
        "failed": fail_count,
        "items": items,
    }


def main() -> None:
    args = parse_args()

    has_single = bool(args.pred_svg and args.gt_svg)
    has_batch = bool(args.pred_root and args.gt_root)
    if not has_single and not has_batch:
        raise SystemExit("Use either (--pred-svg and --gt-svg) or (--pred-root and --gt-root).")

    if has_single:
        if args.mode == "top-only":
            report = _compare_top_only(Path(args.pred_svg), Path(args.gt_svg), args.tol)
        else:
            report = _compare_one(Path(args.pred_svg), Path(args.gt_svg), args.tol)
        report["mode"] = "single"
    else:
        report = _run_batch(Path(args.pred_root), Path(args.gt_root), args.tol, args.mode)

    out = Path(args.out) if args.out else None
    if out is None and has_batch:
        out = Path(args.pred_root) / "aug_svg_validation_report.json"
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved report to: {out}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
