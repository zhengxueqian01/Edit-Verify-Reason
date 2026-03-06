#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

PROJECT_SRC = os.path.abspath(os.path.dirname(__file__))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

from chart_agent.perception.svg_matcher import compare_svgs, resolve_ground_truth_svg


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"
PREDICTION_ROOT = PROJECT_ROOT / "output" / "dataset_records"
SVG_MATCH_ROOT = PROJECT_ROOT / "output" / "svg_match"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate generated SVGs against ground-truth SVGs."
    )
    parser.add_argument("--pred-svg", default="", help="Single predicted SVG path.")
    parser.add_argument("--gt-svg", default="", help="Single ground-truth SVG path.")
    parser.add_argument(
        "--pred-root",
        default="",
        help="Batch prediction root, e.g. output/dataset_records/.../run_xxx",
    )
    parser.add_argument(
        "--dataset-dir",
        default="",
        help="Dataset folder used to resolve ground-truth SVGs, e.g. dataset/task2-line or dataset/task1-mix-area/add-change",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max cases to validate in batch mode; 0 means all.")
    parser.add_argument(
        "--out",
        default="",
        help="Optional output JSON path. In batch mode, defaults to output/svg_match/<pred-root>.json",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_input_dir(raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    candidate = (PROJECT_ROOT / p).resolve()
    if candidate.exists():
        return candidate
    pred_candidate = (PREDICTION_ROOT / raw).resolve()
    if pred_candidate.exists():
        return pred_candidate
    return (DATASET_ROOT / raw).resolve()


def resolve_output_path(raw: str, *, pred_root: Path | None) -> Path | None:
    if raw:
        out_path = Path(raw).expanduser()
        if not out_path.is_absolute():
            out_path = (PROJECT_ROOT / out_path).resolve()
        return out_path
    if pred_root is None:
        return None
    return (SVG_MATCH_ROOT / f"{pred_root.name}.json").resolve()


def list_case_dirs(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and p.name.isdigit():
            out.append(p)
    return out


def find_pred_svg(case_dir: Path) -> Path | None:
    case_id = case_dir.name
    preferred = [
        case_dir / f"{case_id}_updated.svg",
        case_dir / f"{case_id}.svg",
    ]
    for path in preferred:
        if path.exists() and path.is_file():
            return path
    matches = sorted(
        p for p in case_dir.glob("*.svg")
        if p.is_file() and not p.name.endswith("_tool_aug.svg")
    )
    return matches[0] if matches else None


def validate_one(pred_svg: Path, gt_svg: Path) -> dict[str, Any]:
    return compare_svgs(pred_svg, gt_svg)


def validate_batch(pred_root: Path, dataset_dir: Path, limit: int) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    total = 0
    compared = 0
    score_sum = 0.0
    missing_pred = 0
    missing_gt = 0
    failed = 0

    for case_dir in list_case_dirs(pred_root):
        if limit and total >= limit:
            break
        total += 1
        case_id = case_dir.name
        pred_svg = find_pred_svg(case_dir)
        dataset_case_dir = dataset_dir / case_id
        item: dict[str, Any] = {"case": case_id}

        if pred_svg is None:
            missing_pred += 1
            item.update({"ok": False, "error": "predicted_svg_missing"})
            items.append(item)
            continue
        if not dataset_case_dir.exists():
            missing_gt += 1
            item.update({"ok": False, "predicted_svg_path": str(pred_svg), "error": "dataset_case_missing"})
            items.append(item)
            continue

        payload_path = dataset_case_dir / f"{case_id}.json"
        payload = load_json(payload_path) if payload_path.exists() else {}
        gt_svg = resolve_ground_truth_svg(dataset_case_dir, case_id, payload)
        if gt_svg is None:
            missing_gt += 1
            item.update({"ok": False, "predicted_svg_path": str(pred_svg), "error": "ground_truth_svg_missing"})
            items.append(item)
            continue

        result = compare_svgs(pred_svg, gt_svg)
        item.update(result)
        if result.get("ok"):
            compared += 1
            score_sum += float(result.get("score") or 0.0)
        else:
            failed += 1
        items.append(item)

    return {
        "mode": "batch",
        "pred_root": str(pred_root),
        "dataset_dir": str(dataset_dir),
        "total": total,
        "compared": compared,
        "missing_pred": missing_pred,
        "missing_gt": missing_gt,
        "failed": failed,
        "average_score": (score_sum / compared if compared else None),
        "items": items,
    }


def main() -> None:
    args = parse_args()
    result: dict[str, Any]
    out_path: Path | None = None

    if args.pred_svg or args.gt_svg:
        if not args.pred_svg or not args.gt_svg:
            raise SystemExit("Single mode requires both --pred-svg and --gt-svg.")
        pred_svg = Path(args.pred_svg).expanduser().resolve()
        gt_svg = Path(args.gt_svg).expanduser().resolve()
        result = {
            "mode": "single",
            **validate_one(pred_svg, gt_svg),
        }
        out_path = resolve_output_path(args.out, pred_root=None)
    else:
        if not args.pred_root or not args.dataset_dir:
            raise SystemExit("Batch mode requires both --pred-root and --dataset-dir.")
        pred_root = resolve_input_dir(args.pred_root)
        dataset_dir = resolve_input_dir(args.dataset_dir)
        if not pred_root.exists() or not pred_root.is_dir():
            raise SystemExit(f"Prediction root not found: {pred_root}")
        if not dataset_dir.exists() or not dataset_dir.is_dir():
            raise SystemExit(f"Dataset dir not found: {dataset_dir}")
        result = validate_batch(pred_root, dataset_dir, args.limit)
        out_path = resolve_output_path(args.out, pred_root=pred_root)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
