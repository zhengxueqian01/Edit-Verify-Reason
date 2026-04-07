from __future__ import annotations

import json
import os
import re
import shutil
import subprocess


def default_output_paths(svg_path: str, chart_type: str) -> tuple[str, str]:
    base, _ = os.path.splitext(os.path.basename(svg_path))
    m = re.match(r"^(\d{3})(?:[_-].*)?$", base)
    case_id = m.group(1) if m else base
    category = chart_type
    operation = "update"
    scope = _derive_output_scope(svg_path)
    meta = _load_case_meta(svg_path, case_id)
    if meta:
        category = str(meta.get("chart_type") or chart_type).strip() or chart_type
        raw_op = str(meta.get("operation") or "update").strip() or "update"
        operation = _normalize_operation(raw_op)

    prefix = f"{scope}_{case_id}" if scope else case_id
    named = f"{prefix}_{_slug(category)}_{_slug(operation)}_updated"
    svg_out = os.path.join("output", chart_type, f"{named}.svg")
    png_out = os.path.join("output", chart_type, f"{named}.png")
    return svg_out, png_out


def _load_case_meta(svg_path: str, case_id: str) -> dict | None:
    candidate = os.path.join(os.path.dirname(svg_path), f"{case_id}.json")
    if not os.path.exists(candidate):
        return None
    try:
        with open(candidate, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def _normalize_operation(raw: str) -> str:
    value = raw.lower().strip()
    value = value.replace("+", "-")
    value = value.replace("_", "-")
    return re.sub(r"[^a-z0-9-]+", "-", value).strip("-") or "update"


def _derive_output_scope(svg_path: str) -> str:
    parts = [part for part in os.path.normpath(svg_path).split(os.sep) if part]
    if not parts:
        return ""
    for marker in ("dataset", "dataset0313", "dataset03212"):
        if marker in parts:
            idx = parts.index(marker)
            relevant = parts[idx + 1 : -2]
            if relevant:
                return _slug("_".join(relevant))
    return ""


def _slug(text: str) -> str:
    value = str(text).strip().lower()
    value = value.replace(" ", "-")
    value = value.replace("+", "-")
    value = value.replace("_", "-")
    value = re.sub(r"[^a-z0-9-]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or "unknown"


def render_svg_to_png(svg_path: str, output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    resvg_bin = shutil.which("resvg")
    if not resvg_bin:
        raise RuntimeError(
            "resvg binary not found. Install it (e.g. `brew install resvg`) "
            "or make sure it is on PATH."
        )
    subprocess.run([resvg_bin, svg_path, output_path], check=True)
    return output_path
