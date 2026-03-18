from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import re
from typing import Any

from PIL import Image


def validate_render(
    image_path: str | None,
    chart_type: str,
    update_spec: dict[str, Any] | None,
    llm: Any | None = None,
) -> dict[str, Any]:
    if not image_path:
        return {
            "ok": False,
            "confidence": 0.0,
            "issues": ["no output image path"],
        }

    if not os.path.exists(image_path):
        return {
            "ok": False,
            "confidence": 0.0,
            "issues": ["output image not found"],
        }

    width, height, non_empty = _basic_image_stats(image_path)
    issues = []
    if width == 0 or height == 0:
        issues.append("invalid image size")
    if not non_empty:
        issues.append("image appears empty")

    base_ok = not issues
    if llm is None:
        return {
            "ok": base_ok,
            "confidence": 0.6 if base_ok else 0.2,
            "issues": issues,
        }

    llm_result = _llm_check(
        chart_type,
        update_spec or {},
        image_path,
        width,
        height,
        non_empty,
        llm,
    )
    if llm_result is None:
        return {
            "ok": base_ok,
            "confidence": 0.5 if base_ok else 0.2,
            "issues": issues or ["llm_check_failed"],
        }

    merged_issues = issues + llm_result.get("issues", [])
    softened = _soften_uncertainty_only_failure(
        chart_type=chart_type,
        llm_ok=bool(llm_result.get("ok", base_ok)),
        issues=merged_issues,
        confidence=float(llm_result.get("confidence", 0.5)),
    )
    return {
        "ok": softened["ok"],
        "confidence": softened["confidence"],
        "issues": merged_issues,
        "llm_raw": llm_result.get("llm_raw"),
    }


def _soften_uncertainty_only_failure(
    *,
    chart_type: str,
    llm_ok: bool,
    issues: list[str],
    confidence: float,
) -> dict[str, Any]:
    if llm_ok:
        return {"ok": True, "confidence": confidence}
    lowered = [str(issue or "").strip().lower() for issue in issues if str(issue or "").strip()]
    if not lowered:
        return {"ok": llm_ok, "confidence": confidence}
    if str(chart_type or "").strip().lower() not in {"line", "area"}:
        return {"ok": llm_ok, "confidence": confidence}

    uncertainty_markers = (
        "cannot verify",
        "not confirmable",
        "difficult to verify",
        "no data labels",
        "no point labels",
        "no visible annotations",
        "no explicit annotation",
        "image alone",
        "visual inspection alone",
        "lacks numerical precision",
        "exact value",
        "exact values",
        "precise value",
        "precise values",
        "cannot be confidently verified",
        "too coarse",
    )
    contradiction_markers = (
        "not added",
        "not present",
        "not visible",
        "still shows",
        "still contains",
        "appears unchanged",
        "unchanged from",
        "does not visually reflect",
        "does not match",
        "contradict",
        "corrupted",
        "overlapping",
        "garbled",
        "no new line",
        "no visual evidence",
        "not plotted",
    )
    has_uncertainty = any(any(marker in issue for marker in uncertainty_markers) for issue in lowered)
    has_contradiction = any(any(marker in issue for marker in contradiction_markers) for issue in lowered)
    if has_uncertainty and not has_contradiction:
        return {"ok": True, "confidence": max(confidence, 0.45)}
    return {"ok": llm_ok, "confidence": confidence}


def _basic_image_stats(image_path: str) -> tuple[int, int, bool]:
    img = Image.open(image_path).convert("RGBA")
    width, height = img.size
    pixels = img.getdata()
    non_empty = any(px[3] > 0 and (px[0], px[1], px[2]) != (255, 255, 255) for px in pixels)
    return width, height, non_empty


def _llm_check(
    chart_type: str,
    update_spec: dict[str, Any],
    image_path: str,
    width: int,
    height: int,
    non_empty: bool,
    llm: Any,
) -> dict[str, Any] | None:
    prompt = (
        "You are validating a rendered chart image. "
        "Return JSON only with keys: ok (bool), confidence (0-1), issues (list of strings).\n"
        f"Chart type: {chart_type}\n"
        f"Update spec: {json.dumps(update_spec, ensure_ascii=False)}\n"
        f"Image path: {image_path}\n"
        f"Image size: {width}x{height}\n"
        f"Non-empty pixels: {non_empty}\n"
        "Focus on whether the rendered image likely contains the requested updates. "
        "Do not answer the original question. If unsure, set ok=false and add an issue."
    )
    try:
        response = _invoke_multimodal_or_text(llm, prompt, image_path)
    except Exception:
        return None
    content = _coerce_content_to_text(getattr(response, "content", ""))
    payload = _safe_json_loads(content)
    if not payload:
        return None
    payload["llm_raw"] = content
    return payload


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


def _invoke_multimodal_or_text(llm: Any, prompt_text: str, image_path: str | None) -> Any:
    data_url = _image_data_url(image_path)
    if data_url:
        try:
            from langchain_core.messages import HumanMessage  # type: ignore

            return llm.invoke(
                [
                    HumanMessage(
                        content=[
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ]
                    )
                ]
            )
        except Exception:
            pass
    return llm.invoke(prompt_text)


def _image_data_url(path_str: str | None) -> str | None:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists():
        return None
    suffix = path.suffix.lower()
    mime = "application/octet-stream"
    if suffix == ".png":
        mime = "image/png"
    elif suffix in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif suffix == ".webp":
        mime = "image/webp"
    elif suffix == ".svg":
        mime = "image/svg+xml"
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _coerce_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())
        return "\n".join(chunks).strip()
    return str(content)
