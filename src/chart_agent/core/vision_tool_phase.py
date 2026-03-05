from __future__ import annotations

import base64
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from chart_agent.perception.svg_renderer import render_svg_to_png

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "add_point",
        "description": "Draw a point marker at SVG coordinate.",
        "args": {
            "x": "number",
            "y": "number",
            "radius": "number, optional, default 3",
            "color": "string hex color, optional, default #ff2d55",
            "label": "string, optional",
        },
    },
    {
        "name": "draw_line",
        "description": "Draw a line segment between two SVG coordinates.",
        "args": {
            "x1": "number",
            "y1": "number",
            "x2": "number",
            "y2": "number",
            "width": "number, optional, default 1.6",
            "color": "string hex color, optional, default #ff9500",
            "label": "string, optional",
        },
    },
    {
        "name": "highlight_rect",
        "description": "Highlight a rectangle region in SVG coordinates.",
        "args": {
            "x1": "number",
            "y1": "number",
            "x2": "number",
            "y2": "number",
            "width": "number, optional, default 1.2",
            "color": "string hex color, optional, default #007aff",
            "fill_opacity": "number 0-1, optional, default 0.08",
            "label": "string, optional",
        },
    },
    {
        "name": "add_text",
        "description": "Add a small text label at SVG coordinate.",
        "args": {
            "x": "number",
            "y": "number",
            "text": "string",
            "color": "string hex color, optional, default #111111",
            "font_size": "number, optional, default 10",
        },
    },
]


def run_visual_tool_phase(
    *,
    question: str,
    chart_type: str,
    data_summary: dict[str, Any],
    image_path: str | None,
    llm: Any,
    svg_path: str | None = None,
    max_tool_calls: int = 6,
) -> dict[str, Any]:
    svg = Path(svg_path or "").expanduser() if svg_path else None
    base_image = Path(image_path or "").expanduser() if image_path else None
    base_image_ok = bool(base_image and base_image.exists() and base_image.is_file())
    if svg is None or not svg.exists() or svg.suffix.lower() != ".svg":
        return {
            "ok": False,
            "reason": "svg_missing",
            "tool_calls": [],
            "augmented_svg_path": None,
            "augmented_image_path": (str(base_image) if base_image_ok else image_path),
        }

    width, height = _svg_canvas_size(svg)
    plan = _plan_tool_calls(
        question=question,
        chart_type=chart_type,
        data_summary=data_summary,
        image_path=svg,
        llm=llm,
        canvas_width=width,
        canvas_height=height,
    )
    tool_calls = plan.get("tool_calls", [])
    if not isinstance(tool_calls, list) or not tool_calls:
        return {
            "ok": True,
            "reason": "no_tool_needed",
            "tool_calls": [],
            "planner": plan,
            "augmented_svg_path": str(svg),
            "augmented_image_path": (str(base_image) if base_image_ok else str(svg)),
        }

    out_svg = svg.with_name(f"{svg.stem}_tool_aug.svg")
    exec_result = _execute_svg_tool_calls(svg, out_svg, tool_calls, max_tool_calls=max_tool_calls)
    exec_result["planner"] = plan

    out_png = out_svg.with_suffix(".png")
    try:
        render_svg_to_png(str(out_svg), str(out_png))
        exec_result["augmented_image_path"] = str(out_png)
    except Exception:
        exec_result["augmented_image_path"] = (str(base_image) if base_image_ok else str(out_svg))

    return exec_result


def _plan_tool_calls(
    *,
    question: str,
    chart_type: str,
    data_summary: dict[str, Any],
    image_path: Path,
    llm: Any,
    canvas_width: float,
    canvas_height: float,
) -> dict[str, Any]:
    prompt = (
        "You can use visual markup tools on a chart image to improve answer reliability.\n"
        "Important: chart updates have already been applied to this SVG.\n"
        "Do NOT re-apply edits and do NOT add factual conclusions onto the chart.\n"
        "Only add light visual guides for answering the QA question.\n"
        "First, explicitly state your understanding of the QA question.\n"
        "Return JSON only with schema:\n"
        "{\"qa_understanding\":string,\"tool_calls\":[{\"tool\":string,\"args\":object}],\"notes\":string}\n"
        "Rules:\n"
        "- Use only tools listed below.\n"
        "- Use SVG coordinates only.\n"
        f"- Canvas size is width={canvas_width:.2f}, height={canvas_height:.2f}.\n"
        "- Keep overlays minimal and non-occluding.\n"
        "- If no visual enhancement is needed, return empty tool_calls.\n"
        "- Max 6 tool calls.\n"
        f"Question: {question}\n"
        f"Chart type: {chart_type}\n"
        f"Tools: {json.dumps(TOOL_SPECS, ensure_ascii=False)}\n"
    )

    content = ""
    try:
        response = _invoke_multimodal_or_text(llm, prompt, image_path)
        content = _coerce_content_to_text(getattr(response, "content", ""))
    except Exception as exc:
        return {"tool_calls": [], "notes": f"planner_error: {exc}", "llm_success": False}

    payload = _safe_json_loads(content)
    if not payload:
        return {"tool_calls": [], "notes": "planner_non_json", "llm_success": False, "llm_raw": content}
    raw_calls = payload.get("tool_calls", [])
    calls = _coerce_tool_calls(raw_calls)
    return {
        "qa_understanding": str(payload.get("qa_understanding") or ""),
        "tool_calls": calls,
        "notes": str(payload.get("notes") or ""),
        "llm_success": True,
        "llm_raw": content,
    }


def _coerce_tool_calls(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        tool = str(item.get("tool") or "").strip()
        args = item.get("args", {})
        if tool not in {"add_point", "draw_line", "highlight_rect", "add_text"}:
            continue
        if not isinstance(args, dict):
            args = {}
        out.append({"tool": tool, "args": args})
    return out


def _execute_svg_tool_calls(
    svg_path: Path,
    out_svg: Path,
    tool_calls: list[dict[str, Any]],
    *,
    max_tool_calls: int,
) -> dict[str, Any]:
    tree = ET.parse(svg_path)
    root = tree.getroot()
    ns = _svg_ns(root)
    canvas_w, canvas_h = _svg_canvas_size(svg_path)
    parent = _find_overlay_parent(root, ns)
    overlay = _ensure_overlay_group(parent, ns)

    executed: list[dict[str, Any]] = []
    errors: list[str] = []

    for idx, call in enumerate(tool_calls[:max_tool_calls]):
        tool = str(call.get("tool") or "")
        args = call.get("args", {}) if isinstance(call.get("args"), dict) else {}
        try:
            if tool == "add_point":
                _svg_add_point(overlay, ns, args, canvas_w, canvas_h)
            elif tool == "draw_line":
                _svg_draw_line(overlay, ns, args, canvas_w, canvas_h)
            elif tool == "highlight_rect":
                _svg_highlight_rect(overlay, ns, args, canvas_w, canvas_h)
            elif tool == "add_text":
                _svg_add_text(overlay, ns, args, canvas_w, canvas_h)
            else:
                raise ValueError(f"unknown tool: {tool}")
            executed.append({"index": idx + 1, "tool": tool, "args": args})
        except Exception as exc:
            errors.append(f"tool[{idx + 1}] {tool} failed: {exc}")

    out_svg.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_svg, encoding="utf-8", xml_declaration=True)
    return {
        "ok": len(executed) > 0,
        "tool_calls": tool_calls[:max_tool_calls],
        "executed": executed,
        "errors": errors,
        "augmented_svg_path": str(out_svg),
        "augmented_image_path": str(out_svg),
    }


def _svg_add_point(overlay: ET.Element, ns: str, args: dict[str, Any], w: float, h: float) -> None:
    x = _clamp(_as_float(args.get("x"), 0.0), 0.0, w)
    y = _clamp(_as_float(args.get("y"), 0.0), 0.0, h)
    r = _clamp(_as_float(args.get("radius"), 3.0), 0.8, 8.0)
    color = _safe_color(args.get("color"), "#ff2d55")

    ET.SubElement(
        overlay,
        _nstag(ns, "circle"),
        {
            "cx": f"{x:.6f}",
            "cy": f"{y:.6f}",
            "r": f"{r:.6f}",
            "fill": color,
            "fill-opacity": "0.88",
            "stroke": "#000000",
            "stroke-width": "0.6",
        },
    )

    label = _short_text(str(args.get("label") or "").strip(), 28)
    if label:
        _svg_text(overlay, ns, x + r + 2, y - r - 1, label, "#111111", 9.5)


def _svg_draw_line(overlay: ET.Element, ns: str, args: dict[str, Any], w: float, h: float) -> None:
    x1 = _clamp(_as_float(args.get("x1"), 0.0), 0.0, w)
    y1 = _clamp(_as_float(args.get("y1"), 0.0), 0.0, h)
    x2 = _clamp(_as_float(args.get("x2"), 0.0), 0.0, w)
    y2 = _clamp(_as_float(args.get("y2"), 0.0), 0.0, h)
    width = _clamp(_as_float(args.get("width"), 1.6), 0.6, 4.0)
    color = _safe_color(args.get("color"), "#ff9500")

    ET.SubElement(
        overlay,
        _nstag(ns, "line"),
        {
            "x1": f"{x1:.6f}",
            "y1": f"{y1:.6f}",
            "x2": f"{x2:.6f}",
            "y2": f"{y2:.6f}",
            "stroke": color,
            "stroke-opacity": "0.92",
            "stroke-width": f"{width:.6f}",
            "stroke-linecap": "round",
        },
    )

    label = _short_text(str(args.get("label") or "").strip(), 28)
    if label:
        _svg_text(overlay, ns, (x1 + x2) / 2.0 + 2, (y1 + y2) / 2.0 - 2, label, "#111111", 9.5)


def _svg_highlight_rect(overlay: ET.Element, ns: str, args: dict[str, Any], w: float, h: float) -> None:
    x1 = _clamp(_as_float(args.get("x1"), 0.0), 0.0, w)
    y1 = _clamp(_as_float(args.get("y1"), 0.0), 0.0, h)
    x2 = _clamp(_as_float(args.get("x2"), 0.0), 0.0, w)
    y2 = _clamp(_as_float(args.get("y2"), 0.0), 0.0, h)
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    width = _clamp(_as_float(args.get("width"), 1.2), 0.6, 4.0)
    color = _safe_color(args.get("color"), "#007aff")
    fill_opacity = _clamp(_as_float(args.get("fill_opacity"), 0.08), 0.0, 0.25)

    area_ratio = ((right - left) * (bottom - top)) / max(1.0, (w * h))
    if area_ratio > 0.35:
        fill_opacity = 0.0
    elif area_ratio > 0.18:
        fill_opacity = min(fill_opacity, 0.05)

    ET.SubElement(
        overlay,
        _nstag(ns, "rect"),
        {
            "x": f"{left:.6f}",
            "y": f"{top:.6f}",
            "width": f"{(right - left):.6f}",
            "height": f"{(bottom - top):.6f}",
            "fill": color,
            "fill-opacity": f"{fill_opacity:.4f}",
            "stroke": color,
            "stroke-opacity": "0.92",
            "stroke-width": f"{width:.6f}",
        },
    )

    label = _short_text(str(args.get("label") or "").strip(), 28)
    if label:
        _svg_text(overlay, ns, left + 2, top + 11, label, "#111111", 9.5)


def _svg_add_text(overlay: ET.Element, ns: str, args: dict[str, Any], w: float, h: float) -> None:
    x = _clamp(_as_float(args.get("x"), 0.0), 0.0, w)
    y = _clamp(_as_float(args.get("y"), 0.0), 0.0, h)
    text = _short_text(str(args.get("text") or "").strip(), 36)
    if not text:
        return
    color = _safe_color(args.get("color"), "#111111")
    size = _clamp(_as_float(args.get("font_size"), 10.0), 7.0, 16.0)
    _svg_text(overlay, ns, x, y, text, color, size)


def _svg_text(
    overlay: ET.Element,
    ns: str,
    x: float,
    y: float,
    text: str,
    color: str,
    size: float,
) -> None:
    t = ET.SubElement(
        overlay,
        _nstag(ns, "text"),
        {
            "x": f"{x:.6f}",
            "y": f"{y:.6f}",
            "fill": color,
            "font-size": f"{size:.2f}",
            "font-family": "DejaVu Sans, Arial, sans-serif",
        },
    )
    t.text = text


def _find_overlay_parent(root: ET.Element, ns: str) -> ET.Element:
    axes = root.find(f".//{_nstag(ns, 'g')}[@id='axes_1']")
    if axes is not None:
        return axes
    return root


def _ensure_overlay_group(parent: ET.Element, ns: str) -> ET.Element:
    existing = parent.find(f"./{_nstag(ns, 'g')}[@id='tool_aug_overlay']")
    if existing is not None:
        return existing
    return ET.SubElement(parent, _nstag(ns, "g"), {"id": "tool_aug_overlay"})


def _svg_ns(root: ET.Element) -> str:
    if root.tag.startswith("{") and "}" in root.tag:
        return root.tag[1 : root.tag.find("}")]
    return "http://www.w3.org/2000/svg"


def _nstag(ns: str, name: str) -> str:
    return f"{{{ns}}}{name}" if ns else name


def _svg_canvas_size(svg_path: Path) -> tuple[float, float]:
    try:
        root = ET.parse(svg_path).getroot()
    except Exception:
        return 1000.0, 800.0

    vb = str(root.get("viewBox") or "").strip()
    if vb:
        parts = vb.replace(",", " ").split()
        if len(parts) == 4:
            try:
                w = float(parts[2])
                h = float(parts[3])
                if w > 0 and h > 0:
                    return w, h
            except Exception:
                pass

    w = _parse_svg_length(root.get("width"))
    h = _parse_svg_length(root.get("height"))
    if w > 0 and h > 0:
        return w, h
    return 1000.0, 800.0


def _parse_svg_length(value: Any) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    m = re.match(r"^(-?\d+(?:\.\d+)?)", text)
    if not m:
        return 0.0
    try:
        return float(m.group(1))
    except Exception:
        return 0.0


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _short_text(text: str, limit: int) -> str:
    value = (text or "").strip()
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


def _safe_color(raw: Any, default: str) -> str:
    text = str(raw or "").strip()
    if re.match(r"^#[0-9a-fA-F]{6}$", text):
        return text
    if re.match(r"^#[0-9a-fA-F]{3}$", text):
        return text
    return default


def _invoke_multimodal_or_text(llm: Any, prompt_text: str, image_path: Path) -> Any:
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


def _image_data_url(path: Path) -> str | None:
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
