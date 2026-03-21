#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chart_agent.perception.area_svg_updater import update_area_svg
from chart_agent.perception.line_svg_updater import update_line_svg
from chart_agent.perception.scatter_svg_updater import update_scatter_svg
from chart_agent.perception.svg_perceiver import perceive_svg
import chart_agent.perception.area_svg_updater as area_svg_mod
import chart_agent.perception.line_svg_updater as line_svg_mod
import chart_agent.perception.scatter_svg_updater as scatter_svg_mod


def _load_dotenv_vars(dotenv_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not dotenv_path.exists():
        return out
    for raw in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            out[key] = value
    return out


DOTENV_VARS = _load_dotenv_vars(PROJECT_ROOT / ".env")


def _env_or_dotenv(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is not None and value != "":
        return value
    return DOTENV_VARS.get(name, default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a dataset case: read JSON + SVG, build question, apply update, export image."
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Task folder name under dataset (e.g. task2-line/del or task1-mix-area/add-change).",
    )
    parser.add_argument("--case", required=True, help="Case id (e.g. 024).")
    parser.add_argument("--qa-index", type=int, default=0, help="Which QA question to use from JSON.")
    parser.add_argument(
        "--question-only",
        action="store_true",
        help="Use only QA question, without synthesizing operation instruction from data_change.",
    )
    parser.add_argument(
        "--use-qa-as-update",
        action="store_true",
        help="Force using QA question as update input. Not recommended for delete/change tasks.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "output" / "dataset_runner"),
        help="Output directory.",
    )
    parser.add_argument(
        "--no-ai-answer",
        action="store_true",
        help="Skip calling aihubmix GPT for QA answering.",
    )
    parser.add_argument(
        "--ai-model",
        default=_env_or_dotenv("GPT_MODEL", "gpt-5.2"),
        help="Model name used on aihubmix OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--ai-base-url",
        default=_env_or_dotenv("GPT_BASE_URL", "https://aihubmix.com/v1"),
        help="OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--ai-api-key",
        default=(
            _env_or_dotenv("Aihubmix_API_KEY")
            or ""
        ),
        help="API key for aihubmix (falls back to env vars).",
    )
    parser.add_argument(
        "--resvg-bin",
        default=os.getenv("RESVG_BIN", ""),
        help="Optional explicit resvg binary path.",
    )
    return parser.parse_args()


def find_case_dir(dataset_root: Path, task: str, case_id: str) -> Path:
    task_dir = dataset_root / task
    if not task_dir.exists():
        raise FileNotFoundError(f"Task path not found: {task_dir}")
    candidates = list(task_dir.rglob(case_id))
    case_dirs = [p for p in candidates if p.is_dir()]
    if not case_dirs:
        raise FileNotFoundError(f"Case directory not found for case '{case_id}' under {task_dir}")
    return sorted(case_dirs, key=lambda p: len(str(p)))[0]


def find_case_file(case_dir: Path, case_id: str, suffix: str) -> Path:
    preferred = case_dir / f"{case_id}{suffix}"
    if preferred.exists():
        return preferred
    matches = list(case_dir.glob(f"*{suffix}"))
    if not matches:
        raise FileNotFoundError(f"No '{suffix}' file found in {case_dir}")
    return sorted(matches)[0]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def choose_qa_question(payload: dict[str, Any], qa_index: int) -> str:
    qa = payload.get("QA")
    if isinstance(qa, list) and qa:
        idx = max(0, min(qa_index, len(qa) - 1))
        item = qa[idx]
        if isinstance(item, dict) and isinstance(item.get("question"), str):
            return item["question"].strip()
    if isinstance(payload.get("question"), str):
        return payload["question"].strip()
    raise ValueError("No question found in JSON (expected QA[].question or question).")


def _format_points(points: list[dict[str, Any]]) -> str:
    out = []
    for p in points:
        try:
            x = float(p.get("x"))
            y = float(p.get("y"))
        except Exception:
            continue
        out.append(f"({x:.6f},{y:.6f})")
    return " ".join(out)


def synthesize_instruction(payload: dict[str, Any]) -> str:
    change = payload.get("data_change", {}) or {}
    chart_type = str(payload.get("chart_type", "")).lower()

    parts: list[str] = []
    add_block = change.get("add") if isinstance(change, dict) else None
    del_block = change.get("del") if isinstance(change, dict) else None
    change_block = change.get("change") if isinstance(change, dict) else None

    if chart_type == "scatter":
        points = add_block.get("points") if isinstance(add_block, dict) else None
        if points is None and isinstance(change, dict):
            points = change.get("points")
        if isinstance(points, list) and points:
            points_text = _format_points(points)
            if points_text:
                parts.append(f"新增点 {points_text}")

    if isinstance(del_block, dict):
        names = None
        if isinstance(del_block, dict):
            names = del_block.get("category_name") or del_block.get("category")
        if isinstance(names, list) and names:
            quoted = ", ".join(f"\"{name}\"" for name in names)
            parts.append(f"删除类别 {quoted}")
        elif isinstance(names, str) and names.strip():
            parts.append(f"删除类别 \"{names.strip()}\"")

    if isinstance(add_block, dict):
        add_blocks = [add_block]
    elif isinstance(add_block, list):
        add_blocks = [item for item in add_block if isinstance(item, dict)]
    else:
        add_blocks = []
    if add_blocks:
        for idx, block in enumerate(add_blocks):
            values = block.get("values")
            if not isinstance(values, list) or not values:
                continue
            values_text = ", ".join(str(v) for v in values)
            add_name = block.get("category_name")
            add_names = add_name if isinstance(add_name, list) else [add_name]
            name = add_names[idx] if idx < len(add_names) else None
            if isinstance(name, str) and name.strip():
                parts.append(f"新增系列 \"{name.strip()}\" : [{values_text}]")
            else:
                parts.append(f"新增系列: [{values_text}]")

    if isinstance(change_block, dict):
        changes = change_block.get("changes")
        first_change = changes[0] if isinstance(changes, list) and changes and isinstance(changes[0], dict) else {}
        change_name = first_change.get("category_name")
        years = first_change.get("years")
        values = first_change.get("values")
        year = years[0] if isinstance(years, list) and years else None
        value = values[0] if isinstance(values, list) and values else None
        if year is not None and value is not None:
            if isinstance(change_name, str) and change_name.strip():
                parts.append(f"将 \"{change_name.strip()}\" 在 {year} 年改为 {value}")
            else:
                parts.append(f"在 {year} 年改为 {value}")

    return "；然后 ".join(parts).strip()


def build_question(
    payload: dict[str, Any], qa_question: str, question_only: bool, use_qa_as_update: bool
) -> tuple[str, str]:
    if question_only or use_qa_as_update:
        return qa_question, qa_question
    instruction = synthesize_instruction(payload)
    update_question = instruction or qa_question
    return update_question, qa_question


def split_update_commands(question: str) -> list[str]:
    text = str(question or "").strip()
    if not text:
        return []
    parts = re_split(text)
    return [p.strip() for p in parts if p.strip()]


def re_split(text: str) -> list[str]:
    import re

    return re.split(r"\s*(?:[;；\n]+|然后|并且|同时|and then|then)\s*", text, flags=re.IGNORECASE)


def area_command_rank(command: str) -> int:
    import re

    text = str(command or "")
    if re.search(r"(delete|remove|drop|删|删除|去掉|移除|去除|剔除)", text, re.IGNORECASE):
        return 0
    if ("[" not in text and "【" not in text) and (
        re.search(r"\b(19|20)\d{2}\b", text) or re.search(r"\d+\s*年", text)
    ):
        return 1
    return 2


def resolve_resvg_bin(explicit: str = "") -> str | None:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if path.exists() and os.access(path, os.X_OK):
            return str(path)
    direct = shutil.which("resvg")
    if direct:
        return direct

    candidates = [
        Path("/opt/anaconda3/envs/scatter/bin/resvg"),
        Path("/opt/anaconda3/bin/resvg"),
        Path("/opt/anaconda3/pkgs/resvg-0.46.0-h748bcf4_0/bin/resvg"),
    ]
    for c in candidates:
        if c.exists() and os.access(c, os.X_OK):
            return str(c)
    return None


def run_case(
    *,
    json_payload: dict[str, Any],
    update_question: str,
    qa_question: str,
    svg_path: Path,
    output_svg: Path,
    output_png: Path,
    resvg_bin: str = "",
) -> dict[str, Any]:
    resolved_resvg = resolve_resvg_bin(resvg_bin)
    if resolved_resvg:
        bin_dir = str(Path(resolved_resvg).parent)
        os.environ["PATH"] = f"{bin_dir}:{os.environ.get('PATH', '')}"
    has_resvg = bool(shutil.which("resvg"))
    if not has_resvg:
        # Keep SVG updates functional even when PNG renderer is unavailable.
        area_svg_mod.render_svg_to_png = lambda _svg, png: png
        line_svg_mod.render_svg_to_png = lambda _svg, png: png
        scatter_svg_mod.render_svg_to_png = lambda _svg, png: png

    perception = perceive_svg(str(svg_path), question=update_question, llm=None)
    perceived_type = str(perception.get("chart_type") or "").lower()
    json_type = str(json_payload.get("chart_type") or "").lower()
    chart_type = perceived_type if perceived_type and perceived_type != "unknown" else json_type
    mapping_info = perception.get("mapping_info", {}) or {}

    result: dict[str, Any] = {
        "chart_type": chart_type,
        "update_question_used": update_question,
        "qa_question": qa_question,
        "output_svg_path": str(output_svg),
        "output_png_path": str(output_png),
        "resvg_bin": resolved_resvg,
    }

    output_svg.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    try:
        if chart_type == "line":
            png_path = update_line_svg(
                str(svg_path),
                update_question,
                mapping_info,
                output_path=str(output_png),
                svg_output_path=str(output_svg),
                llm=None,
            )
            result["output_png_path"] = png_path
        elif chart_type == "area":
            commands = split_update_commands(update_question) or [update_question]
            if len(commands) > 1:
                commands = sorted(commands, key=area_command_rank)
            current_svg = svg_path
            png_path = str(output_png)
            for idx, command in enumerate(commands):
                is_last = idx == len(commands) - 1
                step_svg = output_svg if is_last else output_svg.with_name(f"{output_svg.stem}_step{idx+1}.svg")
                step_png = output_png if is_last else output_png.with_name(f"{output_png.stem}_step{idx+1}.png")
                png_path = update_area_svg(
                    str(current_svg),
                    command,
                    mapping_info,
                    output_path=str(step_png),
                    svg_output_path=str(step_svg),
                    llm=None,
                )
                current_svg = step_svg
            result["output_png_path"] = png_path
        elif chart_type == "scatter":
            data_change = json_payload.get("data_change") or {}
            points = ((((data_change.get("add") or {}) if isinstance(data_change, dict) else {}).get("points")) or [])
            if not points and isinstance(data_change, dict):
                points = data_change.get("points") or []
            new_points = []
            if isinstance(points, list):
                for p in points:
                    if not isinstance(p, dict):
                        continue
                    try:
                        point = {"x": float(p.get("x")), "y": float(p.get("y"))}
                    except Exception:
                        continue
                    for key in ("color", "point_color", "fill", "rgb"):
                        value = p.get(key)
                        if isinstance(value, str) and value.strip():
                            point[key] = value.strip()
                    new_points.append(point)
            if not new_points:
                raise ValueError("Scatter case has no data_change.add.points for rendering.")
            png_path = update_scatter_svg(
                str(svg_path),
                new_points,
                mapping_info,
                output_path=str(output_png),
                svg_output_path=str(output_svg),
                chart_type="scatter",
            )
            result["output_png_path"] = png_path
        else:
            raise ValueError(f"Unsupported chart_type '{chart_type}' for this runner.")
        result["ok"] = True
        if not has_resvg:
            result["png_ok"] = False
            result["note"] = "SVG updated successfully, but PNG was not rendered because resvg is unavailable."
        else:
            result["png_ok"] = True
    except Exception as exc:
        result["ok"] = False
        result["error"] = str(exc)
        if output_svg.exists():
            result["note"] = "SVG updated successfully, but PNG render failed (likely missing resvg)."
        else:
            result["note"] = "Update failed before SVG output was generated."
    return result


def _mime_from_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix == ".jpg" or suffix == ".jpeg":
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".svg":
        return "image/svg+xml"
    return "application/octet-stream"


def _image_data_url(path: Path) -> str:
    raw = path.read_bytes()
    mime = _mime_from_path(path)
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _extract_answer_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                chunks.append(text.strip())
        return "\n".join(chunks).strip()
    return ""


def call_aihubmix_gpt_answer(
    *,
    question: str,
    image_path: Path,
    model: str,
    base_url: str,
    api_key: str,
    timeout_sec: int = 90,
) -> dict[str, Any]:
    if not api_key.strip():
        return {"ok": False, "error": "missing api key (set --ai-api-key or env Aihubmix_API_KEY)."}
    if not image_path.exists():
        return {"ok": False, "error": f"image not found: {image_path}"}
    try:
        import openai
    except Exception as exc:
        return {
            "ok": False,
            "error": (
                "openai package is required for AI answer call: "
                f"{exc}; python={sys.executable}; version={sys.version.split()[0]}"
            ),
        }

    data_url = _image_data_url(image_path)
    try:
        client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            timeout=timeout_sec,
        )
        response = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        )
    except Exception as exc:  # pragma: no cover - network/runtime guard
        err = str(exc)
        if "CERTIFICATE_VERIFY_FAILED" in err or "certificate verify failed" in err.lower():
            err = (
                f"{err}. TLS cert verify failed. "
                "Set env SSL_CERT_FILE to a valid CA bundle "
                "(for conda: python -c 'import certifi; print(certifi.where())')."
            )
        return {"ok": False, "error": err}
    parsed = response.model_dump() if hasattr(response, "model_dump") else {}
    content = None
    try:
        content = response.choices[0].message.content
    except Exception:
        content = None
    answer = _extract_answer_content(content)
    if not answer:
        return {"ok": False, "error": "empty model answer", "raw": parsed}
    return {
        "ok": True,
        "answer": answer,
        "model": model,
        "base_url": base_url,
        "image_used": str(image_path),
        "raw": parsed if parsed else {"content": content},
    }


def main() -> None:
    args = parse_args()
    dataset_root = PROJECT_ROOT / "dataset"
    case_dir = find_case_dir(dataset_root, args.task, args.case)
    json_path = find_case_file(case_dir, args.case, ".json")
    svg_path = find_case_file(case_dir, args.case, ".svg")

    payload = load_json(json_path)
    qa_question = choose_qa_question(payload, args.qa_index)
    update_question, qa_question = build_question(
        payload, qa_question, args.question_only, args.use_qa_as_update
    )

    out_dir = Path(args.out_dir).resolve() / args.task.replace("/", "_") / args.case
    output_svg = out_dir / f"{args.case}_updated.svg"
    output_png = out_dir / f"{args.case}_updated.png"

    result = run_case(
        json_payload=payload,
        update_question=update_question,
        qa_question=qa_question,
        svg_path=svg_path,
        output_svg=output_svg,
        output_png=output_png,
        resvg_bin=args.resvg_bin,
    )
    result["task"] = args.task
    result["case"] = args.case
    result["case_dir"] = str(case_dir)
    result["json_path"] = str(json_path)
    result["svg_path"] = str(svg_path)

    if not args.no_ai_answer:
        image_for_ai = Path(result["output_png_path"])
        if not image_for_ai.exists():
            svg_fallback = Path(result["output_svg_path"])
            if svg_fallback.exists():
                image_for_ai = svg_fallback
        result["ai_answer"] = call_aihubmix_gpt_answer(
            question=qa_question,
            image_path=image_for_ai,
            model=args.ai_model,
            base_url=args.ai_base_url,
            api_key=args.ai_api_key,
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
