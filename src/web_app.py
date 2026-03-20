#!/usr/bin/env python3
from __future__ import annotations

import cgi
import json
import os
import re
import shutil
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    from .main import run_main
except ImportError:
    from main import run_main

try:
    from .chart_agent.config import get_web_model_options, get_web_task_model_defaults
except ImportError:
    from chart_agent.config import get_web_model_options, get_web_task_model_defaults


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = Path(__file__).resolve().parent / "web"
UPLOAD_ROOT = PROJECT_ROOT / "output" / "web_uploads"


class AppHandler(BaseHTTPRequestHandler):
    server_version = "ChartAgentWeb/1.0"
    protocol_version = "HTTP/1.1"

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_file(self, file_path: Path) -> None:
        if not file_path.exists() or not file_path.is_file():
            self.send_error(404, "Not found")
            return
        suffix = file_path.suffix.lower()
        mime = "application/octet-stream"
        if suffix == ".html":
            mime = "text/html; charset=utf-8"
        elif suffix == ".png":
            mime = "image/png"
        elif suffix == ".svg":
            mime = "image/svg+xml"
        elif suffix in (".jpg", ".jpeg"):
            mime = "image/jpeg"
        elif suffix == ".webp":
            mime = "image/webp"
        raw = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            return self._serve_file(WEB_ROOT / "index.html")
        if parsed.path == "/api/models":
            return self._send_json(
                200,
                {
                    "models": get_web_model_options(),
                    "tasks": get_web_task_model_defaults(),
                },
            )

        if parsed.path.startswith("/files/"):
            rel = parsed.path[len("/files/") :]
            file_path = (PROJECT_ROOT / rel).resolve()
            if PROJECT_ROOT not in file_path.parents and file_path != PROJECT_ROOT:
                self.send_error(403, "Forbidden")
                return
            return self._serve_file(file_path)

        self.send_error(404, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path not in {"/api/run", "/api/run/stream"}:
            self.send_error(404, "Not found")
            return

        ctype, pdict = cgi.parse_header(self.headers.get("content-type", ""))
        if ctype != "multipart/form-data":
            self._send_json(400, {"error": "Content-Type must be multipart/form-data"})
            return

        form = cgi.FieldStorage(  # nosec B310
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("content-type", ""),
            },
        )

        try:
            request_data = _prepare_run_request(form)
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return

        if parsed.path == "/api/run/stream":
            return self._handle_stream_run(request_data)

        try:
            result = run_main(request_data["inputs"])
        except Exception as exc:
            self._send_json(
                500,
                {
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "hint": "Check API key / network / resvg / model connectivity in your runtime environment.",
                },
            )
            return

        payload = _build_response_payload(result, request_data["enhanced_image_path"])
        self._send_json(200 if payload.get("ok") else 422, payload)

    def _handle_stream_run(self, request_data: dict[str, Any]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        cumulative_result: dict[str, Any] = {}

        def send_event(event_name: str, payload: dict[str, Any]) -> None:
            event_payload = dict(payload)
            result = event_payload.get("result")
            if isinstance(result, dict):
                cumulative_result.update(result)
            _merge_stream_result(cumulative_result, event_name, event_payload)
            if cumulative_result:
                event_payload["snapshot"] = _build_response_payload(cumulative_result, request_data["enhanced_image_path"])
            data = json.dumps(event_payload, ensure_ascii=False)
            message = f"event: {event_name}\ndata: {data}\n\n".encode("utf-8")
            self.wfile.write(message)
            self.wfile.flush()

        send_event("started", {"message": "workflow started"})

        try:
            result = run_main(request_data["inputs"], event_callback=send_event)
        except Exception as exc:
            error_payload = {
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "hint": "Check API key / network / resvg / model connectivity in your runtime environment.",
            }
            send_event("error", error_payload)
            return


def _get_str(form: cgi.FieldStorage, key: str) -> str:
    if key not in form:
        return ""
    value = form[key].value
    if value is None:
        return ""
    return str(value).strip()


def _get_int(form: cgi.FieldStorage, key: str, default: int = 0) -> int:
    raw = _get_str(form, key)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _save_optional_enhanced_image(
    enhanced_image_item: cgi.FieldStorage | None,
    upload_dir: Path,
    case_stem: str,
) -> str | None:
    if enhanced_image_item is None or not getattr(enhanced_image_item, "file", None):
        return None

    filename = Path(getattr(enhanced_image_item, "filename", "") or "").name
    suffix = Path(filename).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        content_type = str(getattr(enhanced_image_item, "type", "") or "").lower()
        if "png" in content_type:
            suffix = ".png"
        elif "jpeg" in content_type or "jpg" in content_type:
            suffix = ".jpg"
        elif "webp" in content_type:
            suffix = ".webp"
        else:
            suffix = ".png"

    safe_stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", case_stem).strip("._") or "upload"
    enhanced_path = upload_dir / f"{safe_stem}_enhanced{suffix}"
    with enhanced_path.open("wb") as f:
        shutil.copyfileobj(enhanced_image_item.file, f)
    return str(enhanced_path)


def _prepare_run_request(form: cgi.FieldStorage) -> dict[str, Any]:
    svg_item = form["svg_file"] if "svg_file" in form else None
    enhanced_image_item = form["enhanced_image"] if "enhanced_image" in form else None
    if svg_item is None or not getattr(svg_item, "file", None):
        raise ValueError("svg_file is required")

    question = _get_str(form, "question")
    update_question = _get_str(form, "update_question")
    qa_question = _get_str(form, "qa_question")
    if question:
        update_question = question
        qa_question = question
    if not update_question and qa_question:
        update_question = qa_question
    if not qa_question and update_question:
        qa_question = update_question
    if not update_question:
        raise ValueError("question is required")

    max_render_retries = _get_int(form, "max_render_retries", default=2)

    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    original_name = Path(getattr(svg_item, "filename", "upload.svg") or "upload.svg").name
    case_stem = Path(original_name).stem
    upload_dir = UPLOAD_ROOT / case_stem
    upload_dir.mkdir(parents=True, exist_ok=True)

    svg_path = upload_dir / f"{case_stem}.svg"
    with svg_path.open("wb") as f:
        shutil.copyfileobj(svg_item.file, f)
    enhanced_image_path = _save_optional_enhanced_image(enhanced_image_item, upload_dir, case_stem)
    model_overrides = _collect_model_overrides(form)

    return {
        "inputs": {
            "web_mode": True,
            "question": update_question,
            "update_question": update_question,
            "qa_question": qa_question,
            "svg_path": str(svg_path),
            "text_spec": None,
            "answer_image_path": enhanced_image_path,
            "max_render_retries": max_render_retries,
            "svg_perception_mode": "llm",
            "svg_update_mode": "llm",
            "model_overrides": model_overrides,
        },
        "enhanced_image_path": enhanced_image_path,
    }


def _build_response_payload(result: dict[str, Any], enhanced_image_path: str | None) -> dict[str, Any]:
    output_image_path = result.get("output_image_path")
    tool_phase = result.get("tool_phase") if isinstance(result, dict) else None
    if isinstance(tool_phase, dict):
        augmented_image_path = str(tool_phase.get("augmented_image_path") or "").strip()
        if augmented_image_path:
            output_image_path = augmented_image_path
    output_image_url = None
    if isinstance(output_image_path, str) and output_image_path:
        output_image_url = "/files/" + output_image_path.lstrip("/")

    answer = result.get("answer", {}) if isinstance(result, dict) else {}
    qa_answer = ""
    model_name = ""
    if isinstance(answer, dict):
        qa_answer = str(answer.get("answer") or "").strip()
        model_name = answer.get("model_name", "")
        if not model_name:
            resolved_task_models = result.get("resolved_task_models", {}) if isinstance(result, dict) else {}
            if isinstance(resolved_task_models, dict):
                model_name = str(resolved_task_models.get("answer") or "").strip()
        if not model_name:
            try:
                from chart_agent.config import get_task_model_config
                model_name = get_task_model_config("answer").model
            except Exception:
                pass

    render_check = result.get("render_check", {}) if isinstance(result, dict) else {}
    business_ok = bool((render_check or {}).get("ok"))
    failure_stage = None
    issues = []
    if isinstance(render_check, dict):
        issues = list(render_check.get("issues", []) or [])
    if render_check:
        if not business_ok:
            if "llm_plan_failed" in issues:
                failure_stage = "llm_plan"
            elif "output image not found" in issues or "no output image path" in issues:
                failure_stage = "render_output"
            elif "image appears empty" in issues:
                failure_stage = "render_empty"
            else:
                failure_stage = "render_validate"

    return {
        "ok": business_ok,
        "failure_stage": failure_stage,
        "issues": issues,
        "answer_issues": (answer.get("issues", []) if isinstance(answer, dict) else []),
        "qa_answer": qa_answer,
        "model_name": model_name,
        "enhanced_image_path": enhanced_image_path,
        "enhanced_image_url": ("/files/" + enhanced_image_path.lstrip("/")) if enhanced_image_path else None,
        "output_image_path": output_image_path,
        "output_image_url": output_image_url,
        "result": result,
    }


def _collect_model_overrides(form: cgi.FieldStorage) -> dict[str, str]:
    task_names = ("splitter", "planner", "executor", "answer", "tool_planner")
    overrides: dict[str, str] = {}
    for task in task_names:
        value = _get_str(form, f"model_{task}")
        if value:
            overrides[task] = value
    return overrides


def _merge_stream_result(cumulative_result: dict[str, Any], event_name: str, payload: dict[str, Any]) -> None:
    if event_name == "step_finished":
        perception_steps = payload.get("perception_steps")
        if isinstance(perception_steps, list):
            cumulative_result["perception_steps"] = perception_steps
        output_image_path = payload.get("output_image_path")
        if isinstance(output_image_path, str) and output_image_path:
            cumulative_result["output_image_path"] = output_image_path


def main() -> None:
    host = os.getenv("CHART_WEB_HOST", "127.0.0.1")
    port = int(os.getenv("CHART_WEB_PORT", "8008"))
    server = ThreadingHTTPServer((host, port), AppHandler)
    print(f"Serving on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
