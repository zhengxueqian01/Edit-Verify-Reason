from __future__ import annotations

import base64
import json
from pathlib import Path
import re
from typing import Any

ANSWER_SYSTEM_PROMPT = (
    "Return ONLY valid JSON. Do not include any extra text.\n"
    'Schema: {"answer": string, "confidence": number between 0 and 1, "reason": [string]}\n'
    "For cluster-counting questions, follow the clustering rule stated in the question.\n"
    "If the chart distinguishes categories/colors, points of the same category that are connected by enough intermediate "
    "points should be treated as one cluster when that satisfies the DBSCAN conditions in the question."
)


def answer_question(
    *,
    qa_question: str,
    data_summary: dict[str, Any],
    output_image_path: str | None,
    image_context_note: str | None = None,
    llm: Any,
) -> dict[str, Any]:
    system_prompt = _compose_system_prompt(image_context_note)
    prompt_text = f"Input: {qa_question}\n"

    content = ""
    try:
        response = _invoke_multimodal_or_text(
            llm,
            system_prompt=system_prompt,
            user_prompt=prompt_text,
            image_path=output_image_path,
        )
        content = _coerce_content_to_text(getattr(response, "content", ""))
    except Exception as exc:
        return _normalize_answer_payload(
            {
            "answer": "LLM call failed; unable to answer.",
            "confidence": 0.0,
            "issues": [str(exc)],
            "prompt": prompt_text,
            "system_prompt": system_prompt,
            }
        )

    payload = _safe_json_loads(content)
    if not payload:
        fallback = {
            "answer": "Unable to parse LLM response.",
            "confidence": 0.0,
            "issues": ["llm_response_not_json"],
            "llm_raw": content,
            "prompt": prompt_text,
            "system_prompt": system_prompt,
        }
        return _normalize_answer_payload(fallback)

    payload["llm_raw"] = content
    payload["prompt"] = prompt_text
    payload["system_prompt"] = system_prompt
    return _normalize_answer_payload(payload)


def _compose_system_prompt(image_context_note: str | None) -> str:
    note = str(image_context_note or "").strip()
    if not note:
        return ANSWER_SYSTEM_PROMPT
    return f"{ANSWER_SYSTEM_PROMPT}\nImage context: {note}"


def _normalize_answer_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized["confidence"] = _normalize_confidence(normalized.get("confidence"))
    return normalized


def _normalize_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    if confidence < 0.0:
        return 0.0
    if confidence > 1.0:
        return 1.0
    return confidence


def _invoke_multimodal_or_text(
    llm: Any,
    *,
    system_prompt: str,
    user_prompt: str,
    image_path: str | None,
) -> Any:
    data_url = _image_data_url(image_path)
    if data_url:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore

            return llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(
                        content=[
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ]
                    )
                ]
            )
        except Exception:
            pass
    try:
        from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore

        return llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
    except Exception:
        return llm.invoke(f"{system_prompt}\n\n{user_prompt}")


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
