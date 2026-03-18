from __future__ import annotations

import base64
import json
from pathlib import Path
import re
from typing import Any


def answer_question(
    *,
    qa_question: str,
    chart_type: str,
    data_summary: dict[str, Any],
    output_image_path: str | None,
    image_context_note: str | None = None,
    llm: Any,
) -> dict[str, Any]:
    cluster_result = data_summary.get("cluster_result")
    cluster_params = data_summary.get("cluster_params")
    prompt_text = (
        "Return ONLY valid JSON. Do not include any extra text.\n"
        "Schema: {\"answer\": string, \"confidence\": number between 0 and 1, \"reason\": [string]}\n"
        "Use the provided image to answer the QA question only.\n"
        f"{_image_context_prompt_line(image_context_note)}"
        f"QA Question: {qa_question}\n"
        f"Chart type: {chart_type}\n"
        f"Image path (for reference only): {output_image_path}\n"
        f"{_cluster_prompt_block(cluster_params)}"
    )
    # print(qa_question)

    content = ""
    try:
        response = _invoke_multimodal_or_text(llm, prompt_text, output_image_path)
        content = _coerce_content_to_text(getattr(response, "content", ""))
    except Exception as exc:
        if cluster_result and "cluster" in qa_question.lower():
            return _normalize_answer_payload(
                {
                "answer": f"DBSCAN found {cluster_result.get('clusters')} clusters.",
                "confidence": 0.7,
                "issues": ["llm_call_failed"],
                "dbscan_result": cluster_result,
                "prompt": prompt_text,
                }
            )
        return _normalize_answer_payload(
            {
            "answer": "LLM call failed; unable to answer.",
            "confidence": 0.0,
            "issues": [str(exc)],
            "prompt": prompt_text,
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
        }
        if cluster_result and "cluster" in qa_question.lower():
            fallback["answer"] = f"DBSCAN found {cluster_result.get('clusters')} clusters."
            fallback["confidence"] = 0.7
            fallback["dbscan_result"] = cluster_result
        return _normalize_answer_payload(fallback)

    payload["llm_raw"] = content
    payload["prompt"] = prompt_text
    if cluster_result:
        payload["dbscan_result"] = cluster_result
    return _normalize_answer_payload(payload)


def _cluster_prompt_block(cluster_params: Any) -> str:
    if not isinstance(cluster_params, dict) or not cluster_params:
        return ""
    compact = {key: value for key, value in {
        "algorithm": cluster_params.get("algorithm"),
        "mode": cluster_params.get("mode"),
        "eps": cluster_params.get("eps"),
        "min_samples": cluster_params.get("min_samples"),
        "source": cluster_params.get("source"),
    }.items() if value is not None}
    return (
        "Cluster Counting Rule: "
        "when answering scatter-cluster questions, follow this clustering configuration exactly.\n"
        f"Cluster Parameters: {json.dumps(compact, ensure_ascii=False)}\n"
    )


def _image_context_prompt_line(image_context_note: str | None) -> str:
    note = str(image_context_note or "").strip()
    if not note:
        return ""
    return f"Image context: {note}\n"


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
