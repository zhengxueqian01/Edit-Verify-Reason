from __future__ import annotations

import json
from typing import Literal
from typing import Any


# 回答阶段：所有回答任务共享的基础 prompt。
ANSWER_GENERAL_SYSTEM_PROMPT = (
    "Return ONLY valid JSON. Do not include any extra text.\n"
    'Schema: {"answer": string, "confidence": number between 0 and 1, "reason": [string]}'
)

# 回答阶段：画图后，散点图聚类问答使用的专用 prompt。
ANSWER_UPDATED_SCATTER_SYSTEM_PROMPT = (
    "For scatter cluster-counting questions, follow the clustering rule stated in the question.\n"
    "If the chart distinguishes categories/colors, first separate points by semantic category/color.\n"
    "Never merge points of different colors/categories into the same cluster, even if they are spatially close.\n"
    "Within each color/category group, judge clusters by spatial proximity, separation, density, and the DBSCAN-style rule stated in the question.\n"
    "Do not treat each color/category as one cluster by default.\n"
    "Points of the same color/category can form one cluster if enough intermediate points connect them under the rule in the question, and they can also form multiple clusters if they are spatially separated."
)

# 回答阶段：画图后，折线图问答使用的专用 prompt。
ANSWER_UPDATED_LINE_SYSTEM_PROMPT = (
    "For line-chart questions, identify series by legend/color/label before reasoning about values.\n"
    "Use the line geometry in the chart to judge crossings, ordering, peaks, troughs, and trends.\n"
    "Do not confuse nearby lines; compare the target lines directly at the relevant x positions or intersections."
)

# 回答阶段：画图后，面积图问答使用的专用 prompt。
ANSWER_UPDATED_AREA_SYSTEM_PROMPT = (
    "For area-chart questions, reason from the filled region boundaries shown in the chart.\n"
    "If the chart is stacked, distinguish between a single area band and the stacked total/top envelope before answering.\n"
    "Use the relevant boundary height, area extent, and relative position to answer comparisons, extrema, and trend questions."
)

# 回答阶段：视觉增强后，散点图问答使用的专用 prompt。
ANSWER_TOOL_AUGMENTED_SCATTER_SYSTEM_PROMPT = (
    "The scatter plot includes a visual enhancement where each point is surrounded by a color halo with radius equals eps/2."
    "This halo reflects a local neighborhood around each point, helping to reveal density and spatial relationships."
)

# 回答阶段：视觉增强后，折线图问答使用的专用 prompt。
ANSWER_TOOL_AUGMENTED_LINE_SYSTEM_PROMPT = (
    "The chart highlights the lines relevant to the question to help you focus on their relationships."
    "Use these visual cues to analyze how the selected lines interact."
    "Important: The chart does not explicitly mark intersection points. You must determine the answer through careful reasoning."
)

# 回答阶段：视觉增强后，面积图问答使用的专用 prompt。
ANSWER_TOOL_AUGMENTED_AREA_SYSTEM_PROMPT = (
    "The chart includes a highlighted top boundary that reflects the overall aggregated trend. "
    "Use this visual cue to better understand how the total values evolve over time."
    "Important: The highlighted curve does not indicate the maximum point explicitly. You must determine the answer by analyzing the chart."
)

# 回答阶段：基于“已更新图像”作答时附加的上下文说明。
ANSWER_UPDATED_IMAGE_CONTEXT_PROMPT = (
    "The requested chart update has already been applied to this image. "
    "Answer the QA question only based on the updated chart."
)

# 回答阶段：基于“已更新且做过视觉增强的图像”作答时附加的上下文说明。
ANSWER_TOOL_AUGMENTED_IMAGE_CONTEXT_PROMPT = (
    "The requested chart update has already been applied, and visual augmentation has also been added to help reasoning. "
    "Use the enhancement only as a visual aid and answer the QA question based on this updated chart."
)


def build_answer_system_prompt(
    *,
    qa_question: str,
    data_summary: dict[str, Any] | None = None,
    chart_type: str | None = None,
    answer_stage: Literal["original", "updated", "tool_augmented"] = "updated",
    image_context_note: str | None = None,
) -> str:
    # 回答阶段的 prompt 组装器，会按“阶段 + 图表类型”动态拼接专用规则。
    parts = [ANSWER_GENERAL_SYSTEM_PROMPT]
    resolved_chart_type = _resolve_answer_chart_type(chart_type=chart_type, data_summary=data_summary)
    task_prompt = _select_answer_task_prompt(
        qa_question=qa_question,
        data_summary=data_summary,
        chart_type=resolved_chart_type,
        answer_stage=answer_stage,
    )
    if task_prompt:
        parts.append(task_prompt)
    note = str(image_context_note or "").strip()
    if note:
        parts.append(f"Image context: {note}")
    reasoning_context = _build_unexecuted_update_reasoning_prompt(data_summary)
    if reasoning_context:
        parts.append(reasoning_context)
    return "\n".join(parts)


def _build_unexecuted_update_reasoning_prompt(data_summary: dict[str, Any] | None) -> str:
    if not isinstance(data_summary, dict):
        return ""
    if str(data_summary.get("ablation_mode") or "").strip() != "wo_svg_update":
        return ""
    context = data_summary.get("unexecuted_update_reasoning")
    if not isinstance(context, dict):
        return ""
    steps = context.get("steps")
    if not isinstance(steps, list):
        steps = []
    payload = {
        "normalized_question": str(context.get("normalized_question") or "").strip(),
        "llm_success": bool(context.get("llm_success")),
        "steps": [step for step in steps if isinstance(step, dict)],
    }
    return (
        "Planned update reasoning and operation steps were generated but not executed on the image.\n"
        f"Use this unexecuted update context when answering: {json.dumps(payload, ensure_ascii=False)}"
    )


def build_update_plan_prompt(*, question: str, chart_type: str, retry_hint: str, new_points_schema: str) -> str:
    # 更新规划阶段：把编辑请求规范化，并拆成可执行步骤。
    return (
        "You are planning chart-edit operations.\n"
        f"Chart type: {chart_type}\n"
        "Return JSON only with keys:\n"
        "- normalized_question: concise imperative update instruction in English\n"
        "- steps: array of step objects in execution order; each step has operation and optional question_hint, operation_target, data_change, and new_points fields\n"
        f"- new_points: {new_points_schema}\n"
        f"- retry_hint: {retry_hint or 'none'}\n"
        "Rules:\n"
        "- Do not rewrite or summarize structured data payloads.\n"
        "- question_hint is only a short execution hint, not the source of truth for values.\n"
        "- If the input includes structured operation target or data change, preserve them at step level instead of collapsing them into prose.\n"
        "- For scatter add, if point colors are provided in the question/data payload, copy them through to each new_points item.\n"
        "- For multi-operation requests, steps must follow the operation order stated in the input text. Do not reorder operations unless the input explicitly requires it.\n"
        "- If one operation expands into multiple substeps, keep those substeps grouped inside that operation block and preserve the block order from the input text.\n"
        "Question:\n"
        f"{question}"
    )


def build_svg_intent_plan_prompt(
    *,
    operation_text: str,
    chart_type: str,
    structured_context_summary: dict[str, Any],
    perception_summary: dict[str, Any],
) -> str:
    # SVG 意图规划阶段：把操作文本绑定到具体 SVG 编辑目标。
    return (
        "You are planning SVG edit intent for a chart update.\n"
        "Return JSON only with key: steps.\n"
        "- steps: array of step objects in execution order.\n"
        "- each step may contain: operation, question_hint, operation_target, data_change, new_points.\n"
        "- operation must be one of add|delete|change.\n"
        "- Decide the concrete edit target from the operation text and SVG summary.\n"
        "- Keep structured payloads in operation_target/data_change instead of prose when possible.\n"
        "- For multi-operation requests, steps must follow the operation order stated in the operation text. Do not reorder operations unless the text explicitly requires it.\n"
        "- If one operation expands into multiple substeps, keep those substeps grouped inside that operation block and preserve the block order from the operation text.\n"
        "- Do not output explanations.\n"
        f"Chart type: {chart_type}\n"
        f"Operation text: {operation_text}\n"
        f"Structured context: {json.dumps(structured_context_summary, ensure_ascii=False)}\n"
        f"SVG summary: {json.dumps(perception_summary, ensure_ascii=False)}"
    )


def build_visual_tool_planner_prompt(
    *,
    question: str,
    chart_type: str,
    data_summary: dict[str, Any],
    canvas_width: float,
    canvas_height: float,
    tool_specs: list[dict[str, Any]],
) -> str:
    # 视觉工具规划阶段：决定回答前要添加哪些图表增强标记。
    return (
        "You can use visual markup tools on a chart image to improve answer reliability.\n"
        "Important: chart updates have already been applied to this SVG.\n"
        "Do NOT re-apply edits and do NOT add factual conclusions onto the chart.\n"
        "Only add light visual guides for answering the QA question.\n"
        "You must choose tools according to the chart type and the QA task.\n"
        "First, explicitly state your understanding of the QA question.\n"
        "Return JSON only with schema:\n"
        "{\"qa_understanding\":string,\"tool_calls\":[{\"tool\":string,\"args\":object}],\"notes\":string}\n"
        "Rules:\n"
        "- Use only tools listed below.\n"
        "- Use SVG coordinates only.\n"
        f"- Canvas size is width={canvas_width:.2f}, height={canvas_height:.2f}.\n"
        "- Tool selection by chart type:\n"
        "  * scatter: default to isolate_all_color_topologies to add same-color background halos for all points; use isolate_color_topology only when a single target color is explicitly required; use add_point/draw_line/highlight_rect as light local guides when needed.\n"
        "  * line: for intersection/counting questions, first use isolate_target_lines to fade unrelated lines, then use zoom_and_highlight_intersection to mark crossings; use add_point/draw_line/highlight_rect as light local guides when needed.\n"
        "  * area: use highlight_top_boundary when the task benefits from tracing the upper envelope; do not use add_point, draw_line, or highlight_rect.\n"
        "  * unknown/other: do not use scatter-only, line-only, or area-only tools unless the visual structure truly matches that tool.\n"
        "- Cross-type restrictions:\n"
        "  * Do not use isolate_color_topology or isolate_all_color_topologies unless the chart is a scatter chart or scatter-like point cloud.\n"
        "  * Do not use isolate_target_lines or zoom_and_highlight_intersection unless the chart contains line series and the task is about line crossings/intersections.\n"
        "  * Do not use highlight_top_boundary unless the chart is an area chart.\n"
        "  * Do not use add_point, draw_line, or highlight_rect on area charts.\n"
        "- Keep overlays minimal and non-occluding.\n"
        "- Prefer 1-3 calls; use 4+ only if the question truly requires multiple marks.\n"
        "- Never add duplicate or near-duplicate marks.\n"
        "- Avoid full-chart boxes or long explanatory text.\n"
        "- If no visual enhancement is needed, return empty tool_calls.\n"
        "- Max 6 tool calls.\n"
        f"Question: {question}\n"
        f"Chart type: {chart_type}\n"
        f"Data summary: {json.dumps(data_summary, ensure_ascii=False)}\n"
        f"Tools: {json.dumps(tool_specs, ensure_ascii=False)}\n"
    )


def build_svg_summary_perception_prompt(*, question: str, svg_summary: dict[str, Any]) -> str:
    # SVG 感知阶段：根据结构化 SVG 摘要推断图表类型。
    return (
        "You are perceiving an SVG chart from a compact structured summary. "
        "Infer the chart type from the summary instead of relying on the user question alone. "
        "Return JSON only with key: chart_type. "
        "chart_type must be one of: scatter, line, area, graph, unknown. "
        f"\nQuestion: {question}\nSVG Summary: {json.dumps(svg_summary, ensure_ascii=True)}"
    )


def build_question_parser_prompt(*, question: str) -> str:
    # 问题解析阶段：从原始文本中提取结构化的散点更新信息。
    return (
        "You are parsing a chart-update request. "
        "Identify any explicit numeric updates in the question. "
        "Return JSON only with keys: new_points (list of {x,y}), point_color (string), issues (list). "
        "If the chart type is unclear or not scatter, leave new_points empty and add a note in issues."
        f"\nQuestion: {question}"
    )


def build_render_validator_prompt(
    *,
    chart_type: str,
    update_spec: dict[str, Any],
    image_path: str,
    width: int,
    height: int,
    non_empty: bool,
) -> str:
    # 渲染校验阶段：检查渲染结果是否反映了请求的更新。
    return (
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


def build_line_update_parse_prompt(*, question: str) -> str:
    # 折线图更新解析任务：从文本中提取按顺序排列的序列数值。
    return (
        "You are parsing a line chart update. "
        "Extract the new series values in order. "
        "Return JSON only with keys: values (list of numbers)."
        f"\nQuestion: {question}"
    )


def build_area_update_parse_prompt(*, question: str) -> str:
    # 面积图更新解析任务：从文本中提取按顺序排列的序列数值。
    return (
        "You are parsing a stacked area chart update. "
        "Extract the new series values in order. "
        "Return JSON only with keys: values (list of numbers)."
        f"\nQuestion: {question}"
    )


def build_text_line_parse_prompt(*, text_spec: str) -> str:
    # 文本转折线图任务：从文本规格中解析年份和序列。
    return (
        "Return ONLY valid JSON. Do not include any extra text.\n"
        "Schema: {\"years\": [year,...], \"series\": [{\"label\": string, \"values\": [number, ...]}]}\n"
        "If labels are missing, use \"Series 1\", \"Series 2\", etc.\n"
        "If years are present in the text, include them in the years field in order.\n"
        f"Text: {text_spec}"
    )


def build_text_graph_parse_prompt(*, text: str) -> str:
    # 文本转图结构任务：从文本规格中解析图结构和查询目标。
    return (
        "Return ONLY valid JSON. Do not include any extra text.\n"
        "Schema: {"
        '"directed": bool, "weighted": bool, '
        '"nodes": [string|number], '
        '"edges": [{"source": string|number, "target": string|number, "weight": number?}], '
        '"query": {"source": string|number, "target": string|number}'
        "}\n"
        "If the graph is unweighted, omit weight or set it to 1.\n"
        f"Text: {text}"
    )


def _is_scatter_cluster_question(
    *,
    qa_question: str,
    data_summary: dict[str, Any] | None,
    chart_type: str | None,
) -> bool:
    # 回答阶段的任务分流：只有散点聚类问答会追加那段专用聚类规则。
    resolved_chart_type = _resolve_answer_chart_type(chart_type=chart_type, data_summary=data_summary)
    if resolved_chart_type != "scatter":
        return False
    lowered = str(qa_question or "").lower()
    return any(token in lowered for token in ("cluster", "clusters", "聚类", "簇"))


def _select_answer_task_prompt(
    *,
    qa_question: str,
    data_summary: dict[str, Any] | None,
    chart_type: str,
    answer_stage: Literal["original", "updated", "tool_augmented"],
) -> str:
    # 回答阶段：根据“阶段 + 图表类型”选择专用 prompt。
    if answer_stage == "original":
        return ""
    if chart_type == "scatter":
        updated_prompt = ANSWER_UPDATED_SCATTER_SYSTEM_PROMPT
        if answer_stage == "tool_augmented":
            return "\n".join([updated_prompt, ANSWER_TOOL_AUGMENTED_SCATTER_SYSTEM_PROMPT])
        if _is_scatter_cluster_question(
            qa_question=qa_question,
            data_summary=data_summary,
            chart_type=chart_type,
        ):
            return updated_prompt
        return updated_prompt
    if chart_type == "line":
        if answer_stage == "tool_augmented":
            return "\n".join([ANSWER_UPDATED_LINE_SYSTEM_PROMPT, ANSWER_TOOL_AUGMENTED_LINE_SYSTEM_PROMPT])
        return ANSWER_UPDATED_LINE_SYSTEM_PROMPT
    if chart_type == "area":
        if answer_stage == "tool_augmented":
            return "\n".join([ANSWER_UPDATED_AREA_SYSTEM_PROMPT, ANSWER_TOOL_AUGMENTED_AREA_SYSTEM_PROMPT])
        return ANSWER_UPDATED_AREA_SYSTEM_PROMPT
    return ""


def _resolve_answer_chart_type(*, chart_type: str | None, data_summary: dict[str, Any] | None) -> str:
    # 回答阶段的图表类型推断：优先显式 chart_type，其次回退到摘要统计。
    resolved_chart_type = str(chart_type or "").strip().lower()
    if not resolved_chart_type and isinstance(data_summary, dict):
        mapping_info_summary = data_summary.get("mapping_info_summary")
        if isinstance(mapping_info_summary, dict):
            num_points = mapping_info_summary.get("num_points")
            num_lines = mapping_info_summary.get("num_lines")
            num_areas = mapping_info_summary.get("num_areas")
            try:
                if int(num_points or 0) > 0 and int(num_lines or 0) == 0 and int(num_areas or 0) == 0:
                    resolved_chart_type = "scatter"
                elif int(num_lines or 0) > 0 and int(num_points or 0) == 0 and int(num_areas or 0) == 0:
                    resolved_chart_type = "line"
                elif int(num_areas or 0) > 0 and int(num_points or 0) == 0:
                    resolved_chart_type = "area"
            except (TypeError, ValueError):
                resolved_chart_type = resolved_chart_type or ""
    return resolved_chart_type
