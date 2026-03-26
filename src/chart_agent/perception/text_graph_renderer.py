from __future__ import annotations

import json
import math
import os
import re
from typing import Any

from chart_agent.prompts.prompt import build_text_graph_parse_prompt

_EDGE_TUPLE_PATTERN = re.compile(r"\(([^()]+)\)")
_NODE_RANGE_PATTERN = re.compile(r"nodes?\s+.*?(\d+)\s*(?:to|-)\s*(\d+)", re.IGNORECASE)
_QUERY_FROM_TO_PATTERN = re.compile(
    r"from\s+node\s+([A-Za-z0-9_\-]+)\s+to\s+node\s+([A-Za-z0-9_\-]+)", re.IGNORECASE
)
_QUERY_BETWEEN_PATTERN = re.compile(
    r"between\s+([A-Za-z0-9_\-]+)\s+and\s+([A-Za-z0-9_\-]+)", re.IGNORECASE
)


def render_graph_from_text(
    text_spec: str,
    question: str,
    llm: Any | None = None,
    output_dir: str | None = None,
    image_size: tuple[int, int] = (640, 420),
) -> tuple[str, dict[str, Any]]:
    combined = f"{text_spec}\n{question}".strip()
    graph = _parse_graph(combined, llm)
    directed = graph["directed"]
    weighted = graph["weighted"]
    nodes = graph["nodes"]
    edges = graph["edges"]
    query = graph.get("query")

    png_path, svg_path = _default_output_paths(output_dir)
    _render_graph_image(
        nodes,
        edges,
        directed=directed,
        weighted=weighted,
        highlight_path=None,
        png_path=png_path,
        svg_path=svg_path,
        image_size=image_size,
    )

    meta = {
        "directed": directed,
        "weighted": weighted,
        "nodes": nodes,
        "edges": edges,
        "query": query,
        "png_path": png_path,
        "svg_path": svg_path,
        "issues": graph.get("issues", []),
    }
    if graph.get("llm_meta"):
        meta["llm_meta"] = graph["llm_meta"]
    return png_path, meta


def _parse_graph(text: str, llm: Any | None) -> dict[str, Any]:
    issues: list[str] = []
    llm_meta = None
    payload = None
    if llm is not None:
        payload = _parse_with_llm(text, llm)
        if payload is None:
            issues.append("llm_parse_failed")
        else:
            llm_meta = payload.pop("llm_meta", None)

    if payload is None:
        payload = _parse_with_regex(text)

    directed = _coerce_bool(payload.get("directed"))
    if directed is None:
        directed = _infer_directed(text)
        issues.append("directed_inferred")

    weighted = _coerce_bool(payload.get("weighted"))
    if weighted is None:
        weighted = _infer_weighted(text, payload.get("edges", []))
        issues.append("weighted_inferred")

    nodes = _collect_nodes(payload.get("nodes"), payload.get("edges", []), text)
    edges = _normalize_edges(payload.get("edges", []), weighted)
    query = payload.get("query") or _parse_query(text)

    if not edges:
        issues.append("no_edges_parsed")
    if not nodes:
        issues.append("no_nodes_parsed")

    return {
        "directed": bool(directed),
        "weighted": bool(weighted),
        "nodes": nodes,
        "edges": edges,
        "query": query,
        "issues": issues,
        "llm_meta": llm_meta,
    }


def _parse_with_llm(text: str, llm: Any) -> dict[str, Any] | None:
    prompt = build_text_graph_parse_prompt(text=text)
    try:
        response = llm.invoke(prompt)
    except Exception:
        return None
    content = getattr(response, "content", "")
    payload = _safe_json_loads(content)
    if not payload:
        return None
    payload["llm_meta"] = {"llm_used": True, "llm_raw": content}
    return payload


def _parse_with_regex(text: str) -> dict[str, Any]:
    edges = []
    for match in _EDGE_TUPLE_PATTERN.findall(text):
        parts = [part.strip() for part in match.split(",")]
        if len(parts) not in (2, 3):
            continue
        source = _normalize_node(parts[0])
        target = _normalize_node(parts[1])
        weight = _safe_float(parts[2]) if len(parts) == 3 else None
        edges.append({"source": source, "target": target, "weight": weight})

    return {
        "directed": None,
        "weighted": None,
        "nodes": [],
        "edges": edges,
        "query": _parse_query(text),
    }


def _infer_directed(text: str) -> bool:
    lower = text.lower()
    if "undirected" in lower:
        return False
    if "directed" in lower:
        return True
    return False


def _infer_weighted(text: str, edges: list[dict[str, Any]]) -> bool:
    lower = text.lower()
    if "unweighted" in lower:
        return False
    if "weighted" in lower or "weight" in lower:
        return True
    for edge in edges:
        if edge.get("weight") is not None:
            return True
    return False


def _collect_nodes(
    nodes: Any, edges: list[dict[str, Any]], text: str
) -> list[str]:
    collected: set[str] = set()
    if isinstance(nodes, list):
        for item in nodes:
            collected.add(_normalize_node(item))

    range_match = _NODE_RANGE_PATTERN.search(text)
    if range_match:
        start = int(range_match.group(1))
        end = int(range_match.group(2))
        for val in range(min(start, end), max(start, end) + 1):
            collected.add(str(val))

    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if source is not None:
            collected.add(_normalize_node(source))
        if target is not None:
            collected.add(_normalize_node(target))

    return _sorted_nodes(collected)


def _normalize_edges(edges: Any, weighted: bool) -> list[dict[str, Any]]:
    if not isinstance(edges, list):
        return []
    normalized = []
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        source = edge.get("source")
        target = edge.get("target")
        if source is None or target is None:
            continue
        norm = {
            "source": _normalize_node(source),
            "target": _normalize_node(target),
            "weight": None,
        }
        if weighted:
            weight = _safe_float(edge.get("weight"))
            norm["weight"] = weight if weight is not None else 1.0
        normalized.append(norm)
    return normalized


def _parse_query(text: str) -> dict[str, str] | None:
    match = _QUERY_FROM_TO_PATTERN.search(text)
    if not match:
        match = _QUERY_BETWEEN_PATTERN.search(text)
    if not match:
        return None
    return {
        "source": _normalize_node(match.group(1)),
        "target": _normalize_node(match.group(2)),
    }


def _normalize_node(value: Any) -> str:
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not value.is_integer():
            return str(value)
        return str(int(value))
    value_str = str(value).strip()
    if value_str.isdigit():
        return str(int(value_str))
    return value_str


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "directed"):
            return True
        if lowered in ("false", "no", "undirected"):
            return False
    return None


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sorted_nodes(nodes: set[str]) -> list[str]:
    return sorted(nodes, key=_node_sort_key)


def _node_sort_key(node: str) -> tuple[int, Any]:
    if node.isdigit():
        return (0, int(node))
    return (1, node)


def _render_graph_image(
    nodes: list[str],
    edges: list[dict[str, Any]],
    *,
    directed: bool,
    weighted: bool,
    highlight_path: list[str] | None,
    png_path: str,
    svg_path: str,
    image_size: tuple[int, int],
) -> None:
    width, height = image_size
    dpi = 100
    fig_width = width / dpi
    fig_height = height / dpi

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    ax.set_aspect("equal")
    ax.axis("off")

    if not nodes:
        nodes = ["0"]
    positions = _circular_layout(nodes)
    highlight_edges = _highlight_edges(highlight_path, directed)

    edge_seen: set[tuple[str, str]] = set()
    for edge in edges:
        u = edge["source"]
        v = edge["target"]
        key = (u, v) if directed else _undirected_key(u, v)
        if key in edge_seen:
            continue
        edge_seen.add(key)
        _draw_edge(
            ax,
            positions[u],
            positions[v],
            directed=directed,
            weighted=weighted,
            weight=edge.get("weight"),
            highlight=(u, v) in highlight_edges,
        )

    xs = [positions[node][0] for node in nodes]
    ys = [positions[node][1] for node in nodes]
    ax.scatter(xs, ys, s=520, c="#4C78A8", edgecolors="#2F4B6E", linewidth=1.6, zorder=3)
    for node in nodes:
        x, y = positions[node]
        ax.text(x, y, node, ha="center", va="center", color="white", fontsize=10, zorder=4)

    padding = 1.4
    ax.set_xlim(-padding, padding)
    ax.set_ylim(-padding, padding)

    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def _circular_layout(nodes: list[str]) -> dict[str, tuple[float, float]]:
    count = len(nodes)
    if count == 1:
        return {nodes[0]: (0.0, 0.0)}
    positions = {}
    for idx, node in enumerate(nodes):
        angle = 2 * math.pi * idx / count
        positions[node] = (math.cos(angle), math.sin(angle))
    return positions


def _highlight_edges(highlight_path: list[str] | None, directed: bool) -> set[tuple[str, str]]:
    if not highlight_path or len(highlight_path) < 2:
        return set()
    edges = set()
    for idx in range(len(highlight_path) - 1):
        u = highlight_path[idx]
        v = highlight_path[idx + 1]
        edges.add((u, v))
        if not directed:
            edges.add((v, u))
    return edges


def _undirected_key(u: str, v: str) -> tuple[str, str]:
    return tuple(sorted((u, v), key=_node_sort_key))


def _draw_edge(
    ax: Any,
    source: tuple[float, float],
    target: tuple[float, float],
    *,
    directed: bool,
    weighted: bool,
    weight: float | None,
    highlight: bool,
) -> None:
    color = "#E45756" if highlight else "#444444"
    line_width = 2.4 if highlight else 1.4
    if directed:
        ax.annotate(
            "",
            xy=target,
            xytext=source,
            arrowprops={"arrowstyle": "->", "color": color, "lw": line_width},
            zorder=1,
        )
    else:
        ax.plot([source[0], target[0]], [source[1], target[1]], color=color, lw=line_width, zorder=1)

    if weighted and weight is not None:
        mid_x = (source[0] + target[0]) / 2
        mid_y = (source[1] + target[1]) / 2
        offset = _edge_label_offset(source, target)
        ax.text(
            mid_x + offset[0],
            mid_y + offset[1],
            f"{weight:g}",
            fontsize=9,
            color=color,
            ha="center",
            va="center",
            zorder=2,
        )


def _edge_label_offset(
    source: tuple[float, float], target: tuple[float, float]
) -> tuple[float, float]:
    dx = target[0] - source[0]
    dy = target[1] - source[1]
    norm = math.hypot(dx, dy)
    if norm == 0:
        return (0.0, 0.0)
    offset_scale = 0.06
    return (-dy / norm * offset_scale, dx / norm * offset_scale)


def _default_output_paths(output_dir: str | None) -> tuple[str, str]:
    base_dir = output_dir or os.path.join("output", "graph")
    png_path = os.path.join(base_dir, "text_graph.png")
    svg_path = os.path.join(base_dir, "text_graph.svg")
    return png_path, svg_path


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
