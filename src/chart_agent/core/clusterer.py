from __future__ import annotations

import math
import re
from typing import Any


def run_dbscan(
    points: list[tuple[float, float]],
    question: str,
    default_eps: float = 5.0,
    default_min_samples: int = 5,
) -> dict[str, Any]:
    if not points:
        return {
            "clusters": 0,
            "noise": 0,
            "labels": [],
            "eps": default_eps,
            "min_samples": default_min_samples,
        }

    eps = _parse_eps(question) or default_eps
    min_samples = _parse_min_samples(question) or default_min_samples

    labels = _dbscan(points, eps=eps, min_samples=min_samples)
    clusters = len({label for label in labels if label != -1})
    noise = sum(1 for label in labels if label == -1)

    return {
        "clusters": clusters,
        "noise": noise,
        "labels": labels,
        "eps": eps,
        "min_samples": min_samples,
    }


def svg_points_to_data(
    svg_points: list[tuple[float, float]],
    x_ticks: list[tuple[float, float]],
    y_ticks: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    if not svg_points or len(x_ticks) < 2 or len(y_ticks) < 2:
        return []
    data_points = []
    for px, py in svg_points:
        x_val = _pixel_to_data(px, x_ticks)
        y_val = _pixel_to_data(py, y_ticks)
        data_points.append((x_val, y_val))
    return data_points


def _dbscan(points: list[tuple[float, float]], eps: float, min_samples: int) -> list[int]:
    labels = [-1 for _ in points]
    visited = [False for _ in points]
    cluster_id = 0

    for idx in range(len(points)):
        if visited[idx]:
            continue
        visited[idx] = True
        neighbors = _region_query(points, idx, eps)
        if len(neighbors) < min_samples:
            labels[idx] = -1
            continue
        _expand_cluster(points, labels, visited, idx, neighbors, cluster_id, eps, min_samples)
        cluster_id += 1

    return labels


def _expand_cluster(
    points: list[tuple[float, float]],
    labels: list[int],
    visited: list[bool],
    point_idx: int,
    neighbors: list[int],
    cluster_id: int,
    eps: float,
    min_samples: int,
) -> None:
    labels[point_idx] = cluster_id
    i = 0
    while i < len(neighbors):
        n_idx = neighbors[i]
        if not visited[n_idx]:
            visited[n_idx] = True
            n_neighbors = _region_query(points, n_idx, eps)
            if len(n_neighbors) >= min_samples:
                for candidate in n_neighbors:
                    if candidate not in neighbors:
                        neighbors.append(candidate)
        if labels[n_idx] == -1:
            labels[n_idx] = cluster_id
        i += 1


def _region_query(points: list[tuple[float, float]], idx: int, eps: float) -> list[int]:
    px, py = points[idx]
    neighbors = []
    for jdx, (x, y) in enumerate(points):
        if _distance(px, py, x, y) <= eps:
            neighbors.append(jdx)
    return neighbors


def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x1 - x2, y1 - y2)


def _pixel_to_data(pixel: float, ticks: list[tuple[float, float]]) -> float:
    ticks_sorted = sorted(ticks, key=lambda t: t[0])
    for idx in range(len(ticks_sorted) - 1):
        p1, d1 = ticks_sorted[idx]
        p2, d2 = ticks_sorted[idx + 1]
        if min(p1, p2) <= pixel <= max(p1, p2):
            if p2 == p1:
                return d1
            ratio = (pixel - p1) / (p2 - p1)
            return d1 + ratio * (d2 - d1)

    p1, d1 = ticks_sorted[0]
    p2, d2 = ticks_sorted[1]
    if pixel < min(p1, p2):
        if p2 == p1:
            return d1
        ratio = (pixel - p1) / (p2 - p1)
        return d1 + ratio * (d2 - d1)

    p1, d1 = ticks_sorted[-2]
    p2, d2 = ticks_sorted[-1]
    if p2 == p1:
        return d2
    ratio = (pixel - p1) / (p2 - p1)
    return d1 + ratio * (d2 - d1)


def _parse_eps(question: str) -> float | None:
    match = re.search(r"eps\s*=\s*([\d.]+)", question)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def _parse_min_samples(question: str) -> int | None:
    match = re.search(r"min_samples\s*=\s*(\d+)", question)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None
