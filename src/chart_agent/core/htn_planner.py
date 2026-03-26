from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class HtnTask:
    name: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HtnMethod:
    task_name: str
    name: str
    condition: Callable[[HtnTask], bool]
    expand: Callable[[HtnTask], list[HtnTask]]


@dataclass
class HtnPlanResult:
    operators: list[dict[str, Any]] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)


def decompose_tasks(
    root_task: HtnTask,
    *,
    methods: list[HtnMethod],
    operator_names: set[str],
) -> HtnPlanResult:
    result = HtnPlanResult()

    def walk(task: HtnTask, depth: int) -> None:
        for method in methods:
            if method.task_name != task.name:
                continue
            if not method.condition(task):
                continue
            children = method.expand(task)
            result.trace.append(
                {
                    "task": task.name,
                    "method": method.name,
                    "depth": depth,
                    "payload": _summarize_payload(task.payload),
                    "children": [child.name for child in children],
                }
            )
            for child in children:
                walk(child, depth + 1)
            return

        if task.name not in operator_names:
            raise ValueError(f"No HTN method matched task '{task.name}'.")

        operator_payload = dict(task.payload)
        result.trace.append(
            {
                "task": task.name,
                "method": "operator",
                "depth": depth,
                "payload": _summarize_payload(operator_payload),
                "children": [],
            }
        )
        result.operators.append(operator_payload)

    walk(root_task, 0)
    return result


def _summarize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in payload.items():
        if key == "step" and isinstance(value, dict):
            summary[key] = {
                "operation": value.get("operation"),
                "question_hint": value.get("question_hint"),
                "target": value.get("operation_target"),
            }
            continue
        if isinstance(value, list):
            summary[key] = {"type": "list", "size": len(value)}
            continue
        if isinstance(value, dict):
            summary[key] = {"type": "dict", "keys": sorted(value.keys())}
            continue
        summary[key] = value
    return summary
