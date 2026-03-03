from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TraceRecord:
    step: int
    node: str
    action: str
    rationale: str
    inputs_summary: dict[str, Any]
    outputs_summary: dict[str, Any]
    error: str | None = None


def append_trace(
    trace: list[TraceRecord],
    *,
    node: str,
    action: str,
    rationale: str,
    inputs_summary: dict[str, Any],
    outputs_summary: dict[str, Any],
    error: str | None = None,
) -> None:
    record = TraceRecord(
        step=len(trace) + 1,
        node=node,
        action=action,
        rationale=rationale,
        inputs_summary=inputs_summary,
        outputs_summary=outputs_summary,
        error=error,
    )
    trace.append(record)
