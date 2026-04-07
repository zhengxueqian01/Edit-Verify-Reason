from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from chart_agent.core.trace import TraceRecord


@dataclass
class Budget:
    max_retries: int
    retry_count: int = 0


@dataclass
class ErrorState:
    last_error: str | None = None
    error_history: list[str] = field(default_factory=list)


@dataclass
class GraphState:
    inputs: dict[str, Any]
    perception: dict[str, Any] = field(default_factory=dict)
    budget: Budget = field(default_factory=lambda: Budget(max_retries=2))
    trace: list[TraceRecord] = field(default_factory=list)
    errors: ErrorState = field(default_factory=ErrorState)
    current_action: str | None = None
