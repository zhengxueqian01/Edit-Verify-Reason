from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from chart_agent.config import ModelConfig, get_model_config


@dataclass
class MockResponse:
    content: str


class MockLLM:
    def invoke(self, prompt: str | dict[str, Any]) -> MockResponse:
        payload = {
            "chart_type": "unknown",
            "confidence": 0.3,
            "parsed_data": {},
            "rationale": "MockLLM fallback response.",
        }
        return MockResponse(content=json.dumps(payload))


def make_llm(model_config: ModelConfig | None = None) -> Any:
    config = model_config or get_model_config()
    if not config.api_key:
        return MockLLM()

    try:
        from langchain_openai import ChatOpenAI
    except Exception as exc:  # pragma: no cover - import guard
        raise ImportError(
            "langchain_openai is required for OpenAI-compatible LLM usage."
        ) from exc

    return ChatOpenAI(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=config.temperature,
    )
